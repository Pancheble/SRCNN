import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_paths, input_size, output_size, stride=14, upscale_factor=3):
        self.img_paths = glob.glob(img_paths + '/' + '*.png')
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.input_size = input_size
        self.output_size = output_size
        self.pad = abs(self.input_size - self.output_size) // 2
        self.sub_lr_imgs, self.sub_hr_imgs = self._process_images()

    def _process_images(self):
        sub_lr_imgs, sub_hr_imgs = [], []
        for img_path in self.img_paths:
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            img = self._resize_to_modulo(img, self.upscale_factor)
            label = img.astype(np.float32) / 255.0

            input_img = self._create_low_res_image(label)
            sub_lr, sub_hr = self._extract_patches(input_img, label)
            sub_lr_imgs.extend(sub_lr)
            sub_hr_imgs.extend(sub_hr)
        return np.array(sub_lr_imgs), np.array(sub_hr_imgs)

    def _resize_to_modulo(self, img, factor):
        h = img.shape[0] - np.mod(img.shape[0], factor)
        w = img.shape[1] - np.mod(img.shape[1], factor)
        return img[:h, :w, :]

    def _create_low_res_image(self, label):
        temp_input = cv2.resize(label, dsize=(0, 0), fx=1/self.upscale_factor, fy=1/self.upscale_factor, interpolation=cv2.INTER_AREA)
        return cv2.resize(temp_input, dsize=(0, 0), fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_CUBIC)

    def _extract_patches(self, input_img, label_img):
        sub_lr_imgs, sub_hr_imgs = [], []
        for h in range(0, input_img.shape[0] - self.input_size + 1, self.stride):
            for w in range(0, input_img.shape[1] - self.input_size + 1, self.stride):
                sub_lr = input_img[h:h+self.input_size, w:w+self.input_size, :].transpose((2, 0, 1))
                sub_hr = label_img[h+self.pad:h+self.pad+self.output_size, w+self.pad:w+self.pad+self.output_size, :].transpose((2, 0, 1))
                sub_lr_imgs.append(sub_lr)
                sub_hr_imgs.append(sub_hr)
        return sub_lr_imgs, sub_hr_imgs

    def __len__(self):
        return len(self.sub_lr_imgs)

    def __getitem__(self, idx):
        return self.sub_lr_imgs[idx], self.sub_hr_imgs[idx]