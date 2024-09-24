import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import Dataset, DataLoader

import cv2
import os
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

# Hyperparameters and configuration variables
n1, n2, n3 = 128, 64, 3  # Number of filters for each layer
f1, f2, f3 = 9, 3, 5     # Filter sizes for each convolution layer
upscale_factor = 3       # Upscale factor for Super-Resolution

input_size = 33          # Size of input image patches
output_size = input_size - f1 - f2 - f3 + 3  # Size of output after convolutions

stride = 14              # Stride used in generating sub-images
batch_size = 128         # Number of samples per batch
epochs = 5               # Number of training epochs



path = r"C:\Users\User\Desktop\SRCNN\T91"
save_path = r"C:\Users\User\Desktop\SRCNN\torch_SRCNN_200EPOCHS.pth"
img_paths = r"C:/Users/User/Desktop/SRCNN/T91"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device



# Custom dataset class for loading and processing images
class CustomDataset(Dataset):
    def __init__(self, img_paths, input_size, output_size, stride=14, upscale_factor=3):
        super(CustomDataset, self).__init__()
        
        # Load all image paths from the directory
        self.img_paths = glob.glob(img_paths + '/' + '*.png')
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.sub_lr_imgs = []  # Placeholder for low-resolution patches
        self.sub_hr_imgs = []  # Placeholder for high-resolution patches
        self.input_size = input_size
        self.output_size = output_size
        self.pad = abs(self.input_size - self.output_size) // 2  # Padding to adjust output size

        print("Start {} Images Pre-Processing".format(len(self.img_paths)))
        
        # Loop over all images and process them
        for img_path in self.img_paths:
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

            # Ensure the image dimensions are divisible by upscale factor
            h = img.shape[0] - np.mod(img.shape[0], self.upscale_factor)
            w = img.shape[1] - np.mod(img.shape[1], self.upscale_factor)
            img = img[:h, :w, :]

            # Normalize the image
            label = img.astype(np.float32) / 255.0
            
            # Downscale image (low-resolution) and then upscale it back
            temp_input = cv2.resize(label, dsize=(0,0), fx=1/self.upscale_factor, fy=1/self.upscale_factor,
                                    interpolation=cv2.INTER_AREA)
            input = cv2.resize(temp_input, dsize=(0,0), fx=self.upscale_factor, fy=self.upscale_factor,
                            interpolation=cv2.INTER_CUBIC)

            # Extract patches from both the high-resolution and low-resolution images
            for h in range(0, input.shape[0] - self.input_size + 1, self.stride):
                for w in range(0, input.shape[1] - self.input_size + 1, self.stride):
                    sub_lr_img = input[h:h+self.input_size, w:w+self.input_size, :]  # Low-res patch
                    sub_hr_img = label[h+self.pad:h+self.pad+self.output_size, w+self.pad:w+self.pad+self.output_size, :]  # High-res patch

                    # Rearrange the image channels for PyTorch (channels, height, width)
                    sub_lr_img = sub_lr_img.transpose((2, 0, 1))
                    sub_hr_img = sub_hr_img.transpose((2, 0, 1))

                    # Append the patches to their respective lists
                    self.sub_lr_imgs.append(sub_lr_img)
                    self.sub_hr_imgs.append(sub_hr_img)

        print("Finish, Created {} Sub-Images".format(len(self.sub_lr_imgs)))

        # Convert lists to numpy arrays
        self.sub_lr_imgs = np.asarray(self.sub_lr_imgs)
        self.sub_hr_imgs = np.asarray(self.sub_hr_imgs)

    # Return the number of sub-images
    def __len__(self):
        return len(self.sub_lr_imgs)

    # Return a specific low-res and high-res image patch
    def __getitem__(self, idx):
        lr_img = self.sub_lr_imgs[idx]
        hr_img = self.sub_hr_imgs[idx]
        return lr_img, hr_img



train_dataset = CustomDataset(img_paths=img_paths, 
                              input_size=input_size, 
                              output_size=output_size, 
                              stride=stride, 
                              upscale_factor=upscale_factor)

train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0, 
                              pin_memory=True)



# Define the Super-Resolution CNN model (SRCNN)
class SRCNN(nn.Module):
    def __init__(self, kernel_list, filters_list, num_channels=3):
        super(SRCNN, self).__init__()

        # Unpack kernel sizes and filter numbers
        f1, f2, f3 = kernel_list
        n1, n2, n3 = filters_list
        
        # Define the 3 convolutional layers
        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2)
        self.conv3 = nn.Conv2d(n2, num_channels, kernel_size=f3)
        self.relu = nn.ReLU(inplace=True)

        # Xavier initialization for weights
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)

        # Zero initialization for biases
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)

    # Define the forward pass
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Function to train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Loop through each batch of data
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Function to test the model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            # Forward pass (without backprop)
            pred = model(X)
            test_loss += loss_fn(pred, y)
    
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

# Instantiate the SRCNN model
model = SRCNN(kernel_list=[f1, f2, f3], filters_list=[n1, n2, n3]).to(device)
print(model)

# Optimizer and loss function setup
params = model.parameters()
optimizer = optim.Adam(params=params, lr=1e-3)
loss_fn = nn.MSELoss()



for i in range(epochs):
    print(f"{i+1} Epochs ... ")
    model.train()
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

# Save the trained model
torch.save(model.state_dict(), save_path)

# Path to the high-resolution test image
hr_img_path = r"C:\Users\User\Desktop\SRCNN\Set5\butterfly.png"

# Load and preprocess the test image
hr_img = cv2.imread(hr_img_path)
hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
print("img shape: {}".format(hr_img.shape))

# Normalize the high-resolution image and create a bicubic low-resolution image
hr_img = hr_img.astype(np.float32) / 255.0
temp_img = cv2.resize(hr_img, dsize=(0,0), fx=1/upscale_factor, fy=1/upscale_factor, interpolation=cv2.INTER_AREA)
bicubic_img = cv2.resize(temp_img, dsize=(0,0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

# Evaluate the model on the low-resolution image
model.eval()
input_img = bicubic_img.transpose((2,0,1))  # Rearrange channels for PyTorch
input_img = torch.tensor(input_img).unsqueeze(0).to(device)

with torch.no_grad():
    srcnn_img = model(input_img)

srcnn_img = srcnn_img.squeeze().cpu().numpy().transpose((1,2,0))

fig, axes = plt.subplots(1, 3, figsize=(10,5))

axes[0].imshow(hr_img)
axes[1].imshow(bicubic_img)
axes[2].imshow(np.squeeze(srcnn_img))

axes[0].set_title('hr_img')
axes[1].set_title('bicubic_img')
axes[2].set_title('srcnn_img')

plt.tight_layout()
plt.show()