import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SRCNN

# Function to load an image, preprocess it, and apply the SRCNN model
def load_and_preprocess_image(image_path, upscale_factor):
    hr_img = cv2.imread(image_path)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    hr_img = hr_img.astype(np.float32) / 255.0  # Normalize the image

    # Generate low-resolution image (bicubic downsample)
    temp_img = cv2.resize(hr_img, dsize=(0, 0), fx=1/upscale_factor, fy=1/upscale_factor, interpolation=cv2.INTER_AREA)
    bicubic_img = cv2.resize(temp_img, dsize=(0, 0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    return hr_img, bicubic_img

# Function to visualize the high-res, bicubic, and SRCNN output images
def visualize_results(lr_image, sr_image, hr_image):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [hr_image, lr_image, sr_image]
    titles = ['High-Resolution', 'Low-Resolution', 'Super-Resolution']

    for i, ax in enumerate(axes):
        ax.imshow(np.clip(images[i], 0, 1))  # Clip values to ensure proper visualization
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model from the specified path
    model_path = r"C:\Users\User\Desktop\SRCNN\result\torch_SRCNN_200EPOCHS.pth"
    model = SRCNN(kernel_list=[9, 3, 5], filters_list=[128, 64, 3]).to(device)  # Replace kernel_list and filters_list if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and preprocess the high-resolution test image
    test_image_path = r"C:\Users\User\Desktop\SRCNN\data\Set5\butterfly.png"  # Set the path to your test image
    hr_img, bicubic_img = load_and_preprocess_image(test_image_path, upscale_factor=3)

    # Prepare the input image for the SRCNN model
    input_img = bicubic_img.transpose((2, 0, 1))  # Change from (H, W, C) to (C, H, W)
    input_img = torch.tensor(input_img).unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

    # Pass through the model to get the super-resolution image
    with torch.no_grad():
        srcnn_img = model(input_img).squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert back to (H, W, C)

    # Visualize the results: low-res, super-res, and high-res images
    visualize_results(bicubic_img, srcnn_img, hr_img)

if __name__ == '__main__':
    main()
    
    