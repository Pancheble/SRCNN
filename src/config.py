import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters and configuration
config = {
    'n1': 128,               # First layer filters
    'n2': 64,                # Second layer filters
    'n3': 3,                 # Output layer filters
    'f1': 9,                 # First layer kernel size
    'f2': 3,                 # Second layer kernel size
    'f3': 5,                 # Third layer kernel size
    'upscale_factor': 3,     # Upscale factor for Super-Resolution
    'input_size': 33,        # Size of input image patches
    'stride': 14,            # Stride used in generating sub-images
    'batch_size': 128,       # Number of samples per batch
    'epochs': 200,             # Number of training epochs
    'lr': 1e-3,              # Learning rate
    'img_paths': r"C:\Users\User\Desktop\SRCNN\data\T91",
    'save_path': r"C:\Users\User\Desktop\SRCNN\result\torch_SRCNN_200EPOCHS.pth",
    'hr_img_path': r"C:\Users\User\Desktop\SRCNN\data\Set5\butterfly.png"
}
