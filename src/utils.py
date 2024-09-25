import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(lr_image, sr_image, hr_image):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [lr_image, sr_image, hr_image]
    titles = ['Low-Resolution', 'Super-Resolution', 'High-Resolution']
    
    for i, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(np.transpose(images[i], (1, 2, 0)), cv2.COLOR_BGR2RGB))
        ax.set_title(titles[i])
        ax.axis('off')
    plt.show()