import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_sample_plot():
    folder_path = os.path.join("Mango_Dataset", "Partially_Ripe")
    
    if not os.path.exists(folder_path):
        print("Error: Dataset folder not found.")
        return
        
    sample_file = os.listdir(folder_path)[0]
    img_path = os.path.join(folder_path, sample_file)
    
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (224, 224))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Gaussian Blur')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cleaned, cmap='gray')
    plt.title('Morphological Closing')
    plt.axis('off')
    
    plt.suptitle(f'Sample image: {sample_file}')
    plt.tight_layout()
    plt.savefig('figure_3_3_sample_output.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_sample_plot()