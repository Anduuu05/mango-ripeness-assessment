import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    resized = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    glcm = graycomatrix(cleaned, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    features = [contrast, correlation, energy, homogeneity]
    return features, (resized, blurred, cleaned)

def main():
    dataset_path = "Mango_Dataset"
    classes = ["Unripe", "Partially_Ripe", "Ripe"]
    
    X = []
    y = []
    sample_images = {}
    
    print("Extracting features from dataset...")
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, file_name)
                features, processed_imgs = extract_glcm_features(img_path)
                
                if features is not None:
                    X.append(features)
                    y.append(label_idx)
                    
                    if class_name not in sample_images:
                        sample_images[class_name] = processed_imgs
                        
    X = np.array(X)
    y = np.array(y)
    
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    
    print(f"Total processed images: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print("Training Random Forest classifier...")
    # 这里换成了 Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    print("\n--- Classification Results ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix for Surface Defect Handling Pipeline')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_limyujie.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()