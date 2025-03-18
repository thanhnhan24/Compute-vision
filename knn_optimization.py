import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# ==== ĐỊNH NGHĨA ĐƯỜNG DẪN ====
extract_path = r'SIFT-KNN-MK2\HAND-CLASSIFICATION.v1i.folder'
train_path = os.path.join(extract_path, "train")
valid_path = os.path.join(extract_path, "valid")

# ==== TRÍCH XUẤT ĐẶC TRƯNG HOG ====
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))  # Resize ảnh về 64x64
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      orientations=9, block_norm='L2-Hys', visualize=True)
    return features

# HÀM ĐỌC DỮ LIỆU TỪ THƯ MỤC
def load_data_from_folder(folder_path):
    X, y = [], []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                features = extract_hog_features(img_path)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Đọc dữ liệu từ tập train và valid
X_train, y_train = load_data_from_folder(train_path)
X_val, y_val = load_data_from_folder(valid_path)

# Mã hóa nhãn thành số
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# ==== TỐI ƯU HÓA RANDOM FOREST ====
param_grid = {
    'n_estimators': np.arange(50, 301, 50),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

print("\n🔍 Đang tìm kiếm siêu tham số tối ưu...")
random_search.fit(X_train, y_train_encoded)
print("✅ Hoàn thành tối ưu hóa!")
print("\n⭐ Siêu tham số tối ưu:")
print(random_search.best_params_)

# ==== ĐÁNH GIÁ MÔ HÌNH ====
best_rf = random_search.best_estimator_
accuracy = best_rf.score(X_val, y_val_encoded)
print(f"\n🎯 Độ chính xác trên tập kiểm tra: {accuracy:.4f}")