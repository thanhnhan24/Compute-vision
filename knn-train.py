import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
from tqdm import tqdm
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# --- 1. Giải nén dữ liệu ---
data_path = r'HAND-CLASSIFICATION.v4i.folder'

train_path = os.path.join(data_path, "train")
valid_path = os.path.join(data_path, "valid")
test_path = os.path.join(data_path, "test")

# --- 2. Hàm trích xuất đặc trưng HOG từ ảnh ---
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh mức xám
    image = cv2.resize(image, (640, 640))  # Resize về 64x64
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                      orientations=9, block_norm='L2-Hys', visualize=True)
    return features

# --- 3. Chuẩn bị dữ liệu ---
def prepare_dataset(dataset_path):
    data, labels = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
                image_path = os.path.join(label_path, image_name)
                try:
                    features = extract_hog_features(image_path)
                    data.append(features)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return np.array(data), np.array(labels)

# Load dữ liệu train và test
X_train, y_train = prepare_dataset(train_path)
X_valid, y_valid = prepare_dataset(valid_path)
X_test, y_test = prepare_dataset(test_path)

# --- 4. Huấn luyện mô hình Random Forest ---
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth=50)
rf_model.fit(X_train, y_train)

# # Lưu mô hình đã huấn luyện
model_path = f"hand_gesture_rf_model.pkl-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")

# # --- 5. Dự đoán và đánh giá ---
y_pred = rf_model.predict(X_valid)  # Dự đoán trên tập validation
accuracy = accuracy_score(y_valid, y_pred)  # So sánh với y_valid
report = classification_report(y_valid, y_pred, output_dict=True)  # So sánh với y_valid
conf_matrix = confusion_matrix(y_valid, y_pred)  # So sánh với y_valid

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_valid, y_pred))

# --- 6. Trực quan hóa Confusion Matrix và Metrics ---
def plot_metrics_and_conf_matrix(cm, report, labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")
    
    # Accuracy & Precision per Class
    class_labels = list(report.keys())[:-3]  # Loại bỏ 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[label]['precision'] for label in class_labels]
    recall = [report[label]['recall'] for label in class_labels]
    f1_score = [report[label]['f1-score'] for label in class_labels]
    
    x = np.arange(len(class_labels))
    width = 0.3
    
    axes[1].bar(x - width, precision, width=width, label='Precision')
    axes[1].bar(x, recall, width=width, label='Recall')
    axes[1].bar(x + width, f1_score, width=width, label='F1-Score')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_labels, rotation=45)
    axes[1].set_title("Metrics per Class")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_metrics_and_conf_matrix(conf_matrix, report, labels=np.unique(y_train))

# --- 7. Dự đoán ảnh đầu vào ---
def predict_image(image_path, model_path):
    model = joblib.load(model_path)
    features = extract_hog_features(image_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    print(f"Predicted Label: {prediction}")
    return prediction

# Ví dụ sử dụng:
# img = cv2.imread(r'SIFT-KNN-MK2\HAND-CLASSIFICATION.v1i.folder\test\C\40_jpg.rf.0a2e1c8201a83fc3ac0ec89fcfb77811.jpg')
# img = cv2.resize(img, (640, 640))
# cv2.imshow('image', img)
# predict_image(r'SIFT-KNN-MK2\HAND-CLASSIFICATION.v1i.folder\test\C\40_jpg.rf.0a2e1c8201a83fc3ac0ec89fcfb77811.jpg', r'hand_gesture_rf_model.pkl')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
