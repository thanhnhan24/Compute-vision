import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from pysift import computeKeypointsAndDescriptors

# --- 1. Hàm trích xuất đặc trưng SIFT ---
def extract_sift_features(image_path, mode=0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh mức xám
    image = cv2.resize(image, (640, 640))  # Resize về 640x640
    
    if mode == 0:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
    else:
        keypoints, descriptors = computeKeypointsAndDescriptors(image)
    
    if descriptors is None:
        return np.zeros(128)  # Nếu không có đặc trưng, trả về vector 0
    return np.mean(descriptors, axis=0)  # Lấy trung bình làm đặc trưng

# --- 2. Chuẩn bị dataset ---
def process_image(image_path, label, mode):
    """Xử lý một ảnh duy nhất và trả về (features, label)."""
    try:
        features = extract_sift_features(image_path, mode)
        return features, label
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def prepare_dataset(dataset_path, mode=1, num_workers=None):
    """Chuẩn bị dataset sử dụng multiprocessing để tăng tốc."""
    image_paths = []

    # Lấy danh sách tất cả các ảnh kèm nhãn
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image_paths.append((image_path, label))  # Lưu đường dẫn và nhãn

    if not image_paths:
        print(f"[WARNING] Không tìm thấy dữ liệu trong {dataset_path}")
        return np.array([]), np.array([])

    # Sử dụng multiprocessing để xử lý ảnh song song
    num_workers = num_workers or multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        print(f"[INFO] Bắt đầu trích xuất đặc trưng SIFT trên {num_workers} tiến trình...")
        process_func = partial(process_image, mode=mode)
        
        results = list(tqdm(pool.starmap(process_func, image_paths), total=len(image_paths), desc="Extracting Features"))

    # Lọc kết quả hợp lệ
    data, labels = zip(*[res for res in results if res is not None])

    return np.array(data), np.array(labels)


# --- 3. Huấn luyện mô hình Random Forest ---
def train_and_save_model(X_train, y_train, X_valid, y_valid):
    rf_model = RandomForestClassifier(
        n_estimators=200, random_state=42,
        min_samples_split=2, min_samples_leaf=2,
        max_features='sqrt', max_depth=50, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Lưu mô hình
    train_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = f"hand_gesture_rf_model_{train_time}.pkl"
    joblib.dump(rf_model, model_path)
    print(f"Model saved to {model_path}")

    # Ghi log thông tin mô hình
    log_path = "training_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"\nTraining Timestamp: {train_time}\n")
        log_file.write(f"Model Parameters: {rf_model.get_params()}\n")
        log_file.write(f"Training Accuracy: {accuracy_score(y_train, rf_model.predict(X_train)):.4f}\n")
        log_file.write(f"Validation Accuracy: {accuracy_score(y_valid, rf_model.predict(X_valid)):.4f}\n")
    print(f"Training log saved to {log_path}")

    return rf_model, model_path

# --- 4. Đánh giá mô hình ---
def evaluate_model(rf_model, X_valid, y_valid):
    y_pred = rf_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    report = classification_report(y_valid, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_valid, y_pred))

    return conf_matrix, report

# --- 5. Trực quan hóa kết quả ---
def plot_metrics_and_conf_matrix(cm, report, labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")
    
    # Accuracy & Precision per Class
    class_labels = list(report.keys())[:-3]
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

# --- 6. Chạy chương trình ---
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Tránh lỗi trên Windows khi chạy multiprocessing

    # Định nghĩa đường dẫn dữ liệu
    data_path = r'HAND-CLASSIFICATION.v4i.folder'
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    # Load dữ liệu train và validation
    X_train, y_train = prepare_dataset(train_path)
    X_valid, y_valid = prepare_dataset(valid_path)

    # Huấn luyện và lưu mô hình
    rf_model, model_path = train_and_save_model(X_train, y_train, X_valid, y_valid)

    # Đánh giá mô hình
    conf_matrix, report = evaluate_model(rf_model, X_valid, y_valid)

    # Trực quan hóa kết quả
    plot_metrics_and_conf_matrix(conf_matrix, report, labels=np.unique(y_train))
