import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from tqdm import tqdm
from pysift import computeKeypointsAndDescriptors
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing
import joblib
import datetime

# ==== ĐỊNH NGHĨA ĐƯỜNG DẪN ====
extract_path = r'HAND-CLASSIFICATION-5'
train_path = os.path.join(extract_path, "train")
valid_path = os.path.join(extract_path, "valid")

# --- 1. Hàm trích xuất đặc trưng SIFT ---
# --- 1. Hàm xử lý một ảnh (thêm đối số img_size, extract_method) ---
def extract_sift_features(image_path, img_size, extract_method):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))

    if extract_method == 0:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
    else:
        keypoints, descriptors = computeKeypointsAndDescriptors(image)

    if descriptors is None:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)

def process_image(image_path, label, img_size, extract_method):
    try:
        features = extract_sift_features(image_path, img_size, extract_method)
        return features, label
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- 2. Hàm chuẩn bị dữ liệu (truyền img_size, extract_method vào partial) ---
def prepare_dataset(dataset_path, img_size, extract_method, num_workers=18):
    image_paths = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image_paths.append((image_path, label))

    if not image_paths:
        print(f"[WARNING] Không tìm thấy dữ liệu trong {dataset_path}")
        return np.array([]), np.array([])

    num_workers = num_workers or multiprocessing.cpu_count()

    print(f"[INFO] Bắt đầu trích xuất đặc trưng SIFT trên {num_workers} tiến trình...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        process_func = partial(process_image, img_size=img_size, extract_method=extract_method)
        results = list(tqdm(pool.starmap(process_func, image_paths),
                            total=len(image_paths),
                            desc="Extracting Features"))

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        print("[ERROR] Không có ảnh nào được xử lý thành công.")
        return np.array([]), np.array([])

    data, labels = zip(*valid_results)
    return np.array(data), np.array(labels)

if __name__ == '__main__':
    img_s_arr = [128, 160, 200, 320, 400, 500, 600, 640]
    for i in img_s_arr:
        extract_method = 1  # 0: thư viện SIFT, 1: pysift
        img_size = i
        os.makedirs("model", exist_ok=True)

        X_train, y_train = prepare_dataset(train_path, img_size, extract_method)
        X_val, y_val = prepare_dataset(valid_path, img_size, extract_method)

        if len(X_train) == 0 or len(X_val) == 0:
            print("[ERROR] Không có dữ liệu hợp lệ để huấn luyện.")
            exit(1)

        # Mã hóa nhãn thành số
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        # ==== TỐI ƯU HÓA RANDOM FOREST ====
        param_grid = {
            'n_estimators': np.arange(50, 301, 50),
            'max_depth': [None, 10, 20, 30, 40, 50, 60],
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

        print("\n Đang tìm kiếm siêu tham số tối ưu...")
        random_search.fit(X_train, y_train_encoded)
        print("\n Hoàn thành tìm kiếm, bắt đầu lưu file")
        # ==== ĐÁNH GIÁ MÔ HÌNH ====
        best_rf = random_search.best_estimator_
        accuracy = best_rf.score(X_val, y_val_encoded)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
        model_path = f'model/best_rf_sift_model-{timestamp}.pkl'
        joblib.dump(best_rf, model_path)
        with open('optimization_history.txt', 'a') as log_file:
            log_file.write("\n\n=======================\n")
            log_file.write(f"Image size: {img_size}x{img_size}\n")
            log_file.write(f"Extract method: {'Built-in library' if extract_method == 0 else 'Manual library'}\n")
            log_file.write(f"Optimization hyper parameter result\n")
            log_file.write(f"{random_search.best_params_}\n")
            log_file.write(f"Accuracy scrore: {accuracy:.4f}\n")
            log_file.write(f"Model path: {model_path}\n")
            print("Lưu file thành công.")