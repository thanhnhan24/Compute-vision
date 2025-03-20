import cv2
import os

def get_next_image_number(output_folder):
    # Lấy danh sách các tệp tin trong thư mục
    files = os.listdir(output_folder)
    # Lọc ra các tệp tin có đuôi .jpg
    images = [file for file in files if file.endswith('.jpg')]
    
    if not images:
        return 1
    else:
        # Lấy số lớn nhất từ tên các tệp tin đã lưu
        numbers = [int(os.path.splitext(image)[0]) for image in images]
        return max(numbers) + 1

def save_frame(frame, output_folder, img_num):
    # Định dạng tên tệp tin
    filename = f"{output_folder}/{img_num}.jpg"
    # Lưu khung hình
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

def main():
    # Đường dẫn thư mục đầu ra
    output_folder = input("Nhập tên của bạn: ")
    
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    
    print("Press 'C' to capture an image. Press 'Q' to quit.")
    
    # Lấy số hình ảnh tiếp theo cần lưu
    img_num = get_next_image_number(output_folder)
    
    while True:
        # Đọc khung hình từ webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame.")
            break
        
        # Hiển thị khung hình
        cv2.imshow("Webcam", frame)
        
        # Chờ phím nhấn
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Lưu khung hình
            save_frame(frame, output_folder, img_num)
            img_num += 1
        elif key == ord('q'):
            # Thoát khỏi vòng lặp
            break
    
    # Giải phóng webcam và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
