# 攝影機
import cv2
import os
import urllib.request

def setup_img_directory(base_dir, folder_name):
    """檢查並創建指定的資料夾"""
    img_dir = os.path.join(base_dir, folder_name)
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def get_next_image_index(img_dir):
    """取得下一個可用的圖片編號"""
    n = 1
    while os.path.isfile(os.path.join(img_dir, f'img{n}.jpg')):
        n += 1
    return n

def download_classifier(classifier_path):
    """下載分類器檔案"""
    url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    print(f"分類器檔案不存在，正在下載: {url}")
    try:
        urllib.request.urlretrieve(url, classifier_path)
        print(f"分類器已下載並儲存到: {classifier_path}")
    except Exception as e:
        print(f"無法下載分類器檔案: {e}")
        raise

def detect_and_display_faces(frame, classifier):
    """檢測人臉並在畫面上顯示"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.1, 9)
    resized_face = None

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (400, 400))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('Face', resized_face)

    return resized_face

def main():
    print('載入基本設定中......')

    # 讓用戶輸入資料夾名稱
    folder_name = input("請輸入要儲存圖片的資料夾名稱: ")
    if not folder_name.strip():
        folder_name = "img"  # 如果未輸入，預設為 "img"

    # 設定資料夾
    base_dir = os.getcwd()
    img_dir = setup_img_directory(base_dir, folder_name)
    next_img_index = get_next_image_index(img_dir)

    # 檢查並下載分類器
    print('檢查分類器資料庫......')
    classifier_path = os.path.join(base_dir, 'haarcascade_frontalface_alt2.xml')
    if not os.path.exists(classifier_path):
        download_classifier(classifier_path)
    classifier = cv2.CascadeClassifier(classifier_path)

    # 開啟攝影機
    print('載入相機中......')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法啟動攝影機")
        return
    print('已準備就緒!')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_face = detect_and_display_faces(frame, classifier)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 儲存圖片
            if resized_face is not None:
                img_path = os.path.join(img_dir, f'img{next_img_index}.jpg')
                cv2.imwrite(img_path, resized_face)
                print(f'儲存圖片: {img_path}')
                next_img_index += 1
        elif key == ord('b'):  # 結束程式
            break

    cap.release()
    cv2.destroyAllWindows()
    print('已關閉相機')

if __name__ == "__main__":
    main()