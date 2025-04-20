import cv2
import os
import numpy as np
import urllib.request

# 自動下載 Haar 特徵分類器
def download_haarcascade():
    haar_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
    if not os.path.exists(haar_path):
        print("正在下載 Haar 特徵分類器...")
        urllib.request.urlretrieve(haar_url, haar_path)
        print("Haar 特徵分類器下載完成！")
    return haar_path

# 加載 Haar 特徵分類器
cascade_path = download_haarcascade()
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("錯誤: 無法加載 Haar 特徵分類器！")
    exit()

# 加載訓練好的模型
model_path = os.path.join(os.getcwd(), "model.xml")  # 在當前路徑下尋找模型文件
if not os.path.exists(model_path):
    print(f"錯誤: 模型文件 {model_path} 不存在！")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
print(f"模型加載成功！路徑: {model_path}")

# 定義標籤到名稱的映射
base_path = os.path.join(os.getcwd(), "img")  # 假設訓練圖像存放在 "img" 資料夾中
if not os.path.exists(base_path):
    print(f"錯誤: 資料夾 {base_path} 不存在！")
    exit()

subfolders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
if not subfolders:
    print(f"錯誤: 資料夾 {base_path} 中沒有子資料夾！")
    exit()

paths = {idx: folder for idx, folder in enumerate(subfolders)}  # 建立標籤到名稱的映射
label_to_name = {idx: folder for idx, folder in paths.items()}  # 反向映射標籤到名稱

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤: 無法開啟攝影機！")
    exit()

print("按下 'q' 鍵退出程式。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("錯誤: 無法讀取攝影機畫面！")
        break

    # 將畫面轉為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        # 繪製人臉框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 擷取人臉區域並調整大小
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (400, 400))

        # 預測人臉
        label, confidence = recognizer.predict(face_resized)

        # 處理預測結果
        if label in label_to_name:
            name = label_to_name[label]
            text = f"{name} ({confidence:.2f})"
            if confidence < 50:  # 假設置信度小於 50 表示可靠
                color = (0, 255, 0)  # 綠色
            else:
                color = (0, 0, 255)  # 紅色
        else:
            text = "未知"
            color = (0, 0, 255)  # 紅色

        # 在畫面上顯示名稱和置信度
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 顯示畫面
    cv2.imshow("Face Recognition", frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()