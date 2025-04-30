import cv2
import os
import numpy as np
import urllib.request
import sys

# 自動下載 Haar 特徵分類器
def download_haarcascade():
    haar_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
    try:
        if not os.path.exists(haar_path):
            print("正在下載 Haar 特徵分類器...")
            urllib.request.urlretrieve(haar_url, haar_path)
            print("Haar 特徵分類器下載完成！")
    except Exception as e:
        print(f"錯誤: 無法下載 Haar 特徵分類器！請檢查網絡連接或手動下載。\n{e}")
        sys.exit()
    return haar_path

# 加載 Haar 特徵分類器
cascade_path = download_haarcascade()
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("錯誤: 無法加載 Haar 特徵分類器！")
    sys.exit()

# 加載訓練好的模型
model_path = os.path.join(os.getcwd(), "model.xml")
if not os.path.exists(model_path):
    print(f"錯誤: 模型文件 {model_path} 不存在！請確保模型已訓練並保存為 'model.xml'。")
    sys.exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
print(f"模型加載成功！路徑: {model_path}")

# 手動定義標籤到名稱的映射
label_to_name = {
    0: "李孟儒",
    1: "李孟昀",
    2: "李宗恩"
}

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤: 無法開啟攝影機！請檢查攝影機是否已連接或被其他應用程序佔用。")
    sys.exit("錯誤: 無法開啟攝影機！")

print("按下 'q' 鍵退出程式。")

CONFIDENCE_THRESHOLD = 50  # 可調整的置信度閾值

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤: 無法讀取攝影機畫面！")
            break

        # 將畫面轉為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        if len(faces) == 0:
            print("警告: 未偵測到人臉，請檢查攝影機畫面或調整參數。")
        
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
                if confidence < CONFIDENCE_THRESHOLD:
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
        if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            print("警告: 顯示窗口被關閉，請重新打開程式。")
            break

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()