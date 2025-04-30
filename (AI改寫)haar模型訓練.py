import cv2
import os
import glob
import numpy as np

# 初始化圖像和標籤列表
images = []
labels = []

# 定義資料夾路徑和對應的標籤（使用相對路徑）
base_path = os.path.join(os.getcwd(), "img")
subfolders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
paths = {os.path.join(base_path, folder): idx for idx, folder in enumerate(subfolders)}

# 遍歷每個資料夾並讀取圖像
for path, label in paths.items():
    if os.path.exists(path):
        for file_path in glob.glob(os.path.join(path, "*")):
            print(f'正在讀取 {file_path} ......')
            with open("training_log.txt", "a") as log_file:
                log_file.write(f"正在讀取 {file_path} ......\n")
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 讀取灰階圖像
            if img is not None:
                img = cv2.resize(img, (400, 400))  # 將圖像調整為 100x100 像素
                images.append(img)
                labels.append(label)
            else:
                print(f'警告: 無法讀取圖像 {file_path}')
    else:
        print(f'警告: 資料夾 {path} 不存在！')

# 確保有足夠的數據進行訓練
if len(images) == 0 or len(labels) == 0:
    print("錯誤: 沒有足夠的圖像進行訓練！")
    exit()

# 創建和訓練模型
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(np.asarray(images), np.asarray(labels))
    # 使用相對路徑儲存模型
    model_path = os.path.join(base_path, "model.xml")
    recognizer.save(model_path)
    print(f'模型已儲存完成！路徑: {model_path}')
except Exception as e:
    print(f"錯誤: 模型訓練或儲存失敗！原因: {e}")