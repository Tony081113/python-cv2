# python-cv2
# Python OpenCV (cv2) 圖像辨識開源代碼

## 簡介
本項目是一個基於 Python OpenCV (`cv2`) 的開源代碼庫，專注於圖像辨識功能，包括物件偵測、人臉識別、特徵匹配以及分類等。

## 安裝
在開始使用此代碼庫之前，請確保您的環境已安裝 Python 以及 OpenCV。

使用以下命令安裝 OpenCV：
```bash
pip install opencv-python
```
如果需要完整功能（包含額外的 OpenCV 模組），請安裝：
```bash
pip install opencv-python-headless
```

## 功能概述
本代碼庫提供以下圖像辨識功能：
- 物件偵測（輪廓檢測、Haar Cascades、YOLO）
- 人臉識別（Haar Cascades、DNN 模型）
- 特徵匹配（SURF）
- 圖像分類（CNN、深度學習模型）

## 使用範例
### 人臉識別（Haar Cascades）
```python
import cv2

# 載入 Haar Cascade 分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 讀取影像
image = cv2.imread('face.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 偵測人臉
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 繪製人臉框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 貢獻方式
歡迎任何人貢獻此開源代碼！請遵循以下步驟：
1. Fork 此存儲庫
2. 創建一個新的分支進行修改
3. 提交 PR（Pull Request）

## 版權與許可
本專案基於 MIT 許可證發布，歡迎自由使用與修改。

---

如果您對此項目有任何建議或問題，請在 GitHub Issues 中提出！

