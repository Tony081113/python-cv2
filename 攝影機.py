#攝影機
import cv2
import os
import numpy
l = input('輸入分類名字:')
print('載入基本設定中......')
n = 0
y = True
xx = None
yy = None
ww = None
hh = None
print('載入分類器資料庫......')
if os.path.exists('img') == False:
    os.mkdir('img')
os.chdir('.\\img')
if os.path.exists(l) == False:
    os.mkdir(l)
os.chdir('.\\' + l)
print('確認基本資料中......')
while y:
    n+=1
    y = os.path.isfile('img' + str(n) + '.jpg')
print('載入模組中......')
os.chdir('D:\\a\\python')
a = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #導入憨兒模組(人臉)
os.chdir('.\\img')
os.chdir('.\\' + l)
print('載入相機中......')
cap = cv2.VideoCapture(0)
o = None
g = False
print('已準備就緒!')
while cap.isOpened():
    k = cv2.waitKey(1)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame', frame)
        d = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        c = a.detectMultiScale(d,1.1,9)
        for (x,y,w,h) in c:
            o = frame[y: y + h,x: x + w].copy()
            oo = cv2.resize(o, (400, 400))
            cv2.rectangle(frame , (x, y), (x + w, y + h), (0, 0, 255), 2)
            if xx == x and yy == y and ww == w and hh == h:
                g = False
            else:
                g = True
            cv2.imshow('Frame', frame)
            cv2.imshow('Face' , oo)
    if k & 0xFF == ord(' ') and g == True:
        p = 'img' + str(n) + '.jpg'
        n+=1
        cv2.imwrite(p, oo)
        g = False
    if k & 0xFF == ord('b'):
        break
cap.release()
cv2.destroyAllWindows()
print('已關閉相機')
