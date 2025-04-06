import cv2
import os , glob
import numpy as np

fi = []
fd = []
fi2 = []
fd2 = []
fi3 = []
fd3 = []

path = "D:\\a\\python\\img\\f0"
path2 = "D:\\a\\python\\img\\f1"
path3 = "D:\\a\\python\\img\\f2"

if os.path.exists(path):
    for i in glob.glob(path + "\\*"):
        fm = os.path.basename(i)
        print('正在讀取'+path+'\\'+fi+'......')
        img = cv2.imread(path + "\\" + fi, 0)
        fi.append(img)
        fd.append(0)
if os.path.exists(path2):
    for i in glob.glob(path2 + "\\*"):
        fm = os.path.basename(i)
        print('正在讀取'+path2+'\\'+fi2+'......')
        img = cv2.imread(path2 + "\\" + fi2, 0)
        fi2.append(img)
        fd2.append(1)
if os.path.exists(path3):
    for i in glob.glob(path3 + "\\*"):
        fm = os.path.basename(i)
        print('正在讀取'+path3+'\\'+fi3+'......')
        img = cv2.imread(path3 + "\\" + fi3, 0)
        fi2.append(img)
        fd2.append(2)

m = cv2.face.LBPHFaceRecognizer_create()
m.train(np.asarray(fi), np.asarray(fd))
m.save("D:\\a\\python\\img\\model.xml")
print('模型已儲存完成！')