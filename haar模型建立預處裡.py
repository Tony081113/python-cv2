import os , glob , cv2

ph = []
phid = []
ph2=[]
phid2=[]

a = './img/f0/'
b = './img/f0/'
aa = './img/f1/'
bb = './img/f1/'

if os.path.exists(a):
    a += '*'
    for i in glob.glob(a):
        fn = os.path.basename(i)
        print('讀取' + fn +'中.....')
        img = cv2.imread(b+fn,cv2.COLOR_BGR2GRAY)
        ph.append(img)
        phid.append(0)
    for i in glob.glob(b):
        fn = os.path.basename(i)
        print('讀取' + fn +'中.....')
        img = cv2.imread(b+fn,cv2.COLOR_BGR2GRAY)
        ph2.append(img)
        phid2.append(1)
else:
    print('路徑錯誤:(')
