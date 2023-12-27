import cv2
import time
import os
import HandTrackingModule as htm
wCap , hCap = 720, 720

cap = cv2.VideoCapture(0)
cap.set(3,wCap)
cap.set(4,hCap)

folderPath = "finger"
imgList = os.listdir(folderPath)
print(imgList)
imgList = sorted(imgList, key=lambda x: int(x.split('.')[0]))

print(imgList)
overlayImg = []

for imgPath in imgList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    # print(f'{folderPath}/{imgPath}')
    overlayImg.append(image)






detector = htm.handDector(detectCon=0.75)
pre_time = 0
figTips = [4,8,12,16,20]

while True:
    success,img = cap.read()
    img = detector.findHand(img)
    lmList = detector.findPosition(img,draw=True)
    # print(lmList)

    if len(lmList) !=0:
        fingers = []

        #Thumb for right hand
        if lmList[figTips[0]][1] > lmList[figTips[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1,5):
            if lmList[figTips[id]][2] < lmList[figTips[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFinger = fingers.count(1)
        print(totalFinger)

        h,w,c = overlayImg[totalFinger-1].shape
        img[0:h, 0:w] = overlayImg[totalFinger-1]

    # h,w,c = overlayImg[totalFinger-1].shape
    # img[0:100, 0:100] = overlayImg[0]
    current_time = time.time()
    fps = 1/(current_time - pre_time)
    pre_time = current_time

    cv2.putText(img,f'fps: {int(fps)}',(600,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,68))

    cv2.imshow("Image",img)
    cv2.waitKey(1)

