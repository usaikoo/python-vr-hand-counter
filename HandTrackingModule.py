import cv2
import mediapipe as mp
import time


class handDector():
    def __init__(self, mode=False, maxHands=2, detectCon = 0.5, trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.my_hands = mp.solutions.hands
        self.hands = self.my_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def findHand(self,img, draw=True):
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        self.results = self.hands.process(img)
        #draw landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms,self.my_hands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img, handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myhand.landmark):
                        h, w , c = img.shape
                        cx, cy = int(lm.x * w),int(lm.y * h)
                        lmList.append([id,cx,cy])
                        # if id == 20 :
                        if draw:
                            cv2.circle(img, (cx, cy), 8, (153, 255, 153), cv2.FILLED)
        return lmList

