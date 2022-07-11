import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
 
# Importing all images
fist = cv2.imread("Resources/fist.png", cv2.IMREAD_UNCHANGED)
monster = cv2.imread("Resources/monster.png", cv2.IMREAD_UNCHANGED)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# variables
hp = 20000

while True:

    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    img = cv2.addWeighted(img, 0.2, img, 0.8, 0)
    lmList = detector

    img = cvzone.overlayPNG(img, monster, (660, 90))

    cv2.putText(img, str(hp), (550, 150), cv2.FONT_HERSHEY_COMPLEX,
                    2, (255, 255, 255), 5)     
   
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        x, y, w, h = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        h1, w1, _ = fist.shape
        y1 = y - h1 // 2
        y1 = np.clip(y1, 0, 1200)

        x1 = x - h1 // 2
        x1 = np.clip(x1, 0, 1200)


        fingers1 = detector.fingersUp(hand1)

        if sum(fingers1) == 0 or sum(fingers1) == 1:
            if x1 < 650 and y1 < 180:
                img = cvzone.overlayPNG(img, fist, (x1, y1))
                if(x1 > 350 and y1 < 200):
                    hp -= 20

        # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)