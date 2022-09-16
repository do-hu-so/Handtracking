
import cv2
from cvzone.HandTrackingModule import HandDetector
import socket
import os
DirPath=('F:\Hand_Pose_3D\images\Training_validation_testing123\Subject_1\charge_cell_phone\color')
Files = os.listdir(DirPath)
#cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
csdl='F:\Hand_Pose_3D\Joint.txt'
dem=0
#while True:
for File in Files:
    # Get the frame from  the webcam
    imgPath = os.path.join(DirPath,File)
    img = cv2.imread(imgPath)
    # Get image frame
    #success, img = cap.read()
        # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
    dem +=1
    with open(csdl, mode='a') as f:
            f.write("{} ".format(File))
    dataleft=[]
    dataright=[]
    if hands:            # Hand 1
        if len(hands)==1:
            hand1 = hands[0]
            X_Y = hand1['X_Y']
            #lmList1 = hand1["lmList"]  # List of 21 Landmark points
            #bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                #centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            if handType1 == "Left":
            
                for lm in X_Y:
                    dataleft.extend([lm[0],lm[1]]) #tạo dữ liệu để giải mã
                

                #fingers1 = detector.fingersUp(hand1)
                with open(csdl, mode='a') as f:
                    f.write("001 ")
                for i in dataleft :
                    with open(csdl, mode='a') as f:
                            f.write("{} ".format(i))
                with open(csdl, mode='a') as f:
                    f.write("\n{} 002 none".format(File))

            elif handType1 == "Right":
                for lm in X_Y:
                    dataright.extend([lm[0],lm[1]]) #tạo dữ liệu để giải mã
            
                #fingers1 = detector.fingersUp(hand1)
                with open(csdl, mode='a') as f:
                    f.write("001 none \n{} 002 ".format(File))
                for i in dataright :
                    with open(csdl, mode='a') as f:
                            f.write("{} ".format(i))                    


        elif len(hands) == 2:
                # Hand 2
                hand1 = hands[0]
                hand2 = hands[1]
                #lmList2 = hand2["lmList"]  # List of 21 Landmark points
                    #bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                    #centerPoint2 = hand2['center']  # center of the hand cx,cy
                #handType1 = hand1["type"]
                #handType2 = hand2["type"]  # Hand Type "Left" or "Right"
                X_Y1 = hand1['X_Y']
                X_Y2 = hand2['X_Y']

                for lm in X_Y1:
                    dataleft.extend([lm[0],lm[1]]) #tạo dữ liệu để giải mã
                for lm in X_Y2:
                    dataright.extend([lm[0],lm[1]]) #tạo dữ liệu để giải mã
            

                fingers1 = detector.fingersUp(hand1)
                fingers2 = detector.fingersUp(hand2)
                with open(csdl, mode='a') as f:
                    f.write("001 ")

                for i in dataleft :
                    with open(csdl, mode='a') as f:
                        f.write("{} ".format(i))
                with open(csdl, mode='a') as f:
                    f.write("\n{} 002 ".format(File))
                for y in dataright :
                    with open(csdl, mode='a') as f:
                        f.write("{} ".format(y))
                # Find Distance between two Landmarks. Could be same hand or different hands
                #length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
                
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
        # Display
    else:
        with open(csdl, mode='a') as f:
                    f.write("001 none\n{} 002 none ".format(File))
    with open(csdl, mode='a') as f:
        f.write("\n")
    cv2.imshow("Image", img)
    cv2.waitKey(1)