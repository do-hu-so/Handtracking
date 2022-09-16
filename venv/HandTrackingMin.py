
import cv2
import mediapipe as mp
import time
import os

#cap =cv2.VideoCapture(0)
DirPath=('F:\Hand_Pose_3D\images\Training_validation_testing123\Subject_1\charge_cell_phone\color')
Files = os.listdir(DirPath)
#modun để phát hiện bàn tay
mpHands = mp.solutions.hands
hands = mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
dem =0
csdl='F:\Hand_Pose_3D\images\Training_validation_testing123\Subject_1\charge_cell_phone\color\Test.txt'
for File in Files:
    # Get the frame from  the webcam
    imgPath = os.path.join(DirPath,File)
    with open(csdl, mode='a') as f:
        f.write("\n")
    dem +=1
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #chuyển đổi từ BGR->RGB
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) #kiểu tra xem có phát hiện tay hay k, in ra tọa độ x,y,z

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks: # handLms để chích xuất thông tin từng tay
            for id, lm in enumerate(handLms.landmark): #chuyền id cho từng điểm trên bàn tay
                #print(id,lm)
                h, w, c = img.shape #cao,rộng và nơi truyền
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #if id == 0: #thử truy vấn vào điểm id=0
                    #cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
                # with open(csdl, mode='a') as f:
                #     f.write("{} {} {} {} \n".format(File,id,cx,cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #vễ điểm được 1 tay
            #handLms: vẽ điểm, mpHands.HAND_CONNECTIONS : vẽ line


    #tính toán fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)
                #làm tròn fps rồi ép str, (10,70): tỉ lệ, font chữ, tỉ lệ, mã màu, độ dày viền chữ
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) #giảm tỉ lệ hình hiển thị
    cv2.imshow("Image",img)
    cv2.waitKey(1)

