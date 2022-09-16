import cv2
import mediapipe as mp
import os
DirPath=('F:\Hand_Pose_3D\images\Training_validation_testing123\Subject_1\charge_cell_phone\color') #reset pass
Files = os.listdir(DirPath)

mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
for File in Files :
    imgPath = os.path.join(DirPath,File)
    img = cv2.imread(imgPath)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for idx, hand in enumerate(result.multi_hand_landmarks):
            # mp_drawing_util.draw_landmarks(
            #     img,
            #     hand,
            #     mp_hand.HAND_CONNECTIONS,
            #     mp_drawing_style.get_default_hand_landmarks_style(),
            #     mp_drawing_style.get_default_hand_connections_style()
            # )

            mp_drawing_util.draw_landmarks(img, hand, mp_hand.HAND_CONNECTIONS)

            lbl = result.multi_handedness[idx].classification[0].label
            if lbl == "Left":
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #if id == 8:
                        #cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                print(id,cx,cy)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Nhan dang ban tay", img)
    key = cv2.waitKey(1)


cap.release()