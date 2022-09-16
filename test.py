from sre_constants import SUCCESS
import cv2
from cvzone.HandTrackingModule import HandDetector
import socket
import os
#parameters
width, height = 1280,720
# #webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, width) #chiều rộng
# cap.set(4,height) #chiều cao

#đọc ảnh
DirPath=('F:\Hand_Pose_3D\images\Training_validation_testing123\Subject_1\charge_cell_phone\color')
Files = os.listdir(DirPath)

# Hand Detector
detecor = HandDetector(maxHands=2, detectionCon=0.8) # số hand tối đa= 1, độ tin cậy = 0.8
csdl='F:\Hand_Pose_3D\images\Training_validation_testing123\Subject_1\charge_cell_phone\color\Test.txt'
#communication
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#serverAddressPort = ("127.0.0.1",5052) #cung cấp địa chỉ máy chủ và các cổng( có thể sử dụng bất kì địa chỉ, số nào miễn là k được dùng ở bất kì nơi nào khác
for File in Files:
    # Get the frame from  the webcam
    imgPath = os.path.join(DirPath,File)
    img = cv2.imread(imgPath)

        #hands
    hands,img = detecor.findHands(img) #trả ra fame ảnh nhận dữ liệu bàn tay
        # Landmark values - (x,y,z)*21 -> sumvalues = 63, rồi lưu vào 1 file để sử lý data
    with open(csdl, mode='a') as f:
        f.write("{} ".format(File))
    data = []
    if hands: #nếu phát hiện hands
            #get the first hand detected
        hand = hands[0]#sẽ cho ra frame hands đầu tiên
            #get the landmark list
        lmList = hand['lmList']
        X_Y = hand['X_Y']
        #print(lmList) #để test
        #print(X_Y)
        for lm in X_Y:
            data.extend([lm[0],lm[1]]) #tạo dữ liệu để giải mã
        

            #sock.sendto(str.encode(str(data)), serverAddressPort) #sent data đến server dưới dạng str để sử lý
    for i in data :
        with open(csdl, mode='a') as f:
             f.write("{} ".format(i))
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) #giảm tỉ lệ hình hiển thị
    with open(csdl, mode='a') as f:
        f.write("\n")
    cv2.imshow("Image",img)
    cv2.waitKey(500) #b1: sau do run test webcam
    cv2.destroyAllWindows()
        #nếu đã chạy ổn thì có thể tiếp tục