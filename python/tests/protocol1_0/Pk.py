import os
from time import sleep
import numpy as np
import RPi.GPIO as GPIO
import cv2
import matplotlib.pyplot as plt

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


from dynamixel_sdk import *                    # Uses Dynamixel SDK library



# Control table address
ADDR_MX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
ADDR_MX_GOAL_POSITION      = 30
ADDR_MX_PRESENT_POSITION   = 36

ADDR_Moving_speed = 32

ADDR_Compliance_slope_CW=28
ADDR_Compliance_slope_CCW=29

ADDR_Compliance_margin_CW=26
ADDR_Compliance_margin_CCW=27

ADDR_Torque_LImit=34

# Data Byte Length
LEN_MX_GOAL_POSITION       = 2
LEN_MX_PRESENT_POSITION    = 2

# Protocol version
PROTOCOL_VERSION            = 1.0               # See which protocol version is used in the Dynamixel

# Default setting
DXL1_ID                     = 1                 # Dynamixel#1 ID : 1
DXL2_ID                     = 5                 # Dynamixel#1 ID : 5


BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600
DEVICENAME                  = "/dev/ttyUSB0"    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
#DXL_MINIMUM_POSITION_VALUE  = 600           # Dynamixel will rotate between this value
#DXL_MAXIMUM_POSITION_VALUE  = 700            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)

DXL_MINIMUM_POSITION_VALUE_1  = 0
DXL_MAXIMUM_POSITION_VALUE_1  = 4095

DXL_MINIMUM_POSITION_VALUE_2  = 0
DXL_MAXIMUM_POSITION_VALUE_2  = 4095


DXL_MINIMUM_SPEED_VALUE  = 50           # Dynamixel will rotate between this value
#DXL_MAXIMUM_SPEED_VALUE  = 100            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 2                # Dynamixel moving status threshold

speed=1023

compliance_slope= 0                #7 values (2,4,8,16,32,64,128)
compliance_margin=1                # 0-255
torque_limit=10                      #0-1023

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)


# Initialize GroupSyncWrite instance
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_MX_GOAL_POSITION, LEN_MX_GOAL_POSITION)

act=11
em=13

GPIO.setmode(GPIO.BOARD)
GPIO.setup(act, GPIO.OUT)   # pin 11 for device 1
GPIO.setup(em, GPIO.OUT)   # pin 13 for device 2

# Turn both devices off initially
GPIO.output(act, GPIO.LOW)
GPIO.output(em, GPIO.LOW)

COLOR = "red" #mention the color of object used

def pixel_to_cartesian(pixels):
    # image pixels (235,31) to (397,31) separated by cartesian distance of 60mm
    # image pixels (317,59) to (320,288) separated by cartesian distance of 85mm
    # scale_x = 60/(397-235) = 60/162
    # lets take scale_x  = 60/143
    # similarly: scale_y = 85/206
    #offset=np.array([0.033,0.035,0])
    pixels=np.array([[pixels[0]-320],[pixels[1]-240]])
    scale_x=60/141 # linear map factor from pixel to cartesian in mm in camera frame
    scale_y=85/200
    offset_y= 115 - 40
    offset_x= 15 - 8
    rotation_cam2robot=np.array([[0,1],[-1,0]])
    #x_cartesian= pixels[0] * scale_x  + offset_x
    #y_cartesian= pixels[1] * scale_y + offset_y
    scale= np.array([[scale_x,0],[0,scale_y]])
    offset=np.array([[offset_x],[offset_y]])
    return (np.dot(rotation_cam2robot,np.dot(scale,pixels)) + offset)/1000       #cartesian coordinates w.r.t robot frame in mm

def rotate_image(img):
    # Create a zeros image
    #img = np.zeros((400,400), dtype=np.uint8)

    # Specify the text location and rotation angle
    text_location = (240,320)
    angle = 35

    # Draw the text using cv2.putText()
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, 'TheAILearner', text_location, font, 1, 255, 2)

    # Rotate the image using cv2.warpAffine()
    M = cv2.getRotationMatrix2D(text_location, angle, 1)
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Display the results
    #cv2.imshow('img',out)
    #cv2.waitKey(0)
    return out


def segmentation_using_hsv():
    color_thres={"red": 200, "green": 100, "yellow": 100, "blue":100, "orange": 100}
    color_data={"red": [(170,130,25),(180,255,255),300], "green": [(35, 40, 25), (70, 255, 255),100], \
        "yellow": [(75,180,25),(105,255,255),100], "blue":[(75,180,25),(105,255,255),100], "orange": [(75,180,25),(105,255,255),100]}
    green=[(35, 40, 25), (70, 255, 255)]
    orange=[(75,180,25),(105,255,255)]
    red=[(170,110,25),(180,255,255)]
    yellow=[(75,180,25),(105,255,255)]
    blue=[(75,180,25),(105,255,255)]
    #black=[(75,180,25),(105,255,255)]

    port=-1 #cam port for webcam
    video=cv2.VideoCapture(port)
    rslt,img=video.read()
    #480,640,3
    print(img.shape)
    cv2.imshow("camera_view",cv2.resize(img,(400,200)))
    cv2.waitKey(2000)
    img=rotate_image(img)
    cv2.imshow("rotated",cv2.resize(img,(400,200)))
    cv2.waitKey(2000)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Create a mask for the object by selecting a possible range of HSV colors that the object can have:
    mask = cv2.inRange(hsv, color_data[COLOR][0],color_data[COLOR][1])
    #slice the object
    imask = mask>0
    #print(imask.shape)
    img_msk = np.zeros_like(img, np.uint8)
    img_msk[imask] = img[imask]
    #orange=np.clip(orange, 0, 255)
    print(img_msk[imask].shape)
    new_image=cv2.cvtColor(img_msk,cv2.COLOR_HSV2RGB)
    #new_image=cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
    cv2.imshow("detection",cv2.resize(img_msk,(400,200)))
    cv2.waitKey(1000)
    #img_gray = cv2.cvtColor(new_image,cv2.COLOR_RGB2GRAY)
    # Blur the image for better edge detection
    #img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    #ret, thresh = cv2.threshold(img_blur, 100, 255,cv2.THRESH_BINARY_INV)
    #thresh= cv2.Canny(img_gray,0,200)
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centers=[]
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] >color_data[COLOR][2]:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cv2.drawContours(new_image, [i], -1, (0, 255, 0), 2)
            #cv2.circle(new_image, (cx, cy), 7, (0, 0, 255), -1)
            #cv2.putText(new_image, "center", (cx - 20, cy - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            centers.append([cx,cy])
    centers=np.array(centers)
    print(centers.shape)
    avg_center=np.mean(centers,axis=0)
    print(avg_center)
    cv2.circle(new_image, (int(avg_center[0]),int(avg_center[1])), 7, (0, 0, 255), -1)
    cv2.imshow('Original', new_image)
    #cv2.imshow('Original', cv2.resize(img,(400,200)))
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return avg_center


def disable_torque():

    # Disable Dynamixel#1 Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

    # Disable Dynamixel#2 Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

    # Close port
    portHandler.closePort()



def setup_dynamixel():
    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()


    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()


    # Enable Dynamixel#1 Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel#%d has been successfully connected" % DXL1_ID)

    # Enable Dynamixel#2 Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL2_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel#%d has been successfully connected" % DXL2_ID)



    #set torque limit
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Torque_LImit, torque_limit)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Torque_LImit, torque_limit)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #set speed
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Moving_speed, speed)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #sleep(0.25)

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Moving_speed, speed)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #sleep(0.25)



    #set compliance margin CW for DXL1_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_margin_CW, compliance_margin)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #set compliance margin CCW for DXL1_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_margin_CCW, compliance_margin)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))


    #set compliance margin CW for DXL2_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_margin_CW, compliance_margin)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #set compliance margin CCW for DXL2_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_margin_CCW, compliance_margin)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #set compliance slope CW for DXL1_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_slope_CW, compliance_slope)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #set compliance slope CCW for DXL1_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_slope_CCW, compliance_slope)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))


    #set compliance slope CW for DXL2_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_slope_CW, compliance_slope)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))

    #set compliance slope CCW for DXL2_ID
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_slope_CCW, compliance_slope)
    if dxl_com_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_com_result))
    elif dxl__error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl__error))



def move_to_location(coordi=InvKin([0,0,0.180])):
    if coordi[0]>=665 and coordi[1]>=665:
        print("max depth exceeded!")
        sys.exit("max depth exceeded!")
    print(f"moving to {coordi}")
    #coordinates=list(np.array(list(map(float,input("enter coordinates in mm: ").split())))/1000)
    #InvKin(coordi)
    index = 1
    dxl_goal_position_1 = [DXL_MINIMUM_POSITION_VALUE_1, coordi[0]]         # Goal position_1
    dxl_goal_position_2 = [DXL_MINIMUM_POSITION_VALUE_2, coordi[1]]         # Goal position_2
    


    #print("target2", DXL_MAXIMUM_POSITION_VALUE_2)
    # Allocate goal position value into byte array
    # param_goal_position = [DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[index])), DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[index])), DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[index])), DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[index]))]
    # because AX12's goal posiion is only 2 bytes. only needed to split them once.
    param_goal_position_1 = [DXL_LOBYTE(dxl_goal_position_1[index]), DXL_HIBYTE(dxl_goal_position_1[index])]
    param_goal_position_2 = [DXL_LOBYTE(dxl_goal_position_2[index]), DXL_HIBYTE(dxl_goal_position_2[index])]

    try:
        # Add Dynamixel#1 goal position value to the Syncwrite parameter storage
        dxl_addparam_result_1 = groupSyncWrite.addParam(DXL1_ID, param_goal_position_1)

        if dxl_addparam_result_1 != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL1_ID)
            quit()

        # Add Dynamixel#2 goal position value to the Syncwrite parameter storage
        dxl_addparam_result_2 = groupSyncWrite.addParam(DXL2_ID, param_goal_position_2)
        if dxl_addparam_result_2 != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL2_ID)
            quit()


        # Syncwrite goal position
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()
        
        while 1:
            # Read Dynamixel#1 present position
            dxl1_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL1_ID, ADDR_MX_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))

            # Read Dynamixel#2 present position
            dxl2_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL2_ID, ADDR_MX_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))

                
            print("[ID:%03d] GoalPos:%03d  PresPos:%03d\t[ID:%03d] GoalPos:%03d  PresPos:%03d\t" % (DXL1_ID, dxl_goal_position_1[index], dxl1_present_position, DXL2_ID, dxl_goal_position_2[index], dxl2_present_position))

            if not ((abs(dxl_goal_position_1[index] - dxl1_present_position) > DXL_MOVING_STATUS_THRESHOLD) or (abs(dxl_goal_position_2[index] - dxl2_present_position) > DXL_MOVING_STATUS_THRESHOLD)):
                break
            
        #sleep(0.01)'''
    except IndexError:
        raise



if __name__=="__main__":
    depth=0.138
    try:
        while True:
            print("enter any key to continue or 'q' to quit: ")
            if getch()=="q":
                break

            setup_dynamixel() # enable dynamixel torque, setting up limits
            #r=0.06
            #points=[[0.050,0,0.14],[0.05,0.02,0.245],[0.05,0,0.200],[-0.05,-0.06,0.16],[0.050,0,0.14],[0.05,0.02,0.245],[0.05,0,0.22],[0,-0.05,0.22]]
            #points=[[0.050,0,0.16]]
            #points=[[r*np.cos(theta*np.pi/180),r*np.sin(theta*np.pi/180),0.2] for theta in range(0,360,1)]
            center=segmentation_using_hsv()
            target= pixel_to_cartesian(center)