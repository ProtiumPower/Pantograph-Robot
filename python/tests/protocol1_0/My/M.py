


import cv2

import numpy as np



from dynamixel_sdk import *                    # Uses Dynamixel SDK library



import os, sys, ctypes

import os, sys, ctypes

if os.name == 'nt':
    import msvcrt
    # ...
else:
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())

# ...







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

LEN_MX_GOAL_POSITION       = 4

LEN_MX_PRESENT_POSITION    = 4



# Protocol version

PROTOCOL_VERSION            = 1.0               # See which protocol version is used in the Dynamixel



# Default setting

DXL1_ID                     = 1                 # Dynamixel#1 ID : 1

DXL2_ID                     = 5                 # Dynamixel#1 ID : 5





BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600

DEVICENAME                  = "COM5"    # Check which port is being used on your controller

                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"



TORQUE_ENABLE               = 1                 # Value for enabling the torque

TORQUE_DISABLE              = 0                 # Value for disabling the torque

#DXL_MINIMUM_POSITION_VALUE  = 600           # Dynamixel will rotate between this value

#DXL_MAXIMUM_POSITION_VALUE  = 700            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)



DXL_MINIMUM_POSITION_VALUE_1  = 821

DXL_MAXIMUM_POSITION_VALUE_1  = 3170



DXL_MINIMUM_POSITION_VALUE_2  = 197

DXL_MAXIMUM_POSITION_VALUE_2  = 2044





DXL_MINIMUM_SPEED_VALUE  = 50           # Dynamixel will rotate between this value

#DXL_MAXIMUM_SPEED_VALUE  = 100            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)

DXL_MOVING_STATUS_THRESHOLD = 2                # Dynamixel moving status threshold



speed=50



compliance_slope= 64                #7 values (2,4,8,16,32,64,128)

compliance_margin=1                # 0-255

torque_limit=90                      #0-1023



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





def setup_dynamixel():

    # Open port

    if portHandler.openPort():

        print("Succeeded to open the port")

    else:

        print("Failed to open the port")

        print("Press any key to terminate...")

        input()

        quit()





    # Set port baudrate

    if portHandler.setBaudRate(BAUDRATE):

        print("Succeeded to change the baudrate")

    else:

        print("Failed to change the baudrate")

        print("Press any key to terminate...")

        input()

        quit()





    # Enable Dynamixel#1 Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL1_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    print("Entered")
    if dxl_comm_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

    elif dxl_error != 0:

        print("%s" % packetHandler.getRxPacketError(dxl_error))

    else:

        print("Dynamixel#%d has been successfully connected" % DXL1_ID)
    print("Exit")


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
        print("Entered3")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Torque_LImit, torque_limit)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered4")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #set speed

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Moving_speed, speed)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered5")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #sleep(0.25)



    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Moving_speed, speed)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered6")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #sleep(0.25)







    #set compliance margin CW for DXL1_ID

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_margin_CW, compliance_margin)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered7")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #set compliance margin CCW for DXL1_ID

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_margin_CCW, compliance_margin)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered8")
        print("%s" % packetHandler.getRxPacketError(dxl__error))





    #set compliance margin CW for DXL2_ID

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_margin_CW, compliance_margin)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered9")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #set compliance margin CCW for DXL2_ID

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_margin_CCW, compliance_margin)
    
    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered10")
        print("%s" % packetHandler.getRxPacketError(dxl__error))







    #set compliance slope CW for DXL1_ID
    
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_slope_CW, compliance_slope)

    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered11")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #set compliance slope CCW for DXL1_ID
    
    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL1_ID, ADDR_Compliance_slope_CCW, compliance_slope)

    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered12")
        print("%s" % packetHandler.getRxPacketError(dxl__error))





    #set compliance slope CW for DXL2_ID

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_slope_CW, compliance_slope)

    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered13")
        print("%s" % packetHandler.getRxPacketError(dxl__error))



    #set compliance slope CCW for DXL2_ID

    dxl_com_result, dxl__error = packetHandler.write2ByteTxRx(portHandler, DXL2_ID, ADDR_Compliance_slope_CCW, compliance_slope)

    if dxl_com_result != COMM_SUCCESS:

        print("%s" % packetHandler.getTxRxResult(dxl_com_result))

    elif dxl__error != 0:
        print("Entered14")
        print("%s" % packetHandler.getRxPacketError(dxl__error))








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



def rotate_image(img):

    # Create a zeros image

    #img = np.zeros((400,400), dtype=np.uint8)

    # Specify the text location and rotation angle

    text_location = (320,240)

    angle = 178.6917

   

    # Draw the text using cv2.putText()

    font = cv2.FONT_HERSHEY_SIMPLEX

    #cv2.putText(img, 'TheAILearner', text_location, font, 1, 255, 2)



    # Rotate the image using cv2.warpAffine()

    M = cv2.getRotationMatrix2D(text_location, angle, 1)

    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))



    # Display the results

    cv2.imwrite('rotated.jpg', out)

    return out

    





COLOR="black"

def segmentation_using_hsv():

    color_thres={"red": 200, "green": 100, "yellow": 100, "blue":100, "orange": 100, "black":200}

    color_data={"red": [(170,130,25),(180,255,255),300], "black": [(0,0,0),(180,255,30),300], "green": [(35, 40, 25), (70, 255, 255),100], \

        "yellow": [(75,180,25),(105,255,255),100], "blue":[(75,180,25),(105,255,255),100], "orange": [(75,180,25),(105,255,255),100]}

    green=[(35, 40, 25), (70, 255, 255)]

    orange=[(75,180,25),(105,255,255)]

    red=[(170,110,25),(180,255,255)]

    yellow=[(75,180,25),(105,255,255)]

    blue=[(75,180,25),(105,255,255)]

    black=[(0,0,0),(180,255,30)]



    port=0 #cam port for webcam

    video=cv2.VideoCapture(port)

    ret,img=video.read()

    if not ret:

        print("Failed to capture frame")

        return None

    #480,640,3

    print(img.shape)

    cv2.waitKey(2000)

    cv2.imshow("camera_view",img)
    cv2.imwrite("camera_view.jpg",img)

    cv2.waitKey(0)

    img=rotate_image(img)

    cv2.imshow("rotated",img)

    cv2.waitKey(0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imwrite('hsv.jpg', hsv)

    #Create a mask for the object by selecting a possible range of HSV colors that the object can have:

    mask = cv2.inRange(hsv, color_data[COLOR][0],color_data[COLOR][1])

    cv2.imwrite('mask.jpg', mask)



    #slice the object

    imask = mask>0

    #print(imask.shape)

    img_msk = np.zeros_like(img, np.uint8)

    img_msk[imask] = img[imask]

    #orange=np.clip(orange, 0, 255)

    new_image=cv2.cvtColor(img_msk,cv2.COLOR_HSV2RGB)

    #new_image=cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)

    cv2.imshow("detection",new_image)

    cv2.waitKey(0)



    contours, hierarchies = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    centers=[] 

    p=0

    for i in contours:

        M = cv2.moments(i)

        if M['m00'] >color_data[COLOR][2]:

            cx = int(M['m10']/M['m00'])

            cy = int(M['m01']/M['m00'])

            #cv2.drawContours(new_image, [i], -1, (0, 255, 0), 2)

            #cv2.circle(new_image, (cx, cy), 7, (0, 0, 255), -1)

            #cv2.putText(new_image, "center", (cx - 20, cy - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if cx>10 and (cy>10 and cy<460):   #for ignoring boundaries

             print(f"x: {cx} y: {cy}")

             centers.append([cx,cy])

             p=p+1

    centers=np.array(centers)

    print("No. of objects detected:",p)

    avg_center=np.mean(centers,axis=0)

    print(avg_center)

    for c in centers:

     center = (int(c[0]), int(c[1]))

     cv2.circle(new_image, center, 4, (0, 0, 255), -1)

    cv2.imshow('Original', new_image)

    cv2.imwrite('Centers.jpg', new_image)

    #cv2.imshow('Original', cv2.resize(img,(400,200)))

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    cv2.waitKey(0)

    return centers



def pixel_to_cartesian(pixels):

    '''Calculate Offset_x and Offset_y (Distance 0f (0,0) pixel from origin) and 

    Calculate scale by putting a coin and getting its pixel coordinates then diving by actual coordinates'''

    scale_x = 0.6  # linear map factor from pixel to Cartesian in mm in the camera frame

    scale_y = 0.63

    offset_y = 370

    offset_x = 150

    cartesian = []

    

    for p in pixels:

        modified_x = p[0] * scale_x - offset_x

        modified_y = -p[1] * scale_y + offset_y

        cartesian.append([modified_x, modified_y])

    print("Cartesian")

    for c in cartesian:

        print(f"x: {c[0]} y: {c[1]}")  # Print modified coordinates



    return cartesian





import math





def inverse_kinematics(coordinates):

    h_solutions = []

    for coord in coordinates:

        x, y = coord



        # Calculate coefficients for the constraint equations

        r1 = 125

        r2 = 195

        r3 = 65



        a1 = r1**2 + y**2 + (x + r3)**2 - r2**2 + 2*(x + r3)*r1

        b1 = -4*y*r1

        c1 = r1**2 + y**2 + (x + r3)**2 - r2**2 - 2*(x + r3)*r1

        a2 = r1**2 + y**2 + (x - r3)**2 - r2**2 + 2*(x - r3)*r1

        b2 = -4*y*r1

        c2 = r1**2 + y**2 + (x - r3)**2 - r2**2 - 2*(x - r3)*r1



        # Calculate z for k = 1 and -1

        discriminant1 = b1**2 - 4*a1*c1

        discriminant2 = b2**2 - 4*a2*c2



        if discriminant1 < 0 or discriminant2 < 0:

            # Handle the case when either discriminant is negative

            print(f"Point ({x}, {y}) is out of reach1")

            continue



        z1 = (-b1 + math.sqrt(discriminant1))/(2*a1)

        z4 = (-b2 - math.sqrt(discriminant2))/(2*a2)



        # Calculate angles h1 and h2 for each z value

        h1 = 2 * math.atan(z1)

        h2 = 2 * math.atan(z4)

        

        # Two links of end effector should have angle more than 55 degree

        '''g = (3.14159-(((x**2 + y**2 + r1**2 * (math.sin(h1) * math.sin(h2) + math.cos(h1) * math.cos(h2)) - r1*(x*(math.cos(h1) + math.cos(h2)) + y*(math.sin(h1) * math.sin(h2))) + r1*r3 * math.cos(h1) - math.cos(h2)) - r3**2)/r2**2))*180/3.14

        print(g)

        f = math.acos(((x**2 + y**2 + r1**2 * (math.sin(h1) * math.sin(h2) + math.cos(h1) * math.cos(h2)) - r1*(x*(math.cos(h1) + math.cos(h2)) + y*(math.sin(h1) * math.sin(h2))) + r1*r3 * math.cos(h1) - math.cos(h2)) - r3**2)/r2**2)



        if ((3.14-f)*180/3.14) < 55:

            # Handle the case when f is less than 55 degrees

            print(f"Point ({x}, {y}) is out of reach2")

            continue'''



        g1 = 2 * math.atan(z1) * 180/3.14

        g2 = 2 * math.atan(z4) * 180/3.14

  

        h_solutions.append((g1, g2))

        



        

         

    

    # Print the solutions
    print("Motor angles in degrees by applying inverse kinematics")
    for solution in h_solutions:

        print(solution)

    return h_solutions





def move_motors(angle_pairs):

    '''# Convert degrees to motor units for position p

    p_motor1_position = int(2048-(((180-k[0][1])/ 360) * (4096)))

    p_motor2_position = int(2048+((k[0][0] / 360) * (4096)))

    mtor1_position = [p_motor1_position]
    mtor2_position = [p_motor2_position]

    print(p_motor1_position, p_motor2_position)'''
    index = 1
    for angles in angle_pairs:

        # Convert degrees to motor units for motor 1

        motor1_position = int(2048-((180-angles[1] / 360) * (4096)))


        # Convert degrees to motor units for motor 2

        motor2_position = int(2048+((angles[0] / 360) * (4096)))

        
        dxl_goal_position_1 = [DXL_MINIMUM_POSITION_VALUE_1, motor1_position]         # Goal position_1
        dxl_goal_position_2 = [DXL_MINIMUM_POSITION_VALUE_2, motor2_position]         # Goal position_2


        param_goal_position_1 = [DXL_LOBYTE(dxl_goal_position_1[index]), DXL_HIBYTE(dxl_goal_position_1[index])]
        param_goal_position_2 = [DXL_LOBYTE(dxl_goal_position_2[index]), DXL_HIBYTE(dxl_goal_position_2[index])]
        # Create a group sync write instance

        group_syncwrite = GroupSyncWrite(port_handler, packet_handler, ADDR_MX_GOAL_POSITION, 2)

 
        dxl_addparam_result_1 = groupSyncWrite.addParam(DXL1_ID, param_goal_position_1)

        if dxl_addparam_result_1 != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL1_ID)
            quit()

        # Add Dynamixel#2 goal position value to the Syncwrite parameter storage
        dxl_addparam_result_2 = groupSyncWrite.addParam(DXL2_ID, param_goal_position_2)
        if dxl_addparam_result_2 != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL2_ID)
            quit()

        # Add the motor positions to the group sync write

        # Sync write the goal positions

        dxl_comm_result = group_syncwrite.txPacket()

        if dxl_comm_result != COMM_SUCCESS:

            print(f"Failed to sync write goal positions. Error code: {dxl_comm_result}")

            continue



        # Clear the sync write parameter storage

        group_syncwrite.clearParam()



        time.sleep(2)  # Stop for 2 seconds



        # Create a group sync write instance for position p

        p_group_syncwrite = GroupSyncWrite(port_handler, packet_handler, ADDR_MX_GOAL_POSITION, 2)



        # Add the motor positions for position p to the group sync write
    
        dxl_goal_position_1 = [DXL_MINIMUM_POSITION_VALUE_1, 2100]         # Goal position_1
        dxl_goal_position_2 = [DXL_MINIMUM_POSITION_VALUE_2, 1900]         # Goal position_2

        param_goal_position_1 = [DXL_LOBYTE(dxl_goal_position_1[index]), DXL_HIBYTE(dxl_goal_position_1[index])]
        param_goal_position_2 = [DXL_LOBYTE(dxl_goal_position_2[index]), DXL_HIBYTE(dxl_goal_position_2[index])]
        # Create a group sync write instance

        group_syncwrite = GroupSyncWrite(port_handler, packet_handler, ADDR_MX_GOAL_POSITION, 2)

 
        dxl_addparam_result_1 = groupSyncWrite.addParam(DXL1_ID, param_goal_position_1)

        if dxl_addparam_result_1 != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL1_ID)
            quit()

        # Add Dynamixel#2 goal position value to the Syncwrite parameter storage
        dxl_addparam_result_2 = groupSyncWrite.addParam(DXL2_ID, param_goal_position_2)
        if dxl_addparam_result_2 != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % DXL2_ID)
            quit()

        # Add the motor positions to the group sync write

        # Sync write the goal positions

        dxl_comm_result = group_syncwrite.txPacket()

        if dxl_comm_result != COMM_SUCCESS:

            print(f"Failed to sync write goal positions. Error code: {dxl_comm_result}")

            continue



        # Clear the sync write parameter storage

        group_syncwrite.clearParam()



    

    dxl_goal_position_1 = [DXL_MINIMUM_POSITION_VALUE_1, 1977]         # Goal position_1
    dxl_goal_position_2 = [DXL_MINIMUM_POSITION_VALUE_2, 2040]         # Goal position_2



    param_goal_position_1 = [DXL_LOBYTE(dxl_goal_position_1[index]), DXL_HIBYTE(dxl_goal_position_1[index])]
    param_goal_position_2 = [DXL_LOBYTE(dxl_goal_position_2[index]), DXL_HIBYTE(dxl_goal_position_2[index])]
    
    # Create a group sync write instance

    group_syncwrite = GroupSyncWrite(port_handler, packet_handler, ADDR_MX_GOAL_POSITION, 2)

 
    dxl_addparam_result_1 = groupSyncWrite.addParam(DXL1_ID, param_goal_position_1)

    if dxl_addparam_result_1 != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL1_ID)
        quit()

    # Add Dynamixel#2 goal position value to the Syncwrite parameter storage
    dxl_addparam_result_2 = groupSyncWrite.addParam(DXL2_ID, param_goal_position_2)
    if dxl_addparam_result_2 != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL2_ID)
        quit()

    # Add the motor positions to the group sync write



    # Clear the sync write parameter storage

    group_syncwrite.clearParam()



def run():

    # Initialize the motors

   ''' if not setup_dynamixel():

        print("Failed to initialize the motors")

        return'''



    # Call the segmentation function to get the pixel coordinates

    pixels = segmentation_using_hsv()



    # Convert pixel coordinates to Cartesian coordinates

    cartesian = pixel_to_cartesian(pixels)



    # Calculate inverse kinematics to get the angle pairs

    angle_pairs = inverse_kinematics(cartesian)

    '''p = [[0, 0]]  # Initialize p as a nested list with one element
    k = None  # Assign a default value to k
    x_input = input("Enter the x-coordinate: ")
    y_input = input("Enter the y-coordinate: ")
    if x_input and y_input:  # Check if both inputs are non-empty
       x = float(x_input)
       y = float(y_input)
       p[0][0] = x
       p[0][1] = y
       k = inverse_kinematics(p)
    else:
      print("Invalid input. Please enter numerical values for both x-coordinate and y-coordinate.")


    if k is not None:
       
    else:
       print("Cannot move motors because k is not assigned a valid value.")'''

    move_motors(angle_pairs)
    disable_torque()

    # Repeat the process for all points from inverse kinematics

    print("Finished")


if __name__=="__main__":
   print("enter any key to continue or 'q' to quit: ")
   if input()=="q":
       exit()

   run()

   exit()

   



