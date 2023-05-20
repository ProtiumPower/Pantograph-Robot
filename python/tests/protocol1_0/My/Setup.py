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
    portHandler.terminate()









if __name__=="__main__":







   print("enter any key to continue or 'q' to quit: ")



   if input()=="q":



       exit()



   setup_dynamixel()
   print("Setup Done")
   exit()

   



