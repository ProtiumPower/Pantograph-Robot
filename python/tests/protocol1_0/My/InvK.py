

import cv2
import numpy as np


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
    cv2.imwrite('new_image.jpg', out)
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
    
    input()
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
        z1 = (-b1 + math.sqrt(b1**2 - 4*a1*c1))/(2*a1)
        z4 = (-b2 - math.sqrt(b2**2 - 4*a2*c2))/(2*a2)

        # Calculate angles h1 and h2 for each z value
        h1 = 2 * math.atan(z1)*180/3.14
        h2 = 2 * math.atan(z4)*180/3.14

        h_solutions.append((h1, h2))

    # Print the solutions
    for solution in h_solutions:
        print(solution)

    return h_solutions


if __name__=="__main__":



   print("enter any key to continue or 'q' to quit: ")

   if input()=="q":

       exit()

   pi=segmentation_using_hsv()
   p=pixel_to_cartesian(pi)
   inverse_kinematics(p)
   exit()
   

