

import cv2
import numpy as np


def rotate_image():
    # Create a zeros image
    #img = np.zeros((400,400), dtype=np.uint8)
    # Specify the text location and rotation angle
    text_location = (320,240)
    angle = 178.6917
    port=0 #cam port for webcam
    video=cv2.VideoCapture(port)
    ret,img=video.read()
    if not ret:
        print("Failed to capture frame")
        return None
    # Draw the text using cv2.putText()
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, 'TheAILearner', text_location, font, 1, 255, 2)

    # Rotate the image using cv2.warpAffine()
    M = cv2.getRotationMatrix2D(text_location, angle, 1)
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Display the results
    cv2.imwrite('new_image.jpg', out)
    #cv2.imshow("rotated",cv2.resize(img,(400,200)))
    #cv2.waitKey(2000)
    return out
    


'''COLOR="red"
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return avg_center'''



if __name__=="__main__":



   print("enter any key to continue or 'q' to quit: ")

   if input()=="q":

       exit()

   rotate_image()

