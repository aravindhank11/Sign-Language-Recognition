import cv2
import numpy as np
import sys

def showImage(image,secondsToWait,heading,width=450,height=500):
    #Show Image
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    imageToShow = cv2.resize(image, (width, height)) 
    cv2.imshow(heading,imageToShow)
    secondsToWait = secondsToWait * 1000
    cv2.waitKey(secondsToWait)    

#Read Image
argumentsPassed = sys.argv
user = argumentsPassed[1]
signNumber = argumentsPassed[2]
trialNumber = argumentsPassed[3]
inputFile = '<PATH>/DATASET/USER-'+user+'/'+user+'-'+signNumber+'-'+trialNumber+'.jpg'
outputFile = '<PATH>/DATASET/USER-PROCESSED-'+user+'/'+user+'-'+signNumber+'-'+trialNumber+'.jpg'
img = cv2.imread(inputFile)

height, width = img.shape[:2]
print(height,width)
showImage(img,2,"Original Image")
kernel = np.ones((5,5),np.uint8)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
frameHSV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)    

skin_min = np.array([100, 133, 100],np.uint8) #0,133,77
skin_max = np.array([255, 255, 255],np.uint8) #255,173,127     

#Thresholding based on skin color
threshSkinColor = cv2.inRange(frameHSV, skin_min, skin_max)
#Closing and Opening the image to eliminate internal noise
threshSkinColor = cv2.morphologyEx(threshSkinColor, cv2.MORPH_CLOSE, kernel)
threshSkinColor = cv2.morphologyEx(threshSkinColor, cv2.MORPH_OPEN, kernel)

_,contours, hierarchy = cv2.findContours(threshSkinColor,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
for i in range(len(contours)):
    cnt=contours[i]
    area = cv2.contourArea(cnt)
    if(area>max_area):
        max_area=area
        ci=i
    cnt=contours[ci]

hull = cv2.convexHull(cnt)
#Hand Mask is initially made using skin color alone
handMask = np.zeros(img.shape[:2],np.uint8)
cv2.fillPoly(handMask, pts =[cnt], color=(255,255,255))
handMask = cv2.inRange(handMask, 1, 255)
showImage(handMask,1,"Hand Mask")

isHandDetected = int(input("Enter 1 if hand is detected properly else enter 0: "))
if(isHandDetected):
    res = cv2.bitwise_and(img,img,mask = handMask)
else:
    # noise removal
    opening = cv2.morphologyEx(handMask,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    showImage(sure_bg,1,"Sure BG")
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    showImage(sure_fg,1,"Sure FG")
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    showImage(unknown,1,"Unknown")
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    #img[markers == -1] = [0,0,255]

    m = cv2.convertScaleAbs(markers)
    ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    showImage(thresh,1,"threshold")
    toNot = int(input("Enter 1 to NOT the image and 0 to keep it as such: "))
    if(toNot):
        thresh = cv2.bitwise_not(thresh)
    showImage(thresh,1,"Not-ed threshold")
    res = cv2.bitwise_and(img,img,mask = thresh)
    res = cv2.bitwise_and(res, res, mask=handMask)

showImage(res,2,"Final answer")
cv2.imwrite(outputFile, res);
