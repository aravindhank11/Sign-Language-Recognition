import cv2
import numpy as np
import sys
from skimage import exposure
from skimage import feature
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


f = open('sign.txt','w+')

font = cv2.FONT_HERSHEY_SIMPLEX

def LBP(image, eps=1e-7, numPoints=32, radius=8):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist


df = pd.read_csv('datasetToLearn.csv',header=None)
X = df.iloc[:,:-1] 
Y = df.iloc[:,-1] 
etc = ExtraTreesClassifier(n_estimators=1000)
scores = cross_val_score(etc, X, Y, cv=10)
print('Extra Tree',"Mean CV Score: {:.3f}".format(scores.mean()))
etc.fit(X, Y)

cap = cv2.VideoCapture(0)
while(1):
    #Capture frames from the camera
    ret, frame = cap.read()
    #img = cv2.flip(frame,1)
    img=frame.copy()
    kernel = np.ones((5,5),np.uint8)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    frameHSV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)    

    skin_min = np.array([0,133,77],np.uint8) #100, 133, 100
    skin_max = np.array([255,173,127],np.uint8) #255, 255, 255     

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

    if('cnt' in globals()):
        #Hand Mask is initially made using skin color alone
        handMask = np.zeros(img.shape[:2],np.uint8)
        cv2.fillPoly(handMask, pts =[cnt], color=(255,255,255))
        handMask = cv2.inRange(handMask, 1, 255)
        res = cv2.bitwise_and(img,img,mask = handMask)
        x,y,w,h = cv2.boundingRect(cnt)    
        crop_img = res[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (64,64))                
        
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)                
        #LBPfeatures = LBP(gray)                
        #HOG
        (H, hogImage) = feature.hog(gray,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        #showImage(hogImage,1,"HOG")
        cv2.imshow("HOG",hogImage)
        HOGfeatures = np.ravel(hogImage)
        
        #features = np.append(HOGfeatures,LBPfeatures)
        predicted = etc.predict([HOGfeatures])
        #showImage(res,2,"Final answer")
        cv2.putText(res,str(predicted),(100,100),font,2,(255,255,255),2)
                
        cv2.imshow("Final",res)
        k = cv2.waitKey(5) & 0xFF
        
        if k == 27:
            f.write(str(predicted))
        
        if(predicted == 5):
            break


cap.release()
cv2.destroyAllWindows()
f.close()
"""
hull = cv2.convexHull(cnt,returnPoints = False)    
    #hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    print(defects.shape)"""
