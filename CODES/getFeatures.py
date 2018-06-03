import numpy as np
from skimage import exposure
from skimage import feature
import cv2
import pandas as pd

def showImage(image,secondsToWait,heading,width=450,height=500):
    #Show Image
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    imageToShow = cv2.resize(image, (width, height)) 
    cv2.imshow(heading,imageToShow)
    secondsToWait = secondsToWait * 1000
    cv2.waitKey(secondsToWait)    

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
    
rowsOfDataset = []
for user in range(1,8):
    for sign in range(1,30):
        for trial in range(1,3):
            if(sign!=10 and sign!=19 and sign!=21 and sign!=23):
                path = '<PATH>/DATASET/USER-PROCESSED-'+str(user)+'/'+str(user)+'-'+str(sign)+'-'+str(trial)+'.jpg'                
                img = cv2.imread(path)
                #showImage(img,1,"img")
                img = cv2.resize(img, (64,64))                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                
                #showImage(gray,1,"gray")
                                
                
                #HOG                
                (H, hogImage) = feature.hog(gray,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
                hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
                hogImage = hogImage.astype("uint8")
                #showImage(hogImage,1,"HOG")
                HOGfeatures = np.ravel(hogImage)
                
                #features = np.append(HOGfeatures,LBPfeatures)
                labelledFeatures = np.append(HOGfeatures,sign)
                rowsOfDataset.append(labelledFeatures)                
df = pd.DataFrame(rowsOfDataset)
df.to_csv('datasetToLearn.csv',columns=None,header=False,index=False)
print(df.shape," written into file 'datasetToLearn'")
