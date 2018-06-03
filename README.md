# Sign-Language-Recognition
Enabling the people with talking disabilities to communicate their thoughts without any obstruction by capturing the hand motion using the techniques  of Image Processing and processing them utilizing Machine Learning Analysis to compose sentences

## Motivation
One of the major drawback of our society is the barrier that is created between disabled or handicapped persons and the normal person. Communication is the only medium by which we can share our thoughts or convey the message but for a person with disability (deaf and dumb) faces difficulty in communication with normal person. Our aim is to design a system to help the person who is hearing impaired to communicate with the rest of the world using sign language or hand gesture recognition techniques by the process of converting the gestures to human readable texts.

## Workflow
1. A video stream of is to be sampled and processed in real time which contains the users showing various signs and the features of the hand gestures (like HOG) is to be extracted.
2. This features is to be considered as test data to previously trained dataset using advanced Machine Learning Models and thus a prediction of hand sign is made
3. Thus, the input video is stream is converted to human readable form sentences with meaning

## Tool Used
> OPEN-CV

## About the Dataset
The dataset consists of RGB images of 25 sign language gestures of the Indian Sign Language taken from 7 subjects. There are a total of 175 images.

## Procedure
1. The Dataset is subjected to hand boundary detection effected by means of strict Skin detection algorithm along with opening, closing, to minimise noise level. 
2. The above steps converts images in directory USER-i-j-k to processed boundary detected hands in directed in USER-PROCESSED-i-j-k.
3. The images in the directory USER-PROCESSED-i-j-k is learnt using ExtraTreeClassifier with 10 fold classification on HOG data description of images and an accuracy of **80%** is achieved. Cross fold classification score of 80 signifies a 80% +/- 5% of test data prediction accuracy.
4. Video is captured in real time, where frames are sampled and is subjected to skin detection, HOG feature extraction and prediction.
5. The predicted stream of data is then converted into text and written in a file
