import cv2, os
import numpy as np
import csv
import glob

label = "Uninfected"
#Setting the general path for images using glob
dirList = glob.glob("cell_images/" + label + "/*.png")
#Opening or creating the csv file in csv folder which will contain the dataset
file = open("csv/dataset.csv", "a")

for img_path in dirList:
    #Reading the image
    img = cv2.imread(img_path)
    #Blurring with Gaussian Blur
    img = cv2.GaussianBlur(img,(5,5), 2)
    #Convertig to Grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Implementing thresholding
    ret, thresh = cv2.threshold(imgGray, 127, 255, 0)
    #Finding Contours
    contours,_ = cv2.findContours(thresh, 1, 2)

    #Drawing Contours on images
    for contour in contours:
        cv2.drawContours(imgGray, contours, -1, (0, 255, 0), 3)

    #Writingt the label and separating the area into different tabs in csv file
    file.write(label)
    file.write(",")

    #Getting the area of contours
    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            #Writing areas in csv file
            file.write(str(area))
        except:
            file.write("0")
        #Separting the values in csv file
        file.write(",")
    file.write("\n")


     