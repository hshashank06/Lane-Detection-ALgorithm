```python
# Importing the required modules
import cv2
import numpy as np
import matplotlib.pyplot as pi

#Defining a function which converts the image into required format for Lane Detection
def Lanedetect(lane_image):
    #Initially we convert the image to Gray scale
    gray = cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
    #Here we use blurr to reduce the noise in the image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #Then we perform edge detection using the Canny function
    #After edge detection we then find the Region of interest
    canny = cv2.Canny(blur,50,150)
    return canny

#Defining to function to bring the seperate lines into single line and managing the height within the lane
def coordinate(image,lines):
    slope, intercept = lines
    print(slope)
    print(intercept)
    y1 = image.shape[0]
    #Here the length of the line is about 3/5 of the height of the image
		#The value 3/5th can be changed depending on the length of the line to be displayed.
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    print(np.array([x1,y1,x2,y2]))
    
    return np.array([x1,y1,x2,y2])

#Definig the function to find all the obtained slopes(+Ve & -Ve) and passing the slopes to coordinate function
def avg_slp_intercept(image,lines):
    #Left fit are the slopes less that zero
    left_fit = []
    #Right fit are the slopes greater than zero
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        # Linear Polynomial
        #This gives an array with first element as slope and the second as the intercept
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        #Left lane will have negative slope and right lane will have positive slope
        #Based on the slopes, we append to the corresponding lists.
        #As a result we will have many points in the lists
        #Then we shall find the average*/
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
            
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        print(left_fit_average, 'left')
        left_line = coordinate(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        print(right_fit_average, 'right')
        right_line = coordinate(image, right_fit_average)
    
    return np.array([left_line,right_line])
    
#To draw a single striaght of the lane from the obtained points from avg_slp_intercept function       
def merged(image,lines):
    image_blank = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(image_blank,(x1,y1),(x2,y2), (0,0,255),10)
    return image_blank

#Defining a function to find Region Of Interest
def region_of_interest(image):
    height = image.shape[0]
    #We determine the points which include the region of interest
	#These points may vary depending upon the image.
	#These points will not work for all images. Hence the points must be changed depending upon the image.
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    #Here we place that region on a blank image whose color is white
    mask = np.zeros_like(image)
    final = cv2.fillPoly(mask,polygons,255)
    cv2.imshow('result',final)
    #Using bitwise and we place 'final' which has the region on interest on the original image
    #Hence cropped_image will have only the image invovling the ROI 
    
    cropped_image = cv2.bitwise_and(image,final)
    return cropped_image

#Reading the Image
vid =cv2.VideoCapture(r'C:\Users\umasr\Downloads\test2.mp4')
while vid.isOpened() == True:
    success,image = vid.read()
    lane_image = np.copy(image)
    canny = Lanedetect(lane_image)
    region = region_of_interest(canny)
    #Hough Lines is a techniques which is used to form the lines from the given points
    lines = cv2.HoughLinesP(region, 2, np.pi/180,100, np.array([]),minLineLength = 40,maxLineGap = 5)
    #In order to get a single straight line we find the avg
    slope = avg_slp_intercept(lane_image,lines)
    lines_display = merged(lane_image,slope)
    #At last we display the lines on the image use a function addWeighted
    #That is simply used to imrove the contrast on the image wrt to the lane lines
    final_image = cv2.addWeighted(image, 0.8,lines_display , 10, 1)
    #Displaying the final Image.
    cv2.imshow('The image',final_image)
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
```
