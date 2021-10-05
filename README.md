# Lane-Detection-ALgorithm

## Description About the Project:-

# *Problem Statement:*

A lane detection car *can easily detect lanes* and can travel anywhere a normal car does. Explorations in AI and Deep Learning have made such innovations possible with essential training. Lane detection is the primary and the most important step in the deployment of an autonomous car.

### *Steps involved in Lane Detection:*
→Reading the image. If reading a video convert into frames.
→Conversion of the image into GRAY scale
→Using filter or Gaussian blur to get a clear image
→Detect the edges using Canny function
→Create a mask for the Canny Image
→Identify the coordinates of the lane
→Fit the detected coordinates into the Canny Image
→Lane detection is completed.

## Gist about Coding:
1. Reading the Image/Video: In case of a video, we first capture the video by using the VideoCapture( ) function. From this we use the .read( ) function to get each and every frame.
2. Conversion of the Image into GRAY scale: The frame that we have are in BGR (Blue, Green, Red). Now we convert this to a gray scale using cv2.cvtColor( ) function by passing the method cv2.COLOR_BGR2GRAY.
3. Using filter or Gaussian blue to get a clear image: In order to reduce the noise or blur in the frame we use the “.filter( )” or “.Gaussianblur( )” function.
4. Detect the edges using Canny function: The “.Canny( )” function can be used to detect all the edges in the frame.
5. Identify the coordinates of the lane: Now we detect the dimensions of the road lane and a mask is created which is of the same dimensions as the frame using the “.mask( )”. The bitwise AND function is then used between the canny image and the mask.

## ***Comprehending The Code!!***

**The following explanation of code works only for the given images and videos, depending upon the region of interest.**

To perform lane detection the Libraries used are:

- Opencv
- Numpy
- matplotlib.pyplot

***Reading The Image:***
Initially we start reading the image
### The path used here is for testing purpose, user can define the path of image on his/her own.

### ***Converting the image into Gray:***
After reading the image, we then convert it to grayscale. This is done to enhance the contrast of the lanes concerning the road. To smoothen the image, and reduce the noise we use the Gaussian-Blur function.

### ***Canny (Edge Detection):***

***The Canny function is used to distinguish the edges of all objects in the image.***

Bringing the whole code for Canny Detection at once...
```
import cv2
import numpy as np
path = "/*give the path of the image*/"
image = cv2.imread(path)
lane_image = np.copy(image)

#converting the lane_image into gray color
gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
#Blurring the image for better detection using Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#Using Canny method for better detectionn of the edges

canny = cv2.Canny(blur, 50, 150)
cv2.imshow("Canny Image", canny)
cv2.waitKey()
```
## **Unearthing the Region of Interest:**
Since we are dealing with lane detection, we want only to detect the lanes. Hence, we need to stipulate the precincts which have the lanes within it. To particularize the points for the region, we use matplotlib.pyplot which gives us a scale.

```
import cv2
import numpy as np
import matplotlib.pyplot as pi

path = "/*give the path of the image*/"
image = cv2.imread(path)
lane_image = np.copy(image)

gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)
pi.imshow(canny)
pi.show()
```
The image provides us with the x and y coordinates to specify the region of interest. The points chosen are [[(200,700),(1100,700),(550,250)]]. The points are such that we get a triangle.

Now by using these coordinates, we merge it with a blank image. This is accomplished so that, *only the region of interest is show in the blank image.*

```
import cv2
import numpy as np
import matplotlib.pyplot as pi

path = "/*give the path of the image*/"
image = cv2.imread(path)
lane_image = np.copy(image)

gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)

height = canny.shape[0]
polygons = np.array([[(200, height), (1100, height), (550, 250)]])
mask = np.zeros_like(image)
final = cv2.fillPoly(mask, polygons, 255)
cv2.imshow("Result", final)
cv2.waitKey()
```
When we utter about a blank image, it is a matrix consisting of factors which are *'zeros*'. Likewise, a white image is a matrix consisting of elements that are *'ones'*. Now, with the help of this image, we shall crop out the undesired region from the canny image using *bitwise_and*. The reason we used bitwise AND is that this function produces an output only "when the bit of two images compared is 1."
Example:

- In the binary system we represent 25 as 11001
- In the binary system, we represent 28 as 11100

```
cropped_image = cv2.bitwise_and(image, mask)
```
## ***Hough Lines:***

This technique is used to detect straight lines and identify the lane boundaries. We know that a line is represented as Y = mX + c in its slope-intercept form. Where m is the slope and c is the intercept.
Consider the XY plane. Let us take a point. There may be diverse lines passing through that point. Note down the slope and intercept of each of those lines. Now we plot these points on another plane, which is known as the *Hough space*.

Now we shall find the points and the draw the lines on a blank image for lane detection:

we use the add weighted function to put these lines to the original image. Add weighted function basically helps improve the contrast of the lines with respect to the road.

## ***Applications:***

### *Self-Driving Cars:*

The fundamental feature of a self-driving car is to find the route accurately. A typical human being can achieve this effortlessly, but for the mechanism of car to do this takes adequate effort. Lane Detection is crucial in the viewpoint of Self-Driving Cars because the car must be able to pursue and detect the path that it should proceed.

### *Real-time illumination invariant lane detection for lane departure warning system:*

Lane detection is an important element in improving driving safety. We propose a real-time and *illumination invariant lane detection method* for lane departure warning system. The proposed method works well in various illumination conditions such as in *bad weather conditions and at night time*. It includes three major components: First, we detect a *vanishing point* based on a voting map and define an *adaptive region of interest (ROI)* to reduce computational complexity. Second, we utilize the *distinct property of lane colors* to achieve illumination invariant lane marker candidate detection. Finally, we find the main lane using a *clustering method* from the lane marker candidates. In case of lane departure situation, our system sends driver alarm signal.

### *Fast lane detection with Randomized Hough Transform:*

Lane detection is an essential component of *autonomous mobile robot applications.* Any lane detection method has to deal with the varying conditions of the lane and surrounding that the robot would encounter while moving. Lane detection procedure can provide estimates for the *position and orientation* of the robot within the lane and also can provide a reference system for locating other obstacles in the path of the robot. In this paper we present a method for lane detection in video frames of a camera mounted on top of the mobile robot. Given video input from the camera, the *gradient of the current lane* in the near field of view are automatically detected. Randomized Hough Transform is used for *extracting parametric curves* from the images acquired. A prior knowledge of the lane position is assumed for better accuracy of lane detection.

Source: [https://scholar.google.co.in/scholar?q=applications+of+lane+detection&hl=en&as_sdt=0&as_vis=1&oi=scholart](https://scholar.google.co.in/scholar?q=applications+of+lane+detection&hl=en&as_sdt=0&as_vis=1&oi=scholart)

