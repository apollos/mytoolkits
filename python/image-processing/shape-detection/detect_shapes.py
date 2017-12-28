# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
print("{}-{}-{}".format(image.shape[0], image.shape[1], image.shape[2]))
'''resized = imutils.resize(image, width=300)'''
resized = image
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
'''cv2.imshow("blurred", blurred)
cv2.waitKey(0)
print(blurred)'''
print("{}-{}".format(blurred.shape[0], blurred.shape[1]))
tmp = np.ravel(blurred)
print("{} {} {} {}".format(np.max(tmp), np.min(tmp), np.mean(tmp), np.median(tmp)))
thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
#print(np.shape(cnts))
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    #print(np.shape(c))
    if(cv2.contourArea(c) < 90000) or (cv2.contourArea(c) > 110000):
        continue
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    if shape is not "rectangle" and shape is not "square":
        continue

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(c)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)