# import the necessary packages
import cv2
from .getcontours import ContoursDetect
from .shapedetector import ShapeDetector
import numpy as np
import imutils


class RemoveBorder:
    def __init__(self, approx_param=0.04, canny_minv=50, canny_maxv=150, blur_kernel=(5, 5)):
        self.approx_param = approx_param
        self.canny_minv = canny_minv
        self.canny_maxv = canny_maxv
        self.blur_kernel = blur_kernel
        pass

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)  # The kernel may be changed according to real image
        return cv2.Canny(blurred, self.canny_minv, self.canny_maxv, L2gradient=True)

    def remove_border(self, img):
        # default remove black border
        # use canny get the gradient
        processimg = self.preprocess(img)
        contourD = ContoursDetect()
        contours = contourD.get_contours(img)
        shapeD = ShapeDetector()

        # Find with the largest rectangle
        areas = 0
        max_contour = None
        print(len(contours))
        for contour in contours:
            # initialize the shape name and approximate the contour
            shape = shapeD.detect(contour)
            if (shape is "square" or shape is "rectangle") and (cv2.contourArea(contour) > areas):
                areas = cv2.contourArea(contour)
                max_contour = contour
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            resized = imutils.resize(img, width=1500)
            cv2.imshow("Image", resized)
            cv2.waitKey(0)
        if max_contour is not None:
            cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)
            resized = imutils.resize(img, width=1500)
            cv2.imshow("Image", resized)
            cv2.waitKey(0)
            x, y, w, h = cv2.boundingRect(max_contour)
            '''
            # Ensure bounding rect should be at least 16:9 or taller
            if w / h > 16 / 9:
                # increase top and bottom margin
                newHeight = w / 16 * 9
                y = y - (newHeight - h) / 2
                h = newHeight
            '''
            return x, y, w, h
        return None, None, None, None
