# import the necessary packages
import cv2
from .getcontours import ContoursDetect
from .shapedetector import ShapeDetector
import numpy as np
import imutils


class RemoveBorder:
    def __init__(self):
        self.approx_param = 0.04
        pass

    def remove_border(self, img):
        # default remove black border
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
            if (shape is "square" or shape is "rectangle") and (cv2.contourArea(contour) > areas) : #or shape is "square"
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
