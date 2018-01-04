# import the necessary packages
import cv2


class ShapeDetector:
    def __init__(self):
        self.approx_param = 0.04
        pass

    def detect(self, cnts, req_shape="rectangle"):
        detect_lst = []
        for i, c in enumerate(cnts):
            # initialize the shape name and approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_param * peri, True)

            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                shape = "rectangle"
                '''
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)

                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                if (ar >= 0.95) and (ar <= 1.05):
                    shape = "square"
                else:
                    shape = "rectangle"
                '''

            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"

            # otherwise, we assume the shape is a circle
            else:
                shape = "unknown"
            if shape == req_shape: #or shape == "unknown":
                detect_lst.append([i, approx, cv2.contourArea(c)])

        # return the name of the shape
        return detect_lst
