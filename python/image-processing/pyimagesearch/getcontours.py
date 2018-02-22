# import the necessary packages
import cv2
import imutils
import numpy as np
from .shapedetector import ShapeDetector
import mylogs
import collections

recordLogs = mylogs.myLogs()


class ContoursDetect:
    def __init__(self, canny_minv=50, canny_maxv=150, blur_kernel=(5, 5), resize_threshold=1024, resize_target=800):
        self.canny_minv = canny_minv
        self.canny_maxv = canny_maxv
        self.blur_kernel = blur_kernel
        self.resize_threshold = resize_threshold
        self.resize_target = resize_target
        pass

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)  # The kernel may be changed according to real image

        return blurred

    def resize_img(self, img):
        if np.shape(img)[1] > self.resize_threshold:
            resized = imutils.resize(img, width=self.resize_target)
        else:
            resized = img
        return img.shape[0] / float(resized.shape[0]), resized

    def cal_square_num(self, shape):
        vertical_num = 6 #fixed for trinasolar
        area = int(int(shape[0]/vertical_num)**2/1000)
        horizon_num = round(vertical_num*shape[1]/shape[0])
        return vertical_num, horizon_num, area

    def get_contours(self, img, shape):
        # default remove black border
        ratio, resized = self.resize_img(img)
        processed = self.preprocess(resized)
        vertical_num, horizon_num, potential_area_ratio = self.cal_square_num(np.shape(processed))
        print(vertical_num, horizon_num, potential_area_ratio)

        cannied = cv2.Canny(processed, self.canny_minv, self.canny_maxv, L2gradient=True)
        cnts = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        shape_detector = ShapeDetector()
        #get [contour index, approx information, cv2.contourArea(c)]
        detect_rst = shape_detector.detect(cnts, shape)
        if len(detect_rst) > 0:
            potential_good_shape = [0, 0] # potential_good_shape[index, area]
            for idx, detect in enumerate(detect_rst):
                if int(detect[2]/1000) == potential_area_ratio and detect[2] > potential_good_shape[1]:
                    potential_good_shape[0] = idx
                    potential_good_shape[1] = detect[2]
            if potential_good_shape[1] != 0:
                (x, y, w, h) = cv2.boundingRect(detect_rst[potential_good_shape[0]][1])
                '''
                minx = int(float(x)*ratio)
                miny = int(float(y)*ratio)
                maxx = minx + int(float(w)*ratio)
                maxy = miny + int(float(h)*ratio)
                cropped = img[miny:maxy, minx:maxx, 0]
                '''
                w = int(float(w)*ratio)
                h = int(float(h)*ratio)
                xmin = 0
                ymin = 0
                concur_list = []

                vertical_margin = int((np.shape(img)[0] - h * vertical_num)/(vertical_num - 1))
                horizon_margin = int((np.shape(img)[1] - w * horizon_num)/(horizon_num - 1))
                margin = min(vertical_margin, horizon_margin)
                for i in range(horizon_num):
                    for j in range(vertical_num):
                        square_dict = collections.defaultdict(dict)
                        square_dict['pos'] = [i, j]
                        square_dict['coordinate'] = [(xmin, ymin), (xmin+w, ymin+h)]
                        concur_list.append(square_dict)
                        ymin += h + margin
                    xmin += w + margin
                    ymin = 0

                for concur in concur_list:
                    cv2.rectangle(img, concur['coordinate'][0], concur['coordinate'][1], (0, 255, 0), 3)
                    print(concur['pos'], concur['coordinate'])

                    resized = imutils.resize(img, width=800)
                    cv2.imshow("image", resized)
                    cv2.waitKey(0)


            else:
                recordLogs.logger.error("Cannot find good contour as sample")
