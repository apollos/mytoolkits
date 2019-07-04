# import the necessary packages
import cv2
import numpy as np
import math
import imutils
import mylogs

recordLogs = mylogs.myLogs()


class RemoveBorder:
    def __init__(self, approx_param=0.04, canny_minv=50, canny_maxv=95, blur_kernel=(5, 5), resize_threshold=1024,
                 resize_target=800):
        self.approx_param = approx_param
        self.canny_minv = canny_minv
        self.canny_maxv = canny_maxv
        self.blur_kernel = blur_kernel
        self.resize_threshold = resize_threshold
        self.resize_target = resize_target
        pass

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)  # The kernel may be changed according to real image
        return cv2.Canny(blurred, self.canny_minv, self.canny_maxv, L2gradient=True)

    def find_border_angel(self, mid_point, image, direction='horizon', reverse_flag=0):
        kernel = [255, 255, 255, 255, 255]
        bias = 100
        image_shape = np.shape(image)
        degree = 0
        if direction == 'horizon':
            bias = min(image_shape[1] - mid_point[0] - int(len(kernel) / 2), mid_point[0] - int(len(kernel) / 2),
                       bias + int(len(kernel) / 2))
            if reverse_flag == 0:
                search_range = range(image_shape[0])
            else:
                search_range = reversed(range(image_shape[0]))
            a_point_y = -1
            b_point_y = -1
            for i in search_range:
                a_tmp = np.dot(image[i, mid_point[0] - bias - 2:mid_point[0] - bias + 3], kernel)
                b_tmp = np.dot(image[i, mid_point[0] + bias - 2:mid_point[0] + bias + 3], kernel)
                if a_tmp >= 130050:
                    a_point_y = i
                if b_tmp >= 130050:
                    b_point_y = i
                if a_point_y != -1 and b_point_y != -1:
                    break
            #print("{} - {}".format(b_point_y, a_point_y))
            if a_point_y != -1 and b_point_y != -1:
                degree = math.degrees(math.atan2(b_point_y - a_point_y, 2 * bias))
                mid_point[1] = round(math.tan(math.radians(degree)) * bias) + a_point_y
            else:
                recordLogs.logger.warn("Cannot find horizon edge")
        elif direction == 'vertical':
            bias = min(image_shape[0] - mid_point[1] - int(len(kernel) / 2), mid_point[1] - int(len(kernel) / 2),
                       bias + int(len(kernel) / 2))
            if reverse_flag == 0:
                search_range = range(image_shape[1])
            else:
                search_range = reversed(range(image_shape[1]))
            a_point_x = -1
            b_point_x = -1
            for i in search_range:
                a_tmp = np.dot(image[mid_point[1] - bias - 2:mid_point[1] - bias + 3, i], kernel)
                b_tmp = np.dot(image[mid_point[1] + bias - 2:mid_point[1] + bias + 3, i], kernel)
                if a_tmp >= 130050:
                    a_point_x = i
                if b_tmp >= 130050:
                    b_point_x = i
                if a_point_x != -1 and b_point_x != -1:
                    break
            #print("{} - {}".format(b_point_x, a_point_x))
            if a_point_x != -1 and b_point_x != -1:
                degree = 0 - math.degrees(math.atan2(b_point_x - a_point_x, 2 * bias))  # same degree
                mid_point[0] = round(math.tan(math.radians(degree)) * bias) + a_point_x
            else:
                recordLogs.logger.warn("Cannot find vertical edge")
        else:
            recordLogs.logger.error("Unsupported Direction {}".format(direction))
        # print("({}, {})".format(mid_point[0], mid_point[1]))
        return degree

    def resize_img(self, img):
        if np.shape(img)[1] > self.resize_threshold:
            resized = imutils.resize(img, width=self.resize_target)
        else:
            resized = img
        return img.shape[0] / float(resized.shape[0]), resized

    def remove_border(self, img, margin=0):
        # default remove black border
        # use canny get the gradient

        ratio, resized = self.resize_img(img)
        '''
        cv2.imshow("image", processimg)
        cv2.waitKey(0)
        '''
        processimg = self.preprocess(resized)

        top_mid = [int(np.shape(processimg)[1] / 2), 0]
        bottom_mid = [int(np.shape(processimg)[1] / 2), np.shape(processimg)[0]]
        left_mid = [0, int(np.shape(processimg)[0] / 2)]
        right_mid = [np.shape(processimg)[1], int(np.shape(processimg)[0] / 2)]

        t_rotate_degree = self.find_border_angel(top_mid, processimg, direction='horizon')
        b_rotate_degree = self.find_border_angel(bottom_mid, processimg, direction='horizon', reverse_flag=1)
        l_rotate_degree = self.find_border_angel(left_mid, processimg, direction='vertical')
        r_rotate_degree = self.find_border_angel(right_mid, processimg, direction='vertical', reverse_flag=1)
        #print(top_mid, bottom_mid, left_mid, right_mid)
        top_mid = np.array(np.array(top_mid).astype('float') * ratio).astype('int')
        bottom_mid = np.array(np.array(bottom_mid).astype('float') * ratio).astype('int')
        left_mid = np.array(np.array(left_mid).astype('float') * ratio).astype('int')
        right_mid = np.array(np.array(right_mid).astype('float') * ratio).astype('int')
        xmin = max(left_mid[0] - margin, 0)
        ymin = max(top_mid[1] - margin, 0)
        xmax = min(right_mid[0] + margin, np.shape(img)[1])
        ymax = min(bottom_mid[1] + margin, np.shape(img)[0])
        #print(xmin, ymin, xmax, ymax)

        return xmin, ymin, xmax, ymax
