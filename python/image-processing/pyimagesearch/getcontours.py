# import the necessary packages
import cv2
import imutils


class ContoursDetect:
    def __init__(self, blur_kernel=(5, 5), adaptive_threshold_block_size=9, dilate_kernel=(2,2), erod_kernel=(3,3)):
        self.blur_kernel = blur_kernel
        self.adaptive_threshold_block_size = adaptive_threshold_block_size
        self.adaptive_threshold_C = 2.7
        self.dilate_kernel = dilate_kernel
        self.erod_kernel = erod_kernel
        self.morph_iter = 1
        self.adaptive_threshold_maxV = 255
        pass

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)  # The kernel may be changed according to real image
        thresh = cv2.adaptiveThreshold(blurred, self.adaptive_threshold_maxV, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, self.adaptive_threshold_block_size, self.adaptive_threshold_C)
        # morph coins by eroding and dilating to remove noise
        morph_rst = cv2.dilate(thresh, self.dilate_kernel, iterations=self.morph_iter)
        morph_rst = cv2.erode(morph_rst, self.erod_kernel, iterations=self.morph_iter)
        return morph_rst

    def get_contours(self, img):
        # default remove black border
        thresh = self.preprocess(img)
        '''
        resized = imutils.resize(thresh, width=1500)
        cv2.imshow("Image", resized)
        cv2.waitKey(0)
        '''
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts[0] if imutils.is_cv2() else cnts[1]
