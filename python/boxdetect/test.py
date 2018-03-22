from boxDetector import checkBoxDetector as cbd
import cv2
import numpy as np

file_path = "images.png"
min_rect_size = 1
filled_coordinates = cbd.get_filled_contours(file_path, min_rect_size)
print(filled_coordinates)

img = cv2.imread(file_path)
for filled_coord in filled_coordinates:
    box = np.int0(filled_coord)
    cv2.drawContours(img, [box], -1, (0,255,0), 0)
cv2.imshow("Result", img)
k = cv2.waitKey(0)



