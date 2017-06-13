import cv2
import argparse


def main(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 7,
                                      minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    for (x, y , w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Face from a image")
    parser.add_argument('-i', "--image", action="store", help="Specify the input image", required=True,
                        type=str, dest="image")
    results = parser.parse_args()
    main(results.image)
