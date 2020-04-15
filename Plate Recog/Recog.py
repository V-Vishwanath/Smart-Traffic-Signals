import cv2
import imutils
import argparse
import numpy as np
import requests
import pytesseract as ocr


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		   "sofa", "train", "TV monitor"]

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'model.caffemodel')

parser = argparse.ArgumentParser(description='Extract License Plate information from vehicles')
parser.add_argument('--image', '-i', help='image to perform detection')
parser.add_argument('--status', '-s', help='Location of the vehicle')

args = parser.parse_args()

img = cv2.imread(args.image)
status = args.status

h, w = img.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

net.setInput(blob)
detections = net.forward()

obj = ''
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])
        obj = CLASSES[idx]

if obj == 'car' :
    win1 = cv2.namedWindow('Car Image', cv2.WINDOW_NORMAL)
    win2 = cv2.namedWindow('Plate Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Image', 1080, 1080)
    cv2.resizeWindow('Plate Image', 1080, 1080)

    img = cv2.resize(img, (640, 480))

    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = cv2.bilateralFilter(gs, 11, 17, 17)

    edges = cv2.Canny(gs, 30, 200)

    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    area = None

    for i in contours :
        peri = cv2.arcLength(i, True)
        sides = cv2.approxPolyDP(i, 0.018*peri, True)
        if len(sides) == 4 :
            area = sides
            break

    mask = np.zeros(gs.shape, np.uint8)
    plate = cv2.drawContours(mask, [area], 0, 255, -1)
    plate = cv2.bitwise_and(img, img, mask=mask)

    x,y = np.where(mask == 255)
    sx,sy,ex,ey = np.min(x), np.min(y), np.max(x), np.max(y)

    plateImg = gs[sx:ex+1, sy:ey+1]

    cv2.imshow('Car Image', img)
    cv2.imshow('Plate Image', plateImg)

    plateNo = ocr.image_to_string(plateImg)
    print(f'Plate number Identified : {plateNo}')

    # endPoint = 'https://carparkingecs.000webhostapp.com/getData.php?token=aBu78&v_no=' + str(plateNo) + '&status=' + str(status)
    
    # res = requests.get(url=endPoint)

    if cv2.waitKey(0) == 27 :
        cv2.destroyAllWindows()