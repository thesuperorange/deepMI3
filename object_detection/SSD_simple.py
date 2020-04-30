import cv2 as cv
import os

if __name__ == '__main__':

    prototxt = 'SSD_model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
    weightsPath = 'SSD_model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'


    labelsPath = 'labels/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    print(LABELS)
    cvNet = cv.dnn.readNetFromTensorflow(weightsPath,prototxt)


    img = cv.imread('pizza.jpg')

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        idx =int(detection[1])-1
        if score > 0.3:
            print('{} {} {}'.format(idx,LABELS[idx],score))
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            cv.imwrite('output.png', img)
