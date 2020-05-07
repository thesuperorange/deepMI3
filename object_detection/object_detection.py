# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
# --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import time

CONFIDENCE_TH = 0.3


def mask_detector(net, image):
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))

    bbList = []
    confList = []
    classList = []
    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > CONFIDENCE_TH:
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            bbList.append((startX, startY, endX, endY))
            confList.append(confidence)
            classList.append(classID)
    return confList, classList, bbList

def SSD_detector(net, image):
    (h, w) = image.shape[:2]

    #blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

    # print("[INFO] computing object detections...")
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    print("[INFO] took {:.6f} seconds".format(end - start))

    bbList = []
    confList = []
    classList = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_TH:
            classID = int(detections[0, 0, i, 1])
            if label_dataset == 'coco':
                classID -= 1

            class_name = LABELS[classID].strip().replace(" ", "_")


            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            bbList.append((startX, startY, endX, endY))
            confList.append(confidence)
            classList.append(classID)

    return confList, classList, bbList


def YOLO_detector(net, image):

    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    bbList = []
    confList = []
    classList = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE_TH:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))

                bbList.append((startX, startY, startX+width, startY+height))
                confList.append(confidence)
                classList.append(classID)

    return confList, classList, bbList

def detect(net, detector_name, image, fo , mode_img):

    if detector_name == 'SSD':
        conflist, classidlist, bblist = SSD_detector(net, image)
    elif detector_name == 'mask':
        conflist, classidlist, bblist = mask_detector(net, image)
    else:
        conflist, classidlist, bblist = YOLO_detector(net, image)

    for id, (startX, startY, endX, endY) in enumerate(bblist):
        class_id = classidlist[id]
        confidence = conflist[id]
        class_name = LABELS[class_id].strip().replace(" ", "_")
        class_color = COLORS[class_id]
        if mode_img:
            text = "{}: {:.2f}%".format(class_name, confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY), class_color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2
            )
        fo.write(class_name + ' ' + str(confidence) + ' ' + str(startX) + ' ' + str(startY) + ' ' + str(
                endX) + ' ' + str(endY)+"\n")

    if mode_img:
        output_name = output_img_folder + "/" + filename
        print(output_name)
        cv2.imwrite(output_name, image)


def get_model(detector_name):
    if detector_name == 'SSD':
        ssd_path = 'SSD_model'
        prototxt = ssd_path+'/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
        weightsPath = ssd_path+'/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
        print("[INFO] loading SSD from disk...")
        Detector = cv2.dnn.readNetFromTensorflow(weightsPath, prototxt)

    elif detector_name =='mask':
        param_mask_rcnn = "mask-rcnn-coco"
        weightsPath = os.path.sep.join([param_mask_rcnn, "frozen_inference_graph.pb"])
        configPath = os.path.sep.join([param_mask_rcnn, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
        print("[INFO] loading Mask R-CNN from disk...")
        Detector = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    else :
        yolo_path = "yolo-model"
        weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
        configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])
        print("[INFO] loading YOLO from disk...")
        Detector = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return Detector

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", required=False, default='YOLO', help="SSD/YOLO/mask")
    ap.add_argument("-i", "--input_path", required=True, help="input image folder")
    ap.add_argument('-v', '--visualize', action='store_true',
                    help="whether or not we are going to visualize each instance")
    ap.add_argument('-o', '--output_folder', required=True, help="output results folder")

    args = vars(ap.parse_args())
    detector_name = args['detector']
    vis = args['visualize']
    input_folder = args['input_path']
    output_folder = 'output/'+args['output_folder']




    # load training dataset class
    class_label_path = 'labels'
    label_dataset = 'coco'
    labelsPath = os.path.sep.join([class_label_path, label_dataset + ".names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

    detector = get_model(detector_name)

    output_result_folder = os.path.join(output_folder, 'detection_results')
    output_img_folder = os.path.join(output_folder, 'output_images')
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    if not os.path.exists(output_result_folder):
        os.makedirs(output_result_folder)


    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(input_folder + "/" + filename)
            fo2 = open(output_result_folder + '/' + filename.replace('jpg', 'txt'), "w")

            detect(detector, detector_name, image, fo2,  vis)
            fo2.close()