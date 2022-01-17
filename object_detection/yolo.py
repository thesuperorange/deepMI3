# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

confidence_threshold = 0.3  # args["threshold"]
nms_threshold = 0.5
param_visualize = 0
METHOD='yolo'

def detect(dataset, foldername, filename, ch, mode_img, bbox_log):
    image_num = os.path.splitext(filename)[0]
    output_folder = 'output/' +METHOD+'_'+ dataset + "_ch" + str(ch)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(foldername + "/" + filename)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()

    #ln = ln[ net.getUnconnectedOutLayers()[-1] -1]
    
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        #print(output.size)
        # loop over each of the detections
        for detection in output:
            #print(detection)
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                if bbox_log:
                    fo.write(
                        str(ch) + "," + image_num + "," + str(x) + "," + str(y) + "," +
                        str(x + width) + "," + str(y + height) + "," + str(confidence) + "," + LABELS[classID] + "\n"
                    )

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0 and mode_img:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # show the output image
    if mode_img:
        output_name = output_folder + "/" + filename
        print(output_name)
        cv2.imwrite(output_name, image)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="input image dataset")
    ap.add_argument("-i", "--input_path", required=True, help="input image folder")
    ap.add_argument('-v', '--visualize', action='store_true',
                    help="whether or not we are going to visualize each instance")
    ap.add_argument('-l', '--savelog', action='store_true', help="whether or not print results in a file")

    args = vars(ap.parse_args())
    dataset = args['dataset']
    vis = args['visualize']
    log = args['savelog']
    MI3path = args['input_path']
    class_label_path = 'labels'
    label_dataset = 'coco'
    print('args: dataset={} input_path={} vis={} log={}'.format(dataset,MI3path,vis,log))
    # load the COCO class labels our YOLO model was trained on
    yolo_path = "yolo-model"
    labelsPath = os.path.sep.join([class_label_path, label_dataset+".names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
#    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    net = cv2.dnn.readNet(configPath, weightsPath)
    method = 'yolo'
    #    dataset = 'Pathway1_1'
    fo = open(method + '_' + dataset + ".txt", "w")
    channel_list = [2, 4, 6]

    for channel in channel_list:
        input_folder = os.path.join(MI3path, dataset, "ch" + str(channel))
        for filename in os.listdir(input_folder):
            detect(dataset, input_folder, filename, channel, vis,log)
    # detect(input_folder, filename,output_folder='output/'+dataset+'ch'+channel)
    fo.close()
