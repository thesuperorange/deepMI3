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
METHOD = 'SSD'


def detect(dataset, foldername, filename, ch, mode_img, bbox_log, output_result_folder, output_img_folder):
    image_num = os.path.splitext(filename)[0]
    fo2 = open(output_result_folder + '/' + dataset + '_' + image_num + ".txt", "w")

    # print(foldername+"/"+filename)
    image = cv2.imread(foldername + "/" + filename)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # print("[INFO] computing object detections...")
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    print("[INFO] took {:.6f} seconds".format(end - start))

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_TH:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(LABELS[idx], confidence * 100)
            # print("[INFO] {}".format(label))
            class_name = LABELS[idx].strip().replace(" ", "_")
            if mode_img:
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
                )

            if bbox_log:
                fo.write(
                    str(ch) + "," + image_num + "," + str(startX) + "," + str(startY) + "," +
                    str(endX) + "," + str(endY) + "," + str(confidence) + "," + class_name + "\n"
                )
                fo2.write(class_name + ' ' + str(confidence) + ' ' + str(startX) + ' ' + str(startY) + ' ' + str(
                    endX) + ' ' + str(endY)+"\n")

    if mode_img:
        # show the output image
        # cv2.imshow("Output", image)
        #		output_folder = 'output/'+METHOD+'_'+dataset+"_ch"+str(ch)
        #		if not os.path.exists(output_folder):
        #			os.mkdir(output_folder)
        output_name = output_img_folder + "/" + filename
        print(output_name)
        cv2.imwrite(output_name, image)
    # cv2.waitKey(0)
    fo2.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="input image dataset")
    ap.add_argument("-i", "--input_path", required=True, help="input image folder")

    ap.add_argument('-v', '--visualize', action='store_true',
                    help="whether or not we are going to visualize each instance")
    ap.add_argument('-l', '--savelog', action='store_true', help="whether or not print results in a file")
    ap.add_argument('-o', '--output_folder', required=True, help="output results folder")

    args = vars(ap.parse_args())
    dataset = args['dataset']
    vis = args['visualize']
    log = args['savelog']
    MI3path = args['input_path']
    output_folder = 'output/'+args['output_folder']

    class_label_path = 'labels'
    label_dataset = 'coco'
    #    prototxt = 'SSD_model/ssd_mobilenet_v1_coco_2017_11_17.pbtxt'
    #    prototxt = 'SSD_model/ssd_mobilenet_v2_coco_2018_03_29/saved_model/saved_model.pb'
    prototxt = 'SSD_model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
    #    prototxt = "SSD_model/ssd_inception_v2_coco_2017_11_17.pbtxt"
    #    prototxt = "SSD_model/MobileNetSSD_deploy.prototxt.txt"
    model = "SSD_model/MobileNetSSD_deploy.caffemodel"

    # modelConfiguration = "yolov3-tiny.cfg"
    # modelBinary = "yolov3.weights"

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    labelsPath = os.path.sep.join([class_label_path, label_dataset + ".names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")

    #    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    weightsPath = 'SSD_model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    net = cv2.dnn.readNetFromTensorflow(weightsPath, prototxt)
    # net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelBinary);
    method = 'SSD'
    #    dataset = 'Pathway1_1'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fo = open(output_folder + '/' + method + '_' + dataset + ".txt", "w")

    channel_list = [2, 4, 6]
    for channel in channel_list:
        input_folder = os.path.join(MI3path, dataset, "ch" + str(channel))
        output_result_folder = os.path.join(output_folder, method + '_ch' + str(channel) + '_' + dataset)
        output_img_folder = os.path.join(output_folder, 'output_images', dataset + '_ch' + str(channel))
        if not os.path.exists(output_img_folder):
            os.makedirs(output_img_folder)
        if not os.path.exists(output_result_folder):
            os.makedirs(output_result_folder)

    for filename in os.listdir(input_folder):
        detect(dataset, input_folder, filename, channel, vis, log, output_result_folder, output_img_folder)
    fo.close()
