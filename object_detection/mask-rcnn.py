# USAGE
# python mask_rcnn.py --mask-rcnn mask-rcnn-coco --image images/example_01.jpg
# python mask_rcnn.py --mask-rcnn mask-rcnn-coco --image images/example_03.jpg --visualize 1

# import the necessary packages
import numpy as np
import argparse
import random
import time
import cv2
import os

param_threshold = 0.3 #args["threshold"]
param_confidence = 0
param_visualize = 0
METHOD='mask-rcnn'

def detect(dataset, foldername, filename, ch, mode_img, bbox_log):
    image_num = os.path.splitext(filename)[0]
    output_folder = os.path.join('output', METHOD+'_'+dataset+ "_ch" + str(ch))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(foldername+"/"+filename)
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
    print("[INFO] boxes shape: {}".format(boxes.shape))
    print("[INFO] masks shape: {}".format(masks.shape))


    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > param_confidence:
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            if bbox_log:
                fo.write(
                    str(ch) + "," + image_num + "," + str(startX) + "," + str(startY) + "," +
                    str(endX) + "," + str(endY) + "," + str(confidence) + "," + LABELS[classID] + "\n"
                )
            if mode_img:
            # clone our original image so we can draw on it
                clone = image.copy()

            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box

                boxW = endX - startX
                boxH = endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                    interpolation=cv2.INTER_NEAREST)
                mask = (mask > param_threshold)

            # extract the ROI of the image
                roi = clone[startY:endY, startX:endX]

            # check to see if are going to visualize how to extract the
            # masked region itself
                if param_visualize > 0:
                # convert the mask from a boolean to an integer mask with
                # to values: 0 or 255, then apply the mask
                    visMask = (mask * 255).astype("uint8")
                    instance = cv2.bitwise_and(roi, roi, mask=visMask)

                # show the extracted ROI, the mask, along with the
                # segmented instance
                    cv2.imshow("ROI", roi)
                    cv2.imshow("Mask", visMask)
                    cv2.imshow("Segmented", instance)

            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
                roi = roi[mask]

            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
                color = random.choice(COLORS)
                blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original image
                clone[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the image
                color = [int(c) for c in color]
                cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
                text = "{}: {:.4f}".format(LABELS[classID], confidence)
                cv2.putText(clone, text, (startX, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # show the output image

                split_filename = os.path.splitext(filename)
                output_name = output_folder + "/" + split_filename[0]+"_"+str(i)+split_filename[1];
                print(output_name)
                cv2.imwrite(output_name, clone)
            #cv2.imshow("Output", clone)
            #cv2.waitKey(0)

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="input image dataset")
    ap.add_argument("-i", "--input_path", required=True, help="input image folder")
    
    # ap.add_argument("-m", "--mask-rcnn", required=True,
    # 	help="base path to mask-rcnn directory")i
    ap.add_argument('-v', '--visualize', action='store_true',help="whether or not we are going to visualize each instance")
    ap.add_argument('-l', '--savelog', action='store_true',help="whether or not print results in a file")	

    # 	help="minimum probability to filter weak detections")
    # ap.add_argument("-t", "--threshold", type=float, default=0.3,
    # 	help="minimum threshold for pixel-wise mask segmentation")
    args = vars(ap.parse_args())
    dataset = args['dataset']
    vis = args['visualize']
    log =args['savelog']
    class_label_path = 'labels'
    label_dataset = 'coco'

    param_mask_rcnn="mask-rcnn-coco"

  # load the COCO class labels our Mask R-CNN was trained on
    labelsPath = os.path.sep.join([class_label_path,label_dataset+'.name'])
    LABELS = open(labelsPath).read().strip().split("\n")

    # load the set of colors that will be used when visualizing a given
    # instance segmentation
    colorsPath = os.path.sep.join([param_mask_rcnn, "colors.txt"])
    COLORS = open(colorsPath).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = os.path.sep.join([param_mask_rcnn,"frozen_inference_graph.pb"])
    configPath = os.path.sep.join([param_mask_rcnn,"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    #img_path ='C:/Users/superorange/Videos/MI3_dataset/Pathway2_3/ORIG/ch4/00151.png'
    #img_path='images/example_02.jpg'

    MI3path = args['input_path']
    method='mask-rcnn'
#    dataset = 'Pathway1_1'
    fo = open(method+'_'+dataset+ ".txt", "w")
    channel_list = [2,4,6]
    for channel in channel_list:
        input_folder = os.path.join(MI3path,dataset,"ch"+str(channel))
        for filename in os.listdir(input_folder):
            print(filename)
            detect(dataset, input_folder, filename, channel,vis,log)
            #detect(input_folder, filename,output_folder='output/'+dataset+'ch'+channel)
    fo.close()
