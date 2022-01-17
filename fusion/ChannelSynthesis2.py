
import pandas as pd
import os
import itertools
import BoundingBoxIOU
from ChannelSynthesis import read_txt_as_dict


if __name__ == '__main__':

    channel_list = [2, 6]
    iter_channel = list(itertools.combinations(channel_list, 2))
    method_list = ['yolo']
   # method_list = ['SSD', 'yolo', 'faster-rcnn', 'mask-rcnn']
    dataset_list = ['Pathway1_1', 'Pathway2_1','Pathway2_2','Pathway2_3', 'Bus_2', 'Staircase_1', 'Doorway_1'
        , 'Doorway_2', 'Room_1']
    #dataset_list = ['Pathway', 'Bus', 'Staircase', 'Doorway', 'Room']
    
    folder = '/home/superorange5/Research/YOLOv4/yolov4_opencv/'


    IOU_th = 0.3
    select_type = 2  # 1 avg, 2 by confidence

    output_type = 1  # 1: output 1 folder/1 txt each image, 2: output 1 file

    for method in method_list:
        for dataset in dataset_list:

            if output_type == 1:

                output_folder = os.path.join(folder, 'merge26' , dataset + '_merge26')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

            inputFile = os.path.join(folder, method+'_'+dataset + '.txt')

            in_dict = read_txt_as_dict(inputFile)

            if output_type == 2:
                fo = open(os.path.join(folder, method + '_' + dataset + '_merge26.txt'), "w")

            # loop in frames
            for idx in in_dict:
                if output_type == 1:

                    output_file_name = dataset + '_ch6_' + str(idx).zfill(5) + ".txt"
                    #output_file_name = idx + ".txt"
                    fo = open(os.path.join(output_folder, output_file_name), "w")

                # print("###"+str(idx))

                iou_table_size = 0
                for ch_tuple in iter_channel:
                    iou_table_size += len(in_dict[idx]["channel" + str(ch_tuple[0])]) \
                                      * len(in_dict[idx]["channel" + str(ch_tuple[1])])
                # store iou between bb in each channel
                iou_table = []

                # only one channel contains bb
                if iou_table_size == 0:
                    for ch in channel_list:
                        if len(in_dict[idx]["channel" + str(ch)]) != 0:
                            for A_idx, bbA in enumerate(in_dict[idx]["channel" + str(ch)]):
                                in_dict[idx]["channel" + str(ch)][A_idx]["tag"] = True
                            break
                else:
                    # iou_idx = 0
                    for ch_tuple in iter_channel:
                        for A_idx, bbA in enumerate(in_dict[idx]["channel" + str(ch_tuple[0])]):
                            for B_idx, bbB in enumerate(in_dict[idx]["channel" + str(ch_tuple[1])]):
                                if bbA["type"] == bbB["type"]:
                                    iou = BoundingBoxIOU.get_iou(bbA["bbox"], bbB["bbox"])
                                    if iou > IOU_th:
                                        iou_table.append([iou, ch_tuple[0], A_idx, ch_tuple[1], B_idx, bbA["type"]])

                iou_table = sorted(iou_table, key=lambda x: x[0], reverse=True)

                for iou_item in iou_table:
                    iouAB = iou_item[0]
                    chA = iou_item[1]
                    bbA = iou_item[2]
                    chB = iou_item[3]
                    bbB = iou_item[4]
                    obj_type = iou_table[0][5]
                    bboxA = in_dict[idx]["channel" + str(chA)][bbA]["bbox"]
                    bboxB = in_dict[idx]["channel" + str(chB)][bbB]["bbox"]
                    confA = in_dict[idx]["channel" + str(chA)][bbA]["confidence"]
                    confB = in_dict[idx]["channel" + str(chB)][bbB]["confidence"]
                    in_dict[idx]["channel" + str(chB)][bbB]["tag"] = False
                    in_dict[idx]["channel" + str(chA)][bbA]["tag"] = False

                    output_confidence = 0
                    if confA > confB:
                        new_bb = [confA, bboxA]

                    else:
                        new_bb = [confB, bboxB]

                    if output_type == 1:
                        fo.write("{} {} {} {} {} {}\n".format(obj_type.strip().replace(" ", "_"), new_bb[0], new_bb[1][0],
                                                              new_bb[1][1], new_bb[1][2],
                                                              new_bb[1][3]))
                    elif output_type == 2:
                        fo.write("{},{},{},{},{},{},{},{}\n".format(0, str(idx).zfill(5), new_bb[1][0], new_bb[1][1],
                                                                    new_bb[1][2],
                                                                    new_bb[1][3], new_bb[0],
                                                                    obj_type.strip().replace(" ", "_")))

                for ch in channel_list:
                    for b_left in in_dict[idx]["channel" + str(ch)]:
                        if "tag" not in b_left or b_left["tag"] is True:
                            if output_type == 1:
                                fo.write("{} {} {} {} {} {}\n".format(b_left["type"].strip().replace(" ", "_"),
                                                                      b_left["confidence"], b_left["bbox"][0],
                                                                      b_left["bbox"][1], b_left["bbox"][2],
                                                                      b_left["bbox"][3]))
                            elif output_type == 2:

                                fo.write("{},{},{},{},{},{},{},{}\n".format(0, str(idx).zfill(5), b_left["bbox"][0],
                                                                            b_left["bbox"][0],
                                                                            b_left["bbox"][1], b_left["bbox"][2],
                                                                            b_left["bbox"][3], b_left["confidence"],
                                                                            b_left["type"].strip().replace(" ", "_")))

                if output_type == 1:
                    fo.close()

            if output_type == 2:
                fo.close()