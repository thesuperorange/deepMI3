import pandas as pd
import os
import itertools
import BoundingBoxIOU



channel_list = [2, 4, 6]


def read_txt_as_dict(in_file):
    df = pd.read_csv(in_file, header=None,
                     names=['channel', 'frame', 'start_x', 'start_y', 'end_x', 'end_y', 'confidence', 'type'])
    df.set_index('frame')['confidence'].to_dict()
    d = {}
    for i in df['frame'].unique():
        tmp = {"channel2": [], "channel4": [], "channel6": []}
        for jj in df[df['frame'] == i].index:

            if(df['channel'][jj] in channel_list):
                tmp["channel" + str(df['channel'][jj])].append(
                    {"bbox": [df['start_x'][jj], df['start_y'][jj], df['end_x'][jj], df['end_y'][jj]],
                     "confidence": df['confidence'][jj], "type": df['type'][jj]})
        d[i] = tmp  # sorted(tmp, reverse = True ,key=lambda x: x['confidence'])
    return d


if __name__ == '__main__':

    iter_channel = list(itertools.combinations(channel_list, 2))

    method_list = ['yolo']


    dataset_list = ['Pathway1_1','Pathway2_1','Pathway2_2','Pathway2_3','Bus_2',
                    'Staircase_1',#'Staircase_2','Staircase_3','Staircase_4',
                    'Doorway_1','Doorway_2',#'Doorway_3',
                    'Room_1' #m=,'Room_2'
                   ]
    #dataset_list = ['Pathway', 'Bus', 'Staircase', 'Doorway', 'Room']

    folder = '/home/superorange5/Research/YOLOv4/yolov4_opencv/'


    isFace = False
    isMDtrack =False

    folder_concat_str = ''
    if isFace:
        folder_concat_str+='_face'
        folder +='_face'
    if isMDtrack:
        folder_concat_str += '_MDtrack'
        folder = '~/Research/YOLOv4/yolov4_opencv/sort2/MDtrack_Results'


    IOU_th = 0.3
    select_type = 2  # 1 avg, 2 by confidence
    output_type =1  #1: output 1 folder/1 txt each image, 2: output 1 file

    for method in method_list:
        for dataset in dataset_list:

            if output_type==1:
                output_folder = os.path.join(folder,'merge', dataset +folder_concat_str+ '_merge')
                
                if not os.path.exists(output_folder):                    
                    os.makedirs(output_folder)

            inputFile = os.path.join(folder, method+'_'+dataset+folder_concat_str+'.txt')
            print("input file={}".format(inputFile))
            in_dict = read_txt_as_dict(inputFile)

            if output_type == 2:
                fo = open(os.path.join(folder, method + '_' + dataset +folder_concat_str+ '_merge.txt'), "w")

            #loop in frames
            for idx in in_dict:
                #print(idx)

                if output_type == 1:
                    #old version name
                    output_file_name = dataset + '_ch6_' + str(idx).zfill(5) + ".txt"
                    #output_file_name=str(idx)+'.txt'
                    if isFace:
                        output_file_name = dataset + '_face_' + str(idx).zfill(5) + ".txt"
                    print("output file={}".format(os.path.join(output_folder, output_file_name)))

                    fo = open(os.path.join(output_folder, output_file_name), "w")

                # print("###"+str(idx))


                iou_table_size = 0
                for ch_tuple in iter_channel:
                    iou_table_size += len(in_dict[idx]["channel" + str(ch_tuple[0])]) \
                                      * len(in_dict[idx]["channel" + str(ch_tuple[1])])
                # store iou between bb in each channel
                iou_table = []

                #only one channel contains bb
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
                                    #print(bbA["bbox"])
                                    #print(bbB["bbox"])
                                    iou = BoundingBoxIOU.get_iou(bbA["bbox"], bbB["bbox"])
                                    if iou > IOU_th:
                                        iou_table.append([iou, ch_tuple[0], A_idx, ch_tuple[1], B_idx, bbA["type"]])

                iou_table = sorted(iou_table, key=lambda x: x[0], reverse=True)

                while len(iou_table) != 0:
                    # print(iou_table)
                    iouAB = iou_table[0][0]
                    chA = iou_table[0][1]
                    bbA = iou_table[0][2]
                    chB = iou_table[0][3]
                    bbB = iou_table[0][4]
                    obj_type = iou_table[0][5]
                    bboxA = in_dict[idx]["channel" + str(chA)][bbA]["bbox"]
                    bboxB = in_dict[idx]["channel" + str(chB)][bbB]["bbox"]
                    confA = in_dict[idx]["channel" + str(chA)][bbA]["confidence"]
                    confB = in_dict[idx]["channel" + str(chB)][bbB]["confidence"]
                    in_dict[idx]["channel" + str(chB)][bbB]["tag"] = False
                    in_dict[idx]["channel" + str(chA)][bbA]["tag"] = False

                    new_group = [[confA, bboxA,chA], [confB, bboxB,chB]]

                    compare_list = []
                    for j in range(1, len(iou_table)):
                        if obj_type == iou_table[j][5]:
                            iouXY = iou_table[j][0]
                            chX = iou_table[j][1]
                            bbX = iou_table[j][2]
                            chY = iou_table[j][3]
                            bbY = iou_table[j][4]

                            # add if one of bb is compare target, another is not
                            if ((chX == chA and bbX == bbA) or (chX == chB and bbX == bbB)) and \
                                    (chY != chA and chY != chB):
                                compare_list.append([iouXY, chY, bbY])

                            elif ((chY == chA and bbY == bbA) or (chY == chB and bbY == bbB)) and \
                                    (chX != chA and chX != chB):
                                compare_list.append([iouXY, chX, bbX])

                                # print("compare_list = {}".format(compare_list))

                    # ABC
                    if compare_list:
                        compare_list = sorted(compare_list, key=lambda x: x[0], reverse=True)
                        chC = compare_list[0][1]
                        bbC = compare_list[0][2]
                        # find group ---avg of 3 bb

                        bboxC = in_dict[idx]["channel" + str(chC)][bbC]["bbox"]
                        in_dict[idx]["channel" + str(chC)][bbC]["tag"] = False
                        confC = in_dict[idx]["channel" + str(chC)][bbC]["confidence"]

                        new_group.append([confC, bboxC,chC])
                        # print("A.channel{} {}, B.channel{} {}, C.channel{} {}".format(chA,bbA,chB,bbB,chC,bbC))
                        if select_type == 1:
                            new_bb = [(confA + confB + confC) / 3,
                                      [(g + h + k) / 3 for g, h, k in zip(bboxA, bboxB, bboxC)]]
                        elif select_type == 2:
                            new_bb = sorted(new_group, key=lambda k: k[0], reverse=True)[0]

                        iou_table = [iou_left for iou_left in iou_table if not
                        ((iou_left[1] == chA and iou_left[2] == bbA) or (iou_left[3] == chA and iou_left[4] == bbA)
                        or (iou_left[1] == chB and iou_left[2] == bbB) or (iou_left[3] == chB and iou_left[4] == bbB)
                        or (iou_left[1] == chC and iou_left[2] == bbC) or (iou_left[3] == chC and iou_left[4] == bbC)
                        )]

                    # only AB
                    else:
                        if select_type == 1:
                            new_bb = [(confA + confB) / 2, [(g + h) / 2 for g, h in zip(bboxA, bboxB)]]
                        elif select_type == 2:
                            new_bb = sorted(new_group, key=lambda k: k[0], reverse=True)[0]

                        # print("A.channel{} {}, B.channel{} {}".format(chA,bbA,chB,bbB))

                        iou_table = [iou_left for iou_left in iou_table if not
                        ((iou_left[1] == chA and iou_left[2] == bbA) or (iou_left[3] == chA and iou_left[4] == bbA) or
                         (iou_left[1] == chB and iou_left[2] == bbB) or (iou_left[3] == chB and iou_left[4] == bbB)
                         )]

                    if output_type==1:
                        print("{} {} {} {} {} {}\n".format(obj_type.strip().replace(" ", "_"), new_bb[0], new_bb[1][0], new_bb[1][1], new_bb[1][2],new_bb[1][3]))
                        fo.write("{} {} {} {} {} {}\n".format(obj_type.strip().replace(" ", "_"), new_bb[0], new_bb[1][0], new_bb[1][1], new_bb[1][2],
                                                          new_bb[1][3]))
                    elif output_type == 2:
                        fo.write("{},{},{},{},{},{},{},{}\n".format(new_bb[2],str(idx).zfill(5) , new_bb[1][0],new_bb[1][1], new_bb[1][2],
                                                         new_bb[1][3],new_bb[0],obj_type.strip().replace(" ", "_")))
                    # print(iou_table)
                for ch in channel_list:

                    for b_left in in_dict[idx]["channel" + str(ch)]:
                        if "tag" not in b_left or b_left["tag"] is True:
                            if output_type==1:
                                fo.write("{} {} {} {} {} {}\n".format(b_left["type"].strip().replace(" ", "_"), b_left["confidence"], b_left["bbox"][0],
                                                              b_left["bbox"][1], b_left["bbox"][2], b_left["bbox"][3]))
                            elif output_type == 2:

                                fo.write("{},{},{},{},{},{},{},{}\n".format(ch, str(idx).zfill(5), b_left["bbox"][0],
                                                                            b_left["bbox"][1], b_left["bbox"][2],
                                                                            b_left["bbox"][3], b_left["confidence"],
                                                                            b_left["type"].strip().replace(" ", "_")))

                if output_type==1:
                    fo.close()

            if output_type == 2:
                fo.close()