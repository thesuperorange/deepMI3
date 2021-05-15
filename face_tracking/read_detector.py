import csv

from modules.bb_object import BBox



def readFrameBBoxList(frame_num, filename, channel, fac_det,TH):

    frameBBoxList = [[] for y in range(frame_num+1)]

    # bbox_count = [[0 for x in range(image_num)] for y in range(channel_num)]
    # ppl_count = [[0 for x in range(image_num)] for y in range(channel_num)]
    # confidence_mat = [[0 for x in range(image_num)] for y in range(channel_num)]
    # area_mat = [[0 for x in range(image_num)] for y in range(channel_num)]
    #


    with open(filename, 'rt') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:

            channel_idx = int(row[0])
            confidence = float(row[6])
            if fac_det==False and row[7] !='person':
                continue

            if (channel_idx == channel and confidence >TH):
                img_idx = int(row[1])
                #bbox_count[channel_idx][img_idx] = bbox_count[channel_idx][img_idx] + 1

                startX = int(float(row[2]))
                startY = int(float(row[3]))
                endX = int(float(row[4]))
                endY = int(float(row[5]))
                # already detected one face

                frameBBoxList[img_idx].append(BBox(startX, startY, endX, endY, confidence))

    csvfile.close()
    return frameBBoxList

if __name__ == '__main__':
    frame_num = 548
    channel_num = 7
    channel_list = [2, 4, 6]
    method = 'faster-rcnn2'  # results/yolo, results_new/faster-rcnn2, results_new/mask-rcnn
    dataset = 'Pathway1_1'
    start_frame = 310
    image_num = 550
    filename = method + "_" + dataset + ".txt"

    frameBBoxList = readFrameBBoxList(frame_num,filename,4,False)

    for i in range(0,frame_num):
        print(i)
        if frameBBoxList[i]:
            print(frameBBoxList[i][0].startX)
            print(frameBBoxList[i][0].endX)
            print(frameBBoxList[i][0].startY)
            print(frameBBoxList[i][0].confidence)
            print(frameBBoxList[i][0].endY)

            print(frameBBoxList[i][0].getWidth())
            print(frameBBoxList[i][0].getHeight())
            print(frameBBoxList[i][0].getArea())

