import os
import cv2
import argparse
import pickle

import shutil
def draw_image(input_path,output_folder,bbox_track_list,ext):

    tableau20 = [(31, 119, 180), (255, 127, 14),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    #
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder)

    for idx, track in enumerate(bbox_track_list):
        color = tableau20[idx]
        for bbox in track:
            file_num = bbox[0]
            if ext =='bmp':
                filename = os.path.join(input_path, str(file_num) + ".bmp")
            else:
                filename = os.path.join(input_path, str(file_num).zfill(5)+"."+ext)


            image = cv2.imread(filename)

            startX = int(bbox[1][0])
            startY = int(bbox[1][1])
            endX = int(bbox[1][0] + bbox[1][2])
            endY = int(bbox[1][1] + bbox[1][3])

            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            text = "track {}".format(idx+1)
            cv2.putText(image, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            output_name = output_folder + "/track" + str(idx+1) + "_" + str(file_num) + ".jpg"
#            print(output_name)
            cv2.imwrite(output_name, image)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--channel', default=6, help='which channel')
    parser.add_argument('-m', '--method',  default='faster-rcnn2', help='algorithm')
    parser.add_argument('-d', '--dataset',  default='Pathway1_1', help='dataset')
    parser.add_argument('-e', '--ext', default = 'png',help='bmp,png,jpg')

    args = parser.parse_args()

    dataset = args.dataset
    method = args.method
    channel = int(args.channel)
    ext = args.ext
    input_path = '/home/waue0920/'+dataset+'/ORIG/ch'+str(channel)
    output_folder = method + "_" + dataset + "_ch" + str(channel) + "_results"
    output_pickle_name = 'track_list_'+method+'_'+dataset+'_ch'+str(channel)+'.pkl'
    file2 = open(output_pickle_name, 'rb')
    bbox_track_list_pickle = pickle.load(file2)
    print(bbox_track_list_pickle)
    file2.close()
    draw_image(input_path, output_folder, bbox_track_list_pickle,ext)

