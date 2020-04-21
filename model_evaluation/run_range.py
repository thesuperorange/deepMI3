import os
import pascalvoc

if __name__ == '__main__':


    isTrack = False


    method_list = [ 'SSD','yolo','faster-rcnn','mask-rcnn']

    dataset = 'Pathway1_1'


    channel_list = [2,4,6]

    effective_range = [[(490, 544), (491, 530), (481, 530)] ,# SSD
                       [(456, 545), (420, 544), (415, 519)],# yolo
                       [(466,544),(420,531),(373,524)] , #faster rcnn
                        [(469, 545), (422, 544), (391, 544)] ]# mask-rcnn

    confidence_TH_list = [0.5,0.8,0.9,0.98]
    for j, method in enumerate(method_list):
        for i, channel in enumerate(channel_list):
            for confidence_TH in confidence_TH_list:
                GT_path = 'GT_ch' + str(channel) + '_' + dataset
                range = effective_range[j][i]
                det_path = method+'_ch' + str(channel) + '_' + dataset

                if isTrack:
                    det_path +='_MDtrack'

                gtFolder = "D:/Accuracy/" + GT_path
                detFolder = "D:/Accuracy/" + det_path
                gtformat = 'xyrb'
                detformat = 'xyrb'

                output_str = pascalvoc.evaluation(gtFolder, detFolder, 0.5, gtformat, detformat, savePath=None,confidence_TH=confidence_TH,range=range)
                print(str(confidence_TH)+','+method+','+str(channel)+','+ str(range[0])+','+ str(range[1])+','+ output_str)

            # command="python pascalvoc.py -gt D:/Accuracy/"+GT_path+ " -det D:/Accuracy/"+det_path+" -gtformat xyrb -detformat xyrb -t 0 -np"
            # print(command)
            # os.system(command)
