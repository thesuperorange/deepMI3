

import os
import pascalvoc
if __name__ == '__main__':


    isTrack = False

    method_list = [ 'SSD','yolo','faster-rcnn','mask-rcnn']
    #method_list = ['yolo']
    #dataset_list = ['Pathway1_1']
    dataset_list = ['Pathway1_1','Pathway_all','Bus_2','Staircase_1','Doorway_all','Room_1','All']
    for dataset in dataset_list:


        GT_path = 'GT_ch6_'+dataset

        output_format_list = [0 for x in range(len(method_list)*2)]
        for m_idx,method in enumerate(method_list):
            det_path = method+'_' + dataset+'_merge26'

            if isTrack:
                det_path +='_MDtrack'

            gtFolder = "D:/Accuracy/"+GT_path
            detFolder = "D:/Accuracy/"+det_path
            gtformat = 'xyrb'
            detformat = 'xyrb'

            output_str = pascalvoc.evaluation(gtFolder,detFolder,0.5,gtformat,detformat,None,confidence_TH=0)
            output_split = output_str.split(',')
            AP = output_split[0]
            Fscore=output_split[1]

            output_format_list[m_idx] = AP
            output_format_list[m_idx +len(method_list)] = Fscore

            #print(dataset+','+method+','+output_str)
            #command="python pascalvoc.py -gt D:/Accuracy/"+GT_path+ " -det D:/Accuracy/"+det_path+" -gtformat xyrb -detformat xyrb -t 0.5 -np"
            #print(command)
            #os.system(command)
        outputString = ",".join(output_format_list)
        print(outputString)
