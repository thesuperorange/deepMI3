import os
import pascalvoc
if __name__ == '__main__':


    isTrack = False

    method_list = [ 'SSD','yolo','faster-rcnn','mask-rcnn']
    dataset_list = ['Pathway1_1','Pathway_all','Bus_2','Staircase_1','Doorway_all','Room_1','All']


    #dataset_list = ['Pathway1_1']   #face: Pathway2_3_face   #track :Pathway1_1_face_MDtrack
    channel_list = [2,4,6]

    for dataset in dataset_list:
        for channel in channel_list:



            GT_path = 'GT_ch'+str(channel)+'_'+dataset
            for method in method_list:
                det_path = method+'_ch' + str(channel) + '_' + dataset

                if isTrack:
                    det_path +='_MDtrack'

                gtFolder = "D:/Accuracy/" + GT_path
                detFolder = "D:/Accuracy/" + det_path
                gtformat = 'xyrb'
                detformat = 'xyrb'

                output_str = pascalvoc.evaluation(gtFolder, detFolder, 0.5, gtformat, detformat, None,confidence_TH=0)
                print(dataset + ',' +str(channel)+','+ method + ',' + output_str)

                # command="python pascalvoc.py -gt D:/Accuracy/"+GT_path+ " -det D:/Accuracy/"+det_path+" -gtformat xyrb -detformat xyrb -t 0 -np"
                # print(command)
                # os.system(command)
