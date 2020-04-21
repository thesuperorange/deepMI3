import os
import pascalvoc
if __name__ == '__main__':

    det_path = 'C:/Users/superorange/Google 雲端硬碟/NCHC_backup/SharedDrive/MI3_detect_results/'
    isTrack = False

    #method_list = [ 'SSD','yolo','faster-rcnn','mask-rcnn']
    method_list = ['faster-RCNN_coco']
    dataset_list_new = ['Pathway','Bus','Staircase','Doorway','Room','All']

    dataset_list = ['Pathway_all','Bus_2','Staircase_1','Doorway_all','Room_1','All']
    for idx,dataset in enumerate(dataset_list):


        GT_path = 'GT_ch6_'+dataset
        for method in method_list:


            gtFolder = "D:/Accuracy/"+GT_path
            detFolder = os.path.join(det_path,method,dataset_list_new[idx]+'_merge26')
            if isTrack:
                detFolder +='_MDtrack'
            gtformat = 'xyrb'
            detformat = 'xyrb'

            output_str = pascalvoc.evaluation(gtFolder,detFolder,0.5,gtformat,detformat,None,confidence_TH=0)
            print(dataset+','+method+','+output_str)
            #command="python pascalvoc.py -gt D:/Accuracy/"+GT_path+ " -det D:/Accuracy/"+det_path+" -gtformat xyrb -detformat xyrb -t 0.5 -np"
            #print(command)
            #os.system(command)
