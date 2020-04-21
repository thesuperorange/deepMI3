import os
import pascalvoc
if __name__ == '__main__':


    isTrack = False


    method_list = ['faster-rcnn']
    dataset_list = ['Pathway1_1','Pathway2_3']

    for dataset in dataset_list:
        GT_path = 'GT_ch6_'+dataset
        for method in method_list:
            det_path = method+'_' + dataset+'_merge'

            if isTrack:
                det_path +='_MDtrack'

            gtFolder = "D:/Accuracy/"+GT_path
            detFolder = "D:/Accuracy/"+det_path
            gtformat = 'xyrb'
            detformat = 'xyrb'

            output_str = pascalvoc.evaluation(gtFolder,detFolder,0.5,gtformat,detformat,None,confidence_TH=0)
            print(dataset+','+method+','+output_str)
            #command="python pascalvoc.py -gt D:/Accuracy/"+GT_path+ " -det D:/Accuracy/"+det_path+" -gtformat xyrb -detformat xyrb -t 0.5 -np"
            #print(command)
            #os.system(command)
