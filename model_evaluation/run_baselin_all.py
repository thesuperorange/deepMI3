import os
import pascalvoc
import shutil
if __name__ == '__main__':

    scene_list = ['Room','Doorway','Bus','Pathway'] #'Staircase',

    DATASET='MI3'
    ROUND_NUM = 10
    NET='vgg16'
    #isAll = True
    for isAll in ( False,True):
        
#    dataset_list = ['campus','road','downtown']
#    for dataset in dataset_list:
        scene_list = ['Room','Doorway','Bus','Pathway','Staircase']
        for i,scene in enumerate(scene_list):
            sub_folder='baseline_no'+scene
           

    #        gtFolder = "/home/superorange5/data/KAIST/test_annotations/visible/campus"        
            detFolder = "/home/superorange5/Research/FedWCD/output/"+sub_folder+'/'
            if isAll:
                detFolder = detFolder+"detection_results/"
                gtFolder = "/home/superorange5/MI3_dataset/MI3_dataset/Ann_txt/"

            else:
                input_dir = detFolder+"detection_results/"
                gtFolder = "/home/superorange5/MI3_dataset/MI3_dataset_bydataset/"+scene+"_txt/"
                detFolder = detFolder+scene+"/"
                #------ copy target data to another folder--------------
                
                for file in os.listdir(input_dir):    
                    if scene in file:
                        if not os.path.exists(detFolder):
                            os.makedirs(detFolder)
                        #print("copy {} to {}".format(os.path.join(input_dir, file),os.path.join(output_dir, file)))
                        shutil.copy2(os.path.join(input_dir, file), os.path.join(detFolder, file))
                 #---------------------------#

            gtformat = 'xyrb'
            detformat = 'xyrb'

            confidence_TH = 0
            iou_TH = 0.5


            output_str = pascalvoc.evaluation(gtFolder, detFolder, iou_TH, gtformat, detformat, None, confidence_TH=confidence_TH, range=None)
            print(scene+','+str(isAll)+',' + output_str)

