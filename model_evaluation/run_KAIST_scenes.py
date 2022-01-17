import os
import pascalvoc
if __name__ == '__main__':


    NET='vgg16'   
    sub_folder = 'FedPer'
    dataset_list = ['campus','road','downtown']
    for dataset in dataset_list:
        for round_num in range(10):
            for epoch in range(3):

                gtFolder = "/home/superorange5/data/KAIST/test_annotations/visible/ALL/"
                detFolder = "/home/superorange5/Research/2019_deepMI3/deepMI3/faster-RCNN/output/"+sub_folder+"/KAIST_fasterRCNN_"+NET+"-"+dataset+"_"+str(round_num+1)+"_"+str(epoch+1)+"/detection_results/"
                gtformat = 'xyrb'
                detformat = 'xyrb'

                confidence_TH = 0
                iou_TH = 0.5


                output_str = pascalvoc.evaluation(gtFolder, detFolder, iou_TH, gtformat, detformat, None, confidence_TH=confidence_TH, range=None)
                print(dataset + ',' +str(round_num+1)+','+ str(epoch+1) + ',' + output_str)

