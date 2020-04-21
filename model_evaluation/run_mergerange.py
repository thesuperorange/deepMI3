import os
import pascalvoc
if __name__ == '__main__':



    method_list = [ 'SSD','yolo','faster-rcnn','mask-rcnn']
    effective_range = [[(490, 544), (491, 530), (481, 530)],  # SSD
                       [(456, 545), (420, 544), (415, 519)],  # yolo
                       [(466, 544), (420, 531), (373, 524)],  # faster rcnn
                       [(469, 545), (422, 544), (391, 544)]]  # mask-rcnn


    # for eff in effective_range:
    #     print (min(eff)[0])
    #     print(max(eff)[1])  #530  strange
    #     res1 = list(map(max, zip(*eff)))
    #     res2 = list(map(min, zip(*eff)))
    #     print(res2[0])
    #     print(res1[1])

    dataset = 'Pathway1_1'
    confidence_TH_list = [0.5,0.8,0.9,0.98]

    for confidence_TH in confidence_TH_list:


        GT_path = 'GT_ch6_'+dataset
        for m_idx,method in enumerate(method_list):
            det_path = method+'_' + dataset+'_merge26'



            gtFolder = "D:/Accuracy/"+GT_path
            detFolder = "D:/Accuracy/"+det_path
            gtformat = 'xyrb'
            detformat = 'xyrb'

            res1 = list(map(max, zip(*effective_range[m_idx])))
            res2 = list(map(min, zip(*effective_range[m_idx])))

            range =(res2[0],res1[1])

            output_str = pascalvoc.evaluation(gtFolder,detFolder,0.5,gtformat,detformat,savePath=None,confidence_TH=confidence_TH,range=range)

            print(str(confidence_TH) + ',' + method + ',0,' + str(range[0]) + ',' + str(range[1]) + ',' + output_str)

            #command="python pascalvoc.py -gt D:/Accuracy/"+GT_path+ " -det D:/Accuracy/"+det_path+" -gtformat xyrb -detformat xyrb -t 0.5 -np"
            #print(command)
            #os.system(command)
