import pascalvoc

if __name__ == '__main__':


    dataset='Pathway1_1'
    method = 'faster-rcnn'
    det_path = method + '_' + dataset + '_face_merge'#_MDtrack_merge
    GT_path = 'GT_ch6_' + dataset+'_face'
    gtFolder = "D:/Accuracy/" + GT_path
    detFolder = "D:/Accuracy/" + det_path
    gtformat = 'xyrb'
    detformat = 'xyrb'
    confidence_TH = 0
    sp = 'D:/tempAP'


    output_str = pascalvoc.evaluation(gtFolder, detFolder, 0.5, gtformat, detformat, savePath=sp, confidence_TH=confidence_TH, range=None)
    print(output_str)