###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys
import numpy as np
import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,

                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0),
                     confidence_TH = 0,
                     range=None):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        #print(f)
        nameOfImage = f.replace(".txt", "")

        filename_split = nameOfImage.split('_')
        if range is None or (int(filename_split[3]) >= range[0] and int(filename_split[3]) <= range[1]):
        #print(nameOfImage)
            fh1 = open(f, "r")
            for line in fh1:
                line = line.replace("\n", "")
                if line.replace(' ', '') == '':
                    continue
                splitLine = line.split(" ")

                if isGT:
                    # idClass = int(splitLine[0]) #class
                    idClass = (splitLine[0])  # class
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])
                    bb = BoundingBox(
                        nameOfImage,
                        idClass,
                        x,
                        y,
                        w,
                        h,
                        coordType,
                        imgSize,
                        BBType.GroundTruth,
                        format=bbFormat)
                else:
                    # idClass = int(splitLine[0]) #class
                    idClass = (splitLine[0])  # class
                    confidence = float(splitLine[1])
                    if confidence < confidence_TH:
                        continue

                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])
                    bb = BoundingBox(
                        nameOfImage,
                        idClass,
                        x,
                        y,
                        w,
                        h,
                        coordType,
                        imgSize,
                        BBType.Detected,
                        confidence,
                        format=bbFormat)


                allBoundingBoxes.addBoundingBox(bb)
                if idClass not in allClasses:
                    allClasses.append(idClass)
            fh1.close()
    return allBoundingBoxes, allClasses

def evaluation(gtFolder,detFolder,iouThreshold,gtFormat,detFormat,savePath,confidence_TH,range=None):

# Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))
    #confidence_TH = 0.8
    VERSION = '0.1 (beta)'
    gtCoordType='abs'
    detCoordType='abs'
    showPlot=False
# parser = argparse.ArgumentParser(
#     prog='Object Detection Metrics - Pascal VOC',
#     description='This project applies the most popular metrics used to evaluate object detection '
#     'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
#     'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
#     epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# # formatter_class=RawTextHelpFormatter)
# parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# # Positional arguments
# # Mandatory
# parser.add_argument(
#     '-gt',
#     '--gtfolder',
#     dest='gtFolder',
#     default=os.path.join(currentPath, 'groundtruths'),
#     metavar='',
#     help='folder containing your ground truth bounding boxes')
# parser.add_argument(
#     '-det',
#     '--detfolder',
#     dest='detFolder',
#     default=os.path.join(currentPath, 'detections'),
#     metavar='',
#     help='folder containing your detected bounding boxes')
# # Optional
# parser.add_argument(
#     '-t',
#     '--threshold',
#     dest='iouThreshold',
#     type=float,
#     default=0.5,
#     metavar='',
#     help='IOU threshold. Default 0.5')
# parser.add_argument(
#     '-gtformat',
#     dest='gtFormat',
#     metavar='',
#     default='xywh',
#     help='format of the coordinates of the ground truth bounding boxes: '
#     '(\'xywh\': <left> <top> <width> <height>)'
#     ' or (\'xyrb\': <left> <top> <right> <bottom>)')
# parser.add_argument(
#     '-detformat',
#     dest='detFormat',
#     metavar='',
#     default='xywh',
#     help='format of the coordinates of the detected bounding boxes '
#     '(\'xywh\': <left> <top> <width> <height>) '
#     'or (\'xyrb\': <left> <top> <right> <bottom>)')
# parser.add_argument(
#     '-gtcoords',
#     dest='gtCoordinates',
#     default='abs',
#     metavar='',
#     help='reference of the ground truth bounding box coordinates: absolute '
#     'values (\'abs\') or relative to its image size (\'rel\')')
# parser.add_argument(
#     '-detcoords',
#     default='abs',
#     dest='detCoordinates',
#     metavar='',
#     help='reference of the ground truth bounding box coordinates: '
#     'absolute values (\'abs\') or relative to its image size (\'rel\')')
# parser.add_argument(
#     '-imgsize',
#     dest='imgSize',
#     metavar='',
#     help='image size. Required if -gtcoords or -detcoords are \'rel\'')
# parser.add_argument(
#     '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
# parser.add_argument(
#     '-np',
#     '--noplot',
#     dest='showPlot',
#     action='store_false',
#     help='no plot is shown during execution')
# args = parser.parse_args()
#
# iouThreshold = args.iouThreshold

# Arguments validation
    errors = []
# # Validate formats
    gtFormat = ValidateFormats(gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(detFormat, '-detformat', errors)
# # Groundtruth folder
# if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
#     gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
# else:
#     # errors.pop()
#     gtFolder = os.path.join(currentPath, 'groundtruths')
#     if os.path.isdir(gtFolder) is False:
#         errors.append('folder %s not found' % gtFolder)
# # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(gtCoordType, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(gtCoordType, '-detCoordinates', errors)
    imgSize = (0, 0)
# if gtCoordType == CoordinatesType.Relative:  # Image size is required
#     imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
# if detCoordType == CoordinatesType.Relative:  # Image size is required
#     imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
# # Detection folder
# if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
#     detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
# else:
#     # errors.pop()
#     detFolder = os.path.join(currentPath, 'detections')
#     if os.path.isdir(detFolder) is False:
#         errors.append('folder %s not found' % detFolder)
# print(args.savePath)


    if savePath is not None:
        print(savePath)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        #savePath = ValidatePaths(savePath, '-sp/--savepath', errors)
    else:
        savePath = os.path.join(currentPath, 'results')

# Validate savePath
# If error, show error messages
# if len(errors) is not 0:
#     print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
#                                 [-detformat] [-save]""")
#     print('Object Detection Metrics: error(s): ')
# #    print(e) for e in errors
#     for e in errors:
#         print(e)
#     sys.exit()

    # Create directory to save results
    shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
    os.makedirs(savePath)
    # Show plot during execution
    #showPlot = args.showPlot

# print('iouThreshold= %f' % iouThreshold)
# print('savePath = %s' % savePath)
# print('gtFormat = %s' % gtFormat)
# print('detFormat = %s' % detFormat)
# print('gtFolder = %s' % gtFolder)
# print('detFolder = %s' % detFolder)
# print('gtCoordType = %s' % gtCoordType)
# print('detCoordType = %s' % detCoordType)
# print('showPlot %s' % showPlot)

# Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
       gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize,range=range)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
      detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize,confidence_TH=confidence_TH,range=range)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0
    output_str=''
    if iouThreshold==0:
        AP_sum = 0
        for iouTH in np.arange(0.5, 0.96, 0.05):
            detections = evaluator.PlotPrecisionRecallCurve(
                allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
                IOUThreshold=iouTH,  # IOU threshold
                method=MethodAveragePrecision.EveryPointInterpolation,
                showAP=True,  # Show Average Precision in the title of the plot
                showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
                savePath=savePath,
                showGraphic=showPlot)
            for metricsPerClass in detections:

                # Get metric values per each class
                cl = metricsPerClass['class']
                ap = metricsPerClass['AP']
                precision = metricsPerClass['precision']
                recall = metricsPerClass['recall']
                totalPositives = metricsPerClass['total positives']
                total_TP = metricsPerClass['total TP']
                total_FP = metricsPerClass['total FP']

                if totalPositives > 0:
                    validClasses = validClasses + 1
                    acc_AP = acc_AP + ap
                    prec = ['%.2f' % p for p in precision]
                    rec = ['%.2f' % r for r in recall]
                    ap_str = "{0:.2f}%".format(ap * 100)
                    # ap_str = "{0:.4f}%".format(ap * 100)
                    #print('AP: %s (%s)' % (ap_str, cl))


            mAP = acc_AP / validClasses
            print('mAP: {0:.2f}%'.format(mAP* 100))
            AP_sum +=mAP
        mAP_str = "{0:.2f}%".format(AP_sum * 10)
        print('Total mAP: %s' % mAP_str)

    else:
        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iouThreshold,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
            savePath=savePath,
            showGraphic=showPlot)

        f = open(os.path.join(savePath, 'results.txt'), 'w')
        f.write('Object Detection Metrics\n')
        f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
        f.write('Average Precision (AP), Precision and Recall per class:')

        # each detection is a class
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                prec = ['%.2f' % p for p in precision]

                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}".format(ap * 100)
                # ap_str = "{0:.4f}%".format(ap * 100)
                prec_1=0
                rec_1=0
                if(len(precision)>0):
                    prec_1 =precision[-1]

                if(len(recall)>0):
                    rec_1 = recall[-1]

                fscore=0
                if(prec_1!=0 and rec_1 !=0):
                    fscore = (2 * prec_1 * rec_1) / (prec_1 + rec_1)
                rec_str = '{0:.2f}'.format(rec_1 * 100)
                prec_str = '{0:.2f}'.format(prec_1 * 100)
                fscore_str = '{0:.2f}'.format(fscore * 100)

                #print('precision {0:.2f}%' .format( precision[-1]*100))
                #print('recall {0:.2f}%'  .format(recall[-1]*100))
                #print('fscore={0:.2f}%'.format(fscore * 100))
                #print('AP: %s (%s)' % (ap_str, cl))

                #output_str = ap_str+','+prec_str+','+rec_str+','+fscore_str
                output_str = ap_str +  ',' + fscore_str


                f.write('\n\nClass: %s' % cl)
                f.write('\nAP: %s' % ap_str)
                f.write('\nPrecision: %s' % prec)
                f.write('\nRecall: %s' % rec)

        mAP = acc_AP / validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        #print('mAP: %s' % mAP_str)
        f.write('\n\n\nmAP: %s' % mAP_str)

    return output_str





