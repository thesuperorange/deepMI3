#!/bin/bash
DATASET="Pathway1_1"
Method="faster-rcnn_face9"
FILEPATH='/home/superorange5/MI3/AP/'
MDnetPATH=${FILEPATH}'MDnet_results/withconf/'
CHANNEL=( 2 4 6 )
for ch in "${CHANNEL[@]}"
    do
    GROUNDTRUTH_PATH=$FILEPATH'groundtruth_'$DATASET'_face/ch'$ch
    DETECTION_PATH=$MDnetPATH$Method'_'$DATASET'/ch'$ch
    TH=( 0.5 0 )
    OUTPUT_PATH=$FILEPATH$Method'_'$DATASET'_AP_MDnet_ch'$ch
    for threshold in "${TH[@]}"
    do  
      echo METHOD=$Method, CHANNEL=$ch, TH=$threshold
      if [ "$threshold" = "0" ] 
        then
          echo python pascalvoc.py -gt $GROUNDTRUTH_PATH -det $DETECTION_PATH -t $threshold -gtformat xyrb -detformat xyrb
      else
          echo python pascalvoc.py -gt $GROUNDTRUTH_PATH -det $DETECTION_PATH -sp $OUTPUT_PATH -t $threshold -gtformat xyrb -detformat xyrb
      fi

    done
done

#python pascalvoc.py  -gt ~/MI3/AP/groundtruth_Pathway1_1/ch6 -det ~/MI3/AP/faster-rcnn_Pathway1_1/ch6  -t 0 -gtformat xyrb -detformat xyrb

