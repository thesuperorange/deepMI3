#!/bin/bash
#DATASET=("Pathway2_1" "Pathway2_2" "Pathway2_3")
DATASET=("Room_1" "Doorway_1" "Doorway_2" "Bus_2" "Staircase_1")
METHOD_LIST=( "faster-rcnn" "mask-rcnn" "yolo" "SSD" )
FILEPATH='/work/superorange5/MI3/'
CHANNEL=( 2 4 6 )
for videoclip in "${DATASET[@]}"
do
    for ch in "${CHANNEL[@]}"
        do
            echo dataset=$videoclip, CHANNEL=$ch
            python demo.py --net vgg16 --checksession 1 --checkepoch 10 --checkpoint 625 --cuda --load_dir models --image_dir $FILEPATH --channel $ch --videoclip $videoclip --vis
    done
done

