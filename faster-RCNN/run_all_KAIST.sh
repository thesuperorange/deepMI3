#!/bin/bash
#DATASET=("Pathway2_1" "Pathway2_2" "Pathway2_3")
DATASET=("campus" "road" "downtown" )
CHECKPOINTs=(718 513 566)
FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
NET='vgg16'
model_sub_dir='FedPer'
for round in $(seq 1 10)
do
    for i in "${!DATASET[@]}"
    do
        for epoch in $(seq 1 3)
        do
            cp=${CHECKPOINTs[$i]}
            scene=${DATASET[$i]}
            echo round=$round, dataset=$scene, epoch=$epoch, checkpoint=$cp
            echo python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_KAIST_${scene}_${round}_${epoch}_${cp}.pth --cuda --load_dir models --model_sub_dir ${model_sub_dir} --image_dir $FILEPATH --output_folder ${model_sub_dir}/KAIST_fasterRCNN_${NET}-${scene}_${round}_${epoch}
            python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_KAIST_${scene}_${round}_${epoch}_${cp}.pth --cuda --load_dir models --model_sub_dir ${model_sub_dir} --image_dir $FILEPATH --output_folder ${model_sub_dir}/KAIST_fasterRCNN_${NET}-${scene}_${round}_${epoch} 
        done
    done
done

