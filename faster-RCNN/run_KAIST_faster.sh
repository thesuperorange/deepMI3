#!/bin/bash
FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
NET='vgg16'
model_sub_dir='fasterRD'

for round in $(seq 30 30)
do
    
        echo round=$round
        echo python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_1_${round}_1081.pth --cuda --load_dir models --image_dir $FILEPATH --output_folder KAIST_fasterRCNN_${NET}-AVG_${round}
        python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_KAIST_train_rd_${round}_1081.pth --cuda --load_dir models --model_sub_dir $model_sub_dir --image_dir $FILEPATH --output_folder ${model_sub_dir}/KAIST_fasterRCNN_${NET}_${round}
        
done

