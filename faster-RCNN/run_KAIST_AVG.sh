#!/bin/bash
FILEPATH='/home/superorange5/data/kaist_test/kaist_test_visible/'
NET='vgg16'
model_sub_dir='faster_e30'

for round in $(seq 1 10)
do
    
        echo round=$round
        echo python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_KAIST_AVG_${round}.pth --cuda --load_dir models --image_dir $FILEPATH --output_folder KAIST_fasterRCNN_${NET}-AVG_${round}
        python demoKAIST2.py  --net ${NET} --model_name faster_rcnn_KAIST_AVG_${round}_n.pth --cuda --load_dir models --model_sub_dir $model_sub_dir --image_dir $FILEPATH --output_folder ${model_sub_dir}/KAIST_fasterRCNN_${NET}-AVG_${round}_n
        
done

