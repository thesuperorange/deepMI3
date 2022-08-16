# faster RCNN pytorch

## setting
* python 3.6
* pytorch-1.0 (for pytorch-0.4 please see [original project](https://github.com/jwyang/faster-rcnn.pytorch))

### data preparation
* training data: data/<training_dataset>
* pretrained model: data/pretrained_model/<backbone_>_caffe.pth

### output folder
* detection results(BBs in txt): output/<output_name>/detection_results
* detection results(images): output/<output_name>/output_images
* output models: models/<backbone_>/<training_dataset>/

### pretrained model


1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[VGG-16](https://www.dropbox.com/s/6ief4w7qzka6083/faster_rcnn_1_6_10021.pth?dl=0)     | 1 | 1 | 1e-3 | 5   | 6   |  0.76 hr | 3265MB   | 70.1
[VGG-16](https://www.dropbox.com/s/cpj2nu35am0f9hp/faster_rcnn_1_9_2504.pth?dl=0)     | 1 | 4 | 4e-3 | 8   | 9  |  0.50 hr | 9083MB   | 69.6
[VGG-16](https://www.dropbox.com/s/1a31y7vicby0kvy/faster_rcnn_1_10_625.pth?dl=0)     | 8 | 16| 1e-2 | 8   | 10  |  0.19 hr | 5291MB   | 69.4
[VGG-16](https://www.dropbox.com/s/hkj7i6mbhw9tq4k/faster_rcnn_1_11_416.pth?dl=0)     | 8 | 24| 1e-2 | 10  | 11  |  0.16 hr | 11303MB  | 69.2
[Res-101](https://www.dropbox.com/s/4v3or0054kzl19q/faster_rcnn_1_7_10021.pth?dl=0)   | 1 | 1 | 1e-3 | 5   | 7   |  0.88 hr | 3200 MB  | 75.2
[Res-101](https://www.dropbox.com/s/8bhldrds3mf0yuj/faster_rcnn_1_10_2504.pth?dl=0)    | 1 | 4 | 4e-3 | 8   | 10  |  0.60 hr | 9700 MB  | 74.9
[Res-101](https://www.dropbox.com/s/5is50y01m1l9hbu/faster_rcnn_1_10_625.pth?dl=0)    | 8 | 16| 1e-2 | 8   | 10  |  0.23 hr | 8400 MB  | 75.2 
[Res-101](https://www.dropbox.com/s/cn8gneumg4gjo9i/faster_rcnn_1_12_416.pth?dl=0)    | 8 | 24| 1e-2 | 10  | 12  |  0.17 hr | 10327MB  | 75.1  


2). COCO (Train/Test: coco_train+coco_val-minival/minival, scale=800, max_size=1200, ROI Align)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
VGG-16     | 8 | 16    |1e-2| 4   | 6  |  4.9 hr | 7192 MB  | 29.2
[Res-101](https://www.dropbox.com/s/5if6l7mqsi4rfk9/faster_rcnn_1_6_14657.pth?dl=0)    | 8 | 16    |1e-2| 4   | 6  |  6.0 hr    |10956 MB  | 36.2
[Res-101](https://www.dropbox.com/s/be0isevd22eikqb/faster_rcnn_1_10_14657.pth?dl=0)    | 8 | 16    |1e-2| 4   | 10  |  6.0 hr    |10956 MB  | 37.0

**NOTE**. Since the above models use scale=800, you need add "--ls" at the end of test command.

3). COCO (Train/Test: coco_train+coco_val-minival/minival, scale=600, max_size=1000, ROI Align)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[Res-101](https://www.dropbox.com/s/y171ze1sdw1o2ph/faster_rcnn_1_6_9771.pth?dl=0)    | 8 | 24    |1e-2| 4   | 6  |  5.4 hr    |10659 MB  | 33.9
[Res-101](https://www.dropbox.com/s/dpq6qv0efspelr3/faster_rcnn_1_10_9771.pth?dl=0)    | 8 | 24    |1e-2| 4   | 10  |  5.4 hr    |10659 MB  | 34.5

4). MI3

__general hyperparameters setting__

* network: vgg16
* batch size: 24
* learning rate: 0.01
* lr_decay: 10
* max epoch: 20


      
experiment#     | train | test | AP | model
---------|--------|-----|--------|--------
E1 | channel 2 | channel 2 | 89.08 |  [M1](https://superorange.cos.twcc.ai/model/MI3_model/M1_channel2/faster_rcnn_1_20_52.pth)
E2 | channel 4 | channel 4 | 85.37 | [M2](https://superorange.cos.twcc.ai/model/MI3_model/M2_channel4/faster_rcnn_1_20_76.pth)
E3 | channel 6 | channel 2 | 92.71 | [M3](https://superorange.cos.twcc.ai/model/MI3_model/M3_channel6/faster_rcnn_1_20_76.pth)
E4 | all | all | 93.09 | [M4](https://superorange.cos.twcc.ai/model/MI3_model/M4_all/faster_rcnn_1_20_209.pth)
E5 | all | channel 2 | 90.63 |
E6 | all | channel 4 | 94.7 |
E7 | all | channel 6 | 95.05 | 
E8 | merge 246 to RGB | merge 246 to RGB | 89.08 | [M5](https://superorange.cos.twcc.ai/model/MI3_model/M5_RGB/faster_rcnn_1_20_76.pth)

experiment#     | train | test | AP | model
---------|--------|-----|--------|--------
S1 | pathway, doorway | bus, room, staircase | 55.9 |  [S1](https://superorange.cos.twcc.ai/model/MI3_model/S1_bydataset/faster_rcnn_1_20_331.pth)
S2 | pathway, bus, doorway, room | staircase | 86.14 | [S2](https://superorange.cos.twcc.ai/model/MI3_model/NoStaircase/faster_rcnn_1_20_467.pth)
S3 | pathway, bus, doorway, staircase | room | 98.64 | [S3](https://superorange.cos.twcc.ai/model/MI3_model/NoRoom/faster_rcnn_1_20_529.pth)
S4 | pathway, bus, room, staircase | doorway | 97.77 | [S4](https://superorange.cos.twcc.ai/model/MI3_model/NoDoorway/faster_rcnn_1_20_521.pth)
S5 | pathway, doorway, room, staircase | bus | 99.5 | [S5](https://superorange.cos.twcc.ai/model/MI3_model/NoBus/faster_rcnn_1_20_453.pth)
S6 | bus, doorway, room, staircase | pathway | 94.58 | [S6](https://superorange.cos.twcc.ai/model/MI3_model/NoPathway/faster_rcnn_1_20_265.pth)

## run
### train
```
python trainval_net.py --dataset MI3 --net vgg16 --bs 24 --nw 2 --lr 0.01 --lr_decay_step 10 --cuda --epochs 20
```

### test with MI3 model
```
python demoRGB.py  --net vgg16 --checksession 1 --checkepoch 20 --checkpoint 76 --cuda --load_dir models --image_dir /work/superorange5/MI3_dataset_ch6/Img_Sep/test_img/ --output_folder mi3_ch6 --vis
```

### test with pretrained model
```
python demo.py  --dataset coco --net res101 --checksession 1 --checkepoch 10 --checkpoint 14657 --cuda --load_dir models --image_dir <input image folder> --output_folder <outputname> --vis
```
>--vis whether save output image



## demo

### object
* Bus
<img src="img/merge_bus.jpg">

* Staircase
<img src="img/merge_staircase.jpg">

* Room
<img src="img/merge_room.jpg">

### face
* Pathway2_3
<img src="img/face_output_00419.png">
<br>
<img src="img/face_output_00483.png">

## Authorship
* original project: https://github.com/jwyang/faster-rcnn.pytorch
* This project is created by [Jianwei Yang](https://github.com/jwyang)  and [Jiasen Lu](https://github.com/jiasenlu), and modified by Peggy Lu

## Cite
```
@article{jjfaster2rcnn,
    Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
    Title = {A Faster Pytorch Implementation of Faster R-CNN},
    Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
    Year = {2017}
}

@inproceedings{renNIPS15fasterrcnn,
    Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
    Title = {Faster {R-CNN}: Towards Real-Time Object Detection
             with Region Proposal Networks},
    Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
    Year = {2015}
}
```
