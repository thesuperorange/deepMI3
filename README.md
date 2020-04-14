# deepMI3

## Data preparation
* original data
https://sites.google.com/site/miirsurveillance/
* dataset & groundtruths
https://scidm.nchc.org.tw/dataset/mi3


## Object detection

* pretrained model

| detector  | backbone | dataset |
| ------------- | ------------- | ------------- |
| SSD | MobileNetv2 | COCO |
| YOLOv3 | Darknet  | COCO |
| faster R-CNN | Res-101 | COCO |
| mask R-CNN | Res-101 | COCO |

## face tracking

* Pathway1 
```
python track_detect.py -s --channel 2 -d Pathway1_1 --method faster-rcnn2  --ext bmp

```
* Pathway1 face
```
python track_detect.py -s --channel 2 -d Pathway1_1 --method faster-rcnn_face9 -i

```
* draw by pkl

```
python draw_result.py --channel 2 --method faster-rcnn2 --dataset Pathway1_1 -s

```

* run all
```
run_MDnet.sh
```


## fusion

## Authorship
* Author: Peggy Lu
## Citation

* C.-H. Chan, H.-T. Chen, W.-C. Teng, C.-W. Liu, J.-H. Chuang, "MI3: Multi-Intensity Infrared Illumination Video Database," IEEE Int'l Conf. on Visual Communications and Image Processing (VCIP), Singapore, Singapore, Dec. 13-16, 2015.
