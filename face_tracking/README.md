# face tracking

## overview

* flowchart
<img src=img/tracking.png>

## run

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