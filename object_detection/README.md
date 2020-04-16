
# Object detection

## setting

* dataset labels: labels/<dataset_>.names

## pretrained model

| detector  | backbone | dataset | download |
| ------------- | ------------- | ------------- | ------------- |
| SSD | MobileNetv2 | COCO |  |
| YOLOv3 | Darknet  | COCO |  |
| mask R-CNN | Res-101 | COCO |  |


## mask R-CNN
* pretrained model: mask-rcnn-coco/
* detection

```
python mask-rcnn.py -d <dataset> -i <input_path> -l -v
```
> -d dataset-- Ex: pathway1_1, Room_1 <br>
> -i input path-- Ex: /home/xxxx/MI3 <br>
> -v is for visualization <br>
> -l log


* detect all datasets
```
python run_all.py
```


## SSD
* pretrained model: SSD_model/

```
python SSD.py -i <input_path> -d <dataset> -l -v
```
* detect all datasets
```
python run_all_SSD.py
```

## YOLO
* pretrained model: yolo-model/

```
python yolo.py -i <input_path> -d <dataset> -l -v
```
* detect all datasets
```
python run_all_yolo.py
```