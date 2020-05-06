
# Object detection

## setting

* dataset labels: labels/<dataset_>.names

## pretrained model

| detector  | backbone | dataset | download |
| ------------- | ------------- | ------------- | ------------- |
| SSD | MobileNetv2 | COCO | [SSD_model](https://drive.google.com/open?id=16GR0_LnOKJGJSz8VBUtxlaXbIApvd9IZ) |
| YOLOv3 | Darknet  | COCO | [YOLO model](https://drive.google.com/open?id=1LlLr-cwZaEt4Fhs5ZgF0XmLMr-osQ2vb) |
| mask R-CNN | Res-101 | COCO | [Mask R-CNN model](https://drive.google.com/open?id=1tVzaQQp8PMQlTf0LTP1xglMnKtFD9nf4)  |

##usage

```
python object_detection.py -d <detector> -i <input_path> -v -o <output folder>

```
> -d detector  Ex: pathway1_1, Room_1 <br>
> -i input path  Ex: /work/xxxx/MI3_dataset/JPEGImages/ <br>
> -v is for visualization <br>

## detect by dataset (old version)

### mask R-CNN
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


### SSD
* pretrained model: SSD_model/

```
python SSD.py -i <input_path> -d <dataset> -l -v
```
* detect all datasets
```
python run_all_SSD.py
```

### YOLO
* pretrained model: yolo-model/

```
python yolo.py -i <input_path> -d <dataset> -l -v
```
* detect all datasets
```
python run_all_yolo.py
```
