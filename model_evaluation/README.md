# Object Metrics

## Run

```
python ~/Object-Detection-Metrics/pascalvoc.py -t 0.5 -f xyrb -g /<groundtruths path> -d <detection results path>
```

> -t threshold <br> t=0.5 AP50, t=0.95 AP95, t=0 mAP(avg of 50~95)<br>
> -f xyrb(right/bottom) or xywh (width/height)


## reference

* The Relationship Between Precision-Recall and ROC Curves (Jesse Davis and Mark Goadrich) Department of Computer Sciences and Department of Biostatistics and Medical Informatics, University of Wisconsin
http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

* The PASCAL Visual Object Classes (VOC) Challenge
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf

* Evaluation of ranked retrieval results (Salton and Mcgill 1986)
https://www.amazon.com/Introduction-Information-Retrieval-COMPUTER-SCIENCE/dp/0070544840
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html

* Rafael Padilla, Sergio Lima Netto and Eduardo A. B. da Silva, 'Survey on Performance Metrics for Object-Detection Algorithms', International Conference on Systems, Signals and Image, 2020
