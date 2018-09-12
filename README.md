# Road_Damage_Detection
## Introduction
The following YOLO(You Only Look Once) algorithm uses Keras implementation.
For futher information refer to: http://guanghan.info/blog/en/my-works/train-yolo/

Dataset is taken from : https://github.com/sekilab/RoadDamageDetector.
Annotated images are presented in the same format as PASCAL VOC.

## Libraries used:
* opencv-python
* tensorflow
* keras
* pillow
* numpy

## Usage
To train the model use following command:
  ```
  python src/yolo.py train [Pretrained_Model.h5]
  ```
If saved Keras model option is used, it will read the pretrained model and do training incrmentally.
If the pretrained model option is not used training is done from scratch.

For prediction use:
  ```
  python src/yolo.py test Pretrained_Model.h5 testlist.txt
  ```

## Information about code:
Code explanation:
* workingcfg.txt : The path to cfg file, training list and voc.names is put in this file
* yolo.py : This is used for training and testing the data
* yolodata.py : Read train_data/train.txt file, then generate resized X_train and Y_train numpy matrix
* ddd.py : Make custom YOLO loss function. For further information about YOLO loss eqaution refer to:
           http://pjreddie.com/media/files/papers/yolo_1.pdf
* kerasmodel.py : Create Keras model according to cfg file
* parse.py : This is used to parse the cfg file
* cfgconst.py : The parse.py parses the cfg file after this program reads the workingcfg.txt file.

### Running environment : 
    The above code uses Keras with tensorflow backend
