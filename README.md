# YOLO-Review
Review YOLO v1 ~ v3

##yolo v1:

split image to 7*7 subimage , and every subimage just look one time 

training:  first using 20 convolutional layers and average_pooling and 1*1000 fully-connected layer with 224*224 input to train classify model in ImageNet

second, remove the fully-connected layer and add four convolutional layers and two fully-connected layers and with 448*448 input to train detection model

#No-maximal Suppression(NMS)
for all classes do:
1 discard all boxes with confidence C < C_threshold
2 Sort the predictions starting form the highest confindence C.
3  Choose the box with the highest C and output it as prediction
4 Discard any box with IOU> IOU-threshold with the box in the previous step
5 Start again from step 3 until all remaining predictions are checked


##Fast YOLO and YOLO VGG_16
Fast YOLO is the fast version of YOLO with only 9 convolutional layers instead of 24 .


##YOLO v2
compare with YOLO v1:
1 BatchNormalization
2 High Resolution Classifier
train for classify model based on 224*224 input , then fine-tune the model on resolution 448*448 for 10 epochs  before training for detection , addition fine tune training increase 4% mAP

3 Convolutional with Anchor Boxes(multi-object prediction per grid)
k bounding boxes
difference with FasterRCNN predict bounding boxes using handing picked anchor boxes

instead of using hand pick anchor boxes in faster rcnn,  yolov2 use k-mean clustering to dervie k classes anchor boxes based on the IOU distribution. The finally result is k=5 that compromise between model complexity and high recall

YOLOv2 predict location coordinates relative to the location of the grid cell.
YOLOv2 link k anchor boxes to every grid cell , and bounding box attach to every grid cell and anchor boxes, so every grid cell can has k bounding box

#Network Architecture:
Darknet-19 :  author proposed a new classifier model ---Darknet-19 which contain 19 convoloutional layers and 5 max-pooling layers 
output shape will be : 13×13* k(1+4+20)  
k represent number of  anchor boxes, 20 is the number of classes

#Multi-Scale Training
after every 10 batches network randomly chooses a new images size from {320, 352,384....., 608} , resize the network to the new dimension and continue training. This technology can give the model ability to  predict in different size(resolution) images.

##YOLO 9000
this model can classify more 9k objects and optimizing on detection and classification.

In YOLOv2,  author proposed a joint trianing pipeline which  mix images from classify and detection datasets,  if input images labeled detection ,it will backpropagate based on the full YOLOv2 loss function , otherwise input images labeled classify, it will propagate only based on classification specific parts of the architecture. COCO dataset for detection , and ImageNet dataset for classification.

ImageNet label is usually more specially for object 


##YOLO v3
like YOLO9000 just output 4 coordinates for each bounding box
also give a prediction score for each bounding box using logisit regression, if one anchor box has highest confidence  score in all anchor boxes, then it will be the result box, if one anchor not assign to a ground truth object  it will discard; if one anchor box dose not have the highest IOU but dose has highest confidence score , ignore this prediction.

#Multi-labels prediction:
one object has multi-labels  in some open imageset
YOLO v3 using independent logistic classifiers for every class instead of using softmax as output layer . During training, using binary cross-entropy loss for class predicition.

#Small objects detection:
YOLO struggles with small size object detection, however, yolov3 explore short cut connections to keep featrues from  earlier feature map , it will improve the performance of small object detection.  But YOLO v3 has worse performance on medium and larger size object

#Feature Extractor Network(Darknet-53):
it is a hybird network combine Darknet-19 and Resnet, so it contain short cut connection.
