# TensorFlOW
A collection of trained machine learning models to detect HP bars in the game Overwatch. Images were all collected from the game myself.

Current trained models:
- faster_rcnn_inception_v2
- ssd_mobilenet_v1


## Getting Started
#### Dependencies
- tensorflow-gpu
- cv2
- mss
- numpy

#### Folder Structure
```
+ detection examples: contains example images of detected health bars using the given models
+ images: contains train/test images used to train the models
+ models: contains files of the different models (also has the trained inference graphs)
+ utils: scripts used to create the tfrecords and labels (Credit: https://github.com/datitran/raccoon_dataset)
- file_detection.py: Shows and saves detections from a folder of images
- live_detection.py: Shows detections from screenshots of the screen
```


## Models
All models were trained using the TensorFlow object detection API. FPS was measured on a system with an i7-7700k and a GTX 1080.
### faster_rcnn_inception_v2
Avg. detection FPS: 12

Example 1: 

![Example 1](https://i.imgur.com/vfxXsN1.png)

Example 2:

![Example 2](https://i.imgur.com/HVIOxMZ.png)

Example 3:

![Example 3](https://i.imgur.com/0rJxSTv.png)

### ssd_mobilenet_v1
Avg. detection FPS: 30

Example 1: 

![Example 1](https://i.imgur.com/QnFTHfe.png)

Example 2:

![Example 2](https://i.imgur.com/5UREQHe.png)

Example 3:

![Example 3](https://i.imgur.com/zxYUG8E.png)
