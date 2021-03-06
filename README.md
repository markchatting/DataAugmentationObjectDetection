# DataAugmentationObjectDetection
Increase the amount of data available for training an object detection model to reduce the need for time consuming manual labelling of imges. A vast number of labelled images is still nessecary though. In theory, you could augment one labelled image an infinite number of times but model performane would be affected. These scripts take images and XML files that contain labels for the images and performs a combination of augmentations on the images and bounding boxes by N number of times. The resulting files are text files that have been converted from PASCAL VOC annotation in xml format (xmin, ymin, xmax, ymax) to YOLO txt format (x, y, h, w). The resulting text files are ready for YOLO object detection training. I've uploaded a jupyter notebook and python script for this task.

Augmentations include: 
1)  Flipping
2)  Rotating
3)  Sheering
4)  Gaussian noise
5)  Adjusting brightness.

The script applies two random augmentations to images. We tried to not include augmentations where extreme colour changes are used as for our project we found psychadelic colour alterations drastically affected model preformance.

Original Image
![VID_20191003_105113880_frame96](https://user-images.githubusercontent.com/57613411/140636380-106f2012-d571-448f-80ee-436219f50d4f.jpg)

Augmentation applied
![aug0_VID_20191003_105113880_frame0](https://user-images.githubusercontent.com/57613411/140636386-5fa7c5e3-04e8-4f81-96aa-9ebf7c3c70ee.jpg)

Augmentation applied
![aug2_VID_20191003_105113880_frame0](https://user-images.githubusercontent.com/57613411/140636412-d4e5c243-28b6-45fb-88c2-e3db3a58d390.jpg)

These scripts were borrowed/copied/inspired (whatever you want to call it) by others:

https://github.com/asetkn/Tutorial-Image-and-Multiple-Bounding-Boxes-Augmentation-for-Deep-Learning-in-4-Steps

https://github.com/aleju/imgaug

FYI: There is an issue with this script in that it produces some annotation files with negative values which cannot be used for training. I am still working on rectifying these issues and will update this repo in time.
