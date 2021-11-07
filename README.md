# DataAugmentationObjectDetection
Increase the amount of data available for training an object detection model. These scripts take images and XML files that contain labels for the images and performs a combination of augmentations on the images and bounding boxes by N number of times. The resulting files are text files that have been converted from PASCAL VOC annotation format (xmin, ymin, xmax, ymax) to YOLO format (x, y, h, w). The resulting text files are ready for YOLO object detection training.

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
