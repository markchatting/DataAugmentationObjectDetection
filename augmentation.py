import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
import xml.etree.ElementTree as ET
import shutil


# This function was orginally written by https://github.com/asetkn and has been lifted straight from him.
# Function that will extract label data from XML files and arrange into dataframe
def xml_to_df(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# apply xml_to_df() function to convert all XML files in images/ folder into labels.csv
labels_df = xml_to_df('images/')


# function to convert BoundingBoxesOnImage object into DataFrame
def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

# This setup of augmentation parameters will pick two of four given augmenters and apply them in random order
aug = iaa.SomeOf(2, [
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(rotate=(-60, 60)),
    iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
    iaa.Fliplr(1),
    iaa.Multiply((0.5, 1.5)),
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
])


# This function was orginally written by https://github.com/asetkn and has been lifted straight from him.
def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                              )
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():
        #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        #   read the image
        image = imageio.imread(images_path + filename)
        #   get bounding boxes coordinates and write into array
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        #   disregard bounding boxes which have fallen out of image pane
        bbs_aug = bbs_aug.remove_out_of_image()
        #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()

        #   don't perform any actions with the image if there are no bounding boxes left in it
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass

        #   otherwise continue
        else:
            #   write augmented image to a file
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
            #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
            #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix + x)
            #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
            #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

            # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy

# N is the number of augmented images you would like to create for each image in your dataset
N = 8


# Apply augmentation to our images and save files into 'aug_images/' folder. A new prefix (aug1_, aug2_, ...) is created for
# for each augmentation. The updated images and bounding boxesare then written to a new final_augment_df dataframe.
# BEFORE RUNNING THIS BLOCK PLEASE MAKE SURE YOU HAVE CREATED A FOLDER 'aug_images/' IN THE SAME DIRECTOREY AS 'images/'
final_augment_df = labels_df
for i in range(0, N):
    output_df = image_aug(labels_df, 'images/', 'aug_images/', 'aug' + str(i) + '_', aug)
    output_df = output_df.dropna()
    final_augment_df = final_augment_df.append(output_df)
    print('Augmentation ' + str(i) + ' complete!')

print('Augmented image and label dataframe successfully created')


# This function will convert the bounding box values from XML files to those required for YOLO training. Breifley, the XML
# files are labelled with Pascal VOC notation (bounding coordinates in the XML files correspond to bottom right and top left
# corners of the bounding boxes) and need to be converted to YOLO format (x and y coordinate of the centre of each bounding box
# with the width and height of the bounding box normalized for the image width and height)
def convert(xmin, ymin, xmax, ymax, img_w, img_h):
    dw = 1./(img_w)
    dh = 1./(img_h)
    x = (xmin + xmax)/2.0 - 1
    y = (ymin + ymax)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


# This function converts the class name from the XML files to a number needed for the YOLO text file format.
def convert_class(classname):
    if classname == 'plastic':
        return 0
    elif classname == 'metal':
        return 1
    elif classname == 'glass':
        return 2
    elif classname == 'paper':
        return 3
    elif classname == 'fabric':
        return 4
    elif classname == 'rubber':
        return 5
    elif classname == 'processed wood':
        return 6


# These are the classes we are working with. For training your own dataset, you'd need to input your own class names here
# and make sure they correspond to the same numbers in the convert_class() function created above.
classes = ['plastic', 'metal', 'glass', 'paper', 'fabric','rubber', 'processed wood']

# This loop cycles through each row of each file name and creates a YOLO formatted text file that can be used directly in
# YOLO object detection model training (along with corresponding images).
for j in range(len(final_augment_df['filename'].unique())):
    test_df = final_augment_df[final_augment_df.filename == final_augment_df['filename'].unique()[j]]
    xmin_array = np.array(test_df['xmin'])
    ymin_array = np.array(test_df['ymin'])
    xmax_array = np.array(test_df['xmax'])
    ymax_array = np.array(test_df['ymax'])
    w_array = np.array(test_df['width'])
    h_array = np.array(test_df['height'])
    for i in range(len(test_df)):
        output = convert(xmin_array[i], ymin_array[i], xmax_array[i], ymax_array[i], w_array[i], h_array[i])
        object_class = convert_class(np.array(test_df['class'])[i])
        text_output = str(object_class) + ' ' + str(output[0]) + ' ' + str(output[1]) + ' ' + str(
            output[2]) + ' ' + str(output[3])
        print(text_output, file=open('images/' + str(final_augment_df['filename'].unique()[j])[:-4] + '.txt', 'a'))

    print(str(final_augment_df['filename'].unique()[j]))

print('Augmented image and yolo text files successfully created')



# Now all the augmented image files are copied into the images/ folder so that all augmented images and corresponding .txt
# files are in the same folder. If you would like to use these data for training first delete the original XML files.
for file in os.listdir('aug_images'):
    shutil.copy('aug_images/'+file, 'images/'+file)


