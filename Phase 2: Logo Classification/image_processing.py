# image_processing.py
import cv2
import numpy as np
import os
from keras.applications.vgg16 import preprocess_input, decode_predictions
import tensorflow as tf

cv2.setUseOptimized(True)   #tells OpenCV to use optimized code whenever possible, which can lead to better performance for many operations
cv2.setNumThreads(8)  # This function sets the number of threads that OpenCV will use for parallel operations

def get_iou(bb1, bb2):
    # it checks that the value associated with the key 'x1' is less than the value associated with the key 'x2' in the bb1 dictionary.
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    # smallArea = min(bb1_area, bb2_area)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # iou = intersection_area / float(smallArea)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def load_and_preprocess_image(path, annot):
    train_images = []
    train_labels = []
    train_labelsnum = []
    train_labelsnumSet = set()

    with open(annot) as file:
        count = 0
        mycount = 0
        for item in file:
            print(item , "   -->   ",count)
            #if mycount == 5:
            #    break
            fileData = item.split(" ")
            filename = fileData[0]
            classLbl = fileData[1]
            x1 = int(fileData[3])
            y1 = int(fileData[4])
            x2 = int(fileData[5])
            y2 = int(fileData[6])

            #print( filename) thaer

            ############
            image = cv2.imread(os.path.join(path, filename))

            gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

            gs.setBaseImage(image)

            gs.switchToSelectiveSearchFast()

            rects = gs.process()
            sizeRec = len(rects)

            gtvalues = []

            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "lbl": classLbl})
            train_labels.append(str(classLbl))
            train_labelsnumSet.add(classLbl)
            train_labelsnum.append(list(train_labelsnumSet).index(classLbl))

            imout = image[y1:y1 + y2, x1:x1 + x2]
            #resized = cv2.resize(imout, (224, 224), interpolation=cv2.INTER_AREA)
            # added by thar
            resized = tf.image.resize(imout, (224, 224))
            resized = preprocess_input(resized)
            #resized_numpy = resized.numpy()  # Convert TensorFlow tensor to numpy array
            train_images.append(resized)


            for i in range(sizeRec):

                x21, y21, w2, h2 = rects[i]

                imgOutTest = image[y21:y21+h2, x21:x21+w2]
                try:
                    iou = get_iou({"x1": x1, "x2": x2, "y1": y1, "y2": y2},
                        {"x1": x21, "x2": x21 + w2, "y1": y21, "y2": y21 + h2})

                    if iou > 0.9:
                        count = count +1

                        gtvalues = []

                        gtvalues.append({"x1": x21, "x2": x21 + w2, "y1": y21, "y2": y21 + h2, "lbl": classLbl})
                        train_labels.append(str(classLbl))
                        train_labelsnumSet.add(classLbl)
                        train_labelsnum.append(list(train_labelsnumSet).index(classLbl))

                        imout = image[y21:y21+h2, x21:x21+w2]
                        #resized = cv2.resize(imout, (224, 224), interpolation=cv2.INTER_AREA)
                        #====== added by thaer =============================================
                        resized = tf.image.resize(imout, (224, 224))
                        #resized = resized.astype('float32')
                        resized = preprocess_input(resized)
                        #resized_numpy = resized.numpy()  # Convert TensorFlow tensor to numpy array
                        train_images.append(resized)
                except:
                    print ('Dont worry')

            ##########

            count = count +1
            mycount = mycount +1
    return train_images, train_labels, train_labelsnum, train_labelsnumSet

def load_and_preprocess_image_without_selective(path, annot):
    train_images = []
    train_labels = []
    train_labelsnum = []
    train_labelsnumSet = set()

    with open(annot) as file:
        count = 0
        mycount = 0
        for item in file:
            print(item , "   -->   ",count)
            #if mycount == 5:
            #    break
            fileData = item.split(" ")
            filename = fileData[0]
            classLbl = fileData[1]
            x1 = int(fileData[3])
            y1 = int(fileData[4])
            x2 = int(fileData[5])
            y2 = int(fileData[6])

            #print( filename) thaer

            ############
            image = cv2.imread(os.path.join(path, filename))

            #gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

            #gs.setBaseImage(image)

            #gs.switchToSelectiveSearchFast()

            #rects = gs.process()
            #sizeRec = len(rects)

            gtvalues = []

            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "lbl": classLbl})
            train_labels.append(str(classLbl))
            train_labelsnumSet.add(classLbl)
            train_labelsnum.append(list(train_labelsnumSet).index(classLbl))

            imout = image[y1:y1 + y2, x1:x1 + x2]
            #resized = cv2.resize(imout, (224, 224), interpolation=cv2.INTER_AREA)
            # added by thar
            resized = tf.image.resize(imout, (224, 224))
            resized = preprocess_input(resized)
            #resized_numpy = resized.numpy()  # Convert TensorFlow tensor to numpy array
            train_images.append(resized)

            '''
            for i in range(sizeRec):

                x21, y21, w2, h2 = rects[i]

                imgOutTest = image[y21:y21+h2, x21:x21+w2]
                try:
                    iou = get_iou({"x1": x1, "x2": x2, "y1": y1, "y2": y2},
                        {"x1": x21, "x2": x21 + w2, "y1": y21, "y2": y21 + h2})

                    if iou > 0.7:
                        count = count +1

                        gtvalues = []

                        gtvalues.append({"x1": x21, "x2": x21 + w2, "y1": y21, "y2": y21 + h2, "lbl": classLbl})
                        train_labels.append(str(classLbl))
                        train_labelsnumSet.add(classLbl)
                        train_labelsnum.append(list(train_labelsnumSet).index(classLbl))

                        imout = image[y21:y21+h2, x21:x21+w2]
                        #resized = cv2.resize(imout, (224, 224), interpolation=cv2.INTER_AREA)
                        #====== added by thaer =============================================
                        resized = tf.image.resize(imout, (224, 224))
                        #resized = resized.astype('float32')
                        resized = preprocess_input(resized)
                        #resized_numpy = resized.numpy()  # Convert TensorFlow tensor to numpy array
                        train_images.append(resized)
                except:
                    print ('Dont worry')
                '''
            ##########

            count = count +1
            mycount = mycount +1
    return train_images, train_labels, train_labelsnum, train_labelsnumSet