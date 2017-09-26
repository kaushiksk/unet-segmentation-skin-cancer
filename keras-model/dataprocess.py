import os
import numpy as np
import glob
from skimage.io import imsave, imread

import matplotlib.pyplot as plt

image_rows = 700
image_cols = 900


def create_train_data():
    train_data_path = ('train/')
    images = glob.glob('train/*gray.jpg')
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:

        image_mask_name = image_name.split('gray')[0] + 'Segmentation.png'
        img = imread(image_name, as_grey=True)
        img_mask = imread(image_mask_name, as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))

            plt.imshow(img[0,:,:], cmap='gray')
            plt.show()
            plt.imshow(img_mask[0,:,:], cmap='gray')
            plt.show()
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = 'test/'
    images = glob.glob('test/*gray.jpg')
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    #imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        #img_id = int(image_name.split('.')[0])
        image_mask_name = image_name.split('gray')[0] + 'Segmentation.png'
        img = imread(image_name, as_grey=True)
        img_mask = imread(image_mask_name, as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        
        imgs_mask[i] = img_mask
        imgs[i] = img
        #imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))

        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_test_mask.npy', imgs_mask)
   # np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_test_mask = np.load('imgs_test_mask.npy')
    #imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_test_mask

if __name__ == '__main__':
    create_train_data()
    create_test_data()
