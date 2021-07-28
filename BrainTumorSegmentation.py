import os
import sys
import gc
import pickle
import glob

import numpy as np
import random

import imageio
import nibabel as nib

from keras.layers import  Input, Conv2D, Conv2DTranspose, Concatenate, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class Dataset:
    def __init__(self ,mode ):

        ###################################
        # X data will be 4 channels
        ###################################
        self.data_types = ['flair', 't1', 't1ce', 't2']

        ###################################
        # file path, name informations
        ###################################
        self.train_dir_list = []
        self.test_dir_list = []

        self.train_name_list = []
        self.validate_name_list = []
        self.test_name_list = []

        self.train_name_list = []
        self.test_name_list = []

        self.train_name = []
        self.test_name = []

        ###################################
        # datas for train, validation ,test
        ###################################
        self.X_train_input = []  # data load shape ( num, h , w , 4)
        self.X_train_target = [] # data load shape ( num, h , w , 3) 3 labels

        self.X_val_input = []
        self.X_val_target = []

        self.X_test_input = []
        self.X_test_target = []

        ###################################
        # load data file informations
        ###################################
        self.load_data_path()

        ##################################
        # load mean, std for all datas
        ##################################
        with open(save_dir + 'mean_std_dict.pickle', 'rb') as f:
            data = pickle.load(f)
            self.mean_std_dict = data

        #################################
        # uncomment if no mean / std value
        # this function will save mean / std
        #################################
        # self.mean_std_dict = self.image_normalization()

        if mode == 'train':
            self.split_data()
            # load 2d slice datas in axial orientation
            self.load_image_train()
        else:
            # load 2d slice datas in axial orientation
            self.load_image_test()

    def load_data_path(self):

        if not os.path.isdir(train_data_path):
            sys.exit(' Dataset ' + train_data_path + ' does not exist')
        if not os.path.isdir(test_data_path):
            sys.exit(' Dataset ' + test_data_path + ' does not exist')

        # Image dir names
        self.train_dir_list = sorted(glob.glob(os.path.join(train_data_path, '*')))
        self.test_dir_list = sorted(glob.glob(os.path.join(test_data_path, '*')))
        print(len(self.train_dir_list), len(self.test_dir_list))

        # Image file names
        self.train_name_list = [os.path.basename(x) for x in self.train_dir_list]
        self.test_name_list = [os.path.basename(x) for x in self.test_dir_list]

    def split_data(self):

        # shuffle data
        index = list(range(0, len(self.train_name_list)))
        random.shuffle(index)

        # split validation data and traind data 0.5
        validate_index = index[:len(self.train_name_list) // 2]
        train_index = index[len(self.train_name_list) // 2:]

        self.validate_name_list = [self.train_name_list[i] for i in validate_index]

        # temp
        train_name_list_ = [self.train_name_list[i] for i in train_index]
        self.train_name_list = train_name_list_

        del train_name_list_

    def image_normalization(self):

        mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in self.data_types}

        #####################################
        # get mean and std for all data types
        #####################################

        for i in self.data_types:

            data_temp_list = []

            for j in self.train_name_list:
                img_path = os.path.join(train_data_path, j, j + '_' + i + '.nii.gz')
                img = nib.load(img_path).get_fdata()
                data_temp_list.append(img)

            for j in self.test_name_list:
                img_path = os.path.join(test_data_path, j, j + '_' + i + '.nii.gz')
                img = nib.load(img_path).get_fdata()
                data_temp_list.append(img)

            data_temp_list = np.asarray(data_temp_list)

            mean_std_dict[i]['mean'] = np.mean(data_temp_list)
            mean_std_dict[i]['std'] = np.std(data_temp_list)

        del data_temp_list

        print(mean_std_dict)

        with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
            pickle.dump(mean_std_dict, f, protocol=4)
        return mean_std_dict

    def load_image_train(self):
        mean_std_dict = self.mean_std_dict

        print("Loading Validation data...")

        for i in self.validate_name_list:

            all_input_data = []

            # Input data
            for j in self.data_types:
                img_path = os.path.join(train_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_fdata()
                img = (img - mean_std_dict[j]['mean']) / mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_input_data.append(img)

            # target data
            target_path = os.path.join(train_data_path, i, i + '_seg.nii.gz')
            target_img = nib.load(target_path).get_fdata()
            target_img = np.transpose(target_img, (1, 0, 2))

            for j in range(all_input_data[0].shape[2]):

                # Stack input - 4 channels
                stacked_input = np.dstack(
                    [all_input_data[0][:, :, j], all_input_data[1][:, :, j], all_input_data[2][:, :, j], all_input_data[3][:, :, j]])
                stacked_input = np.transpose(stacked_input, (1, 0, 2))
                stacked_input.astype(np.float32)

                self.X_val_input.append(stacked_input)

                # Stack target - 3 channels
                target_2d = target_img[:, :, j]
                target_2d.astype(int)

                # Loss func - Binary-cross Entropy
                # Make all labels binary class
                # label 1 - necrotic and non - enhancing tumor
                # label 2 - edema
                # label 4 - enhancing tumor
                label_1 = np.where(target_2d == 1, 1, 0)
                label_2 = np.where(target_2d == 2, 1, 0)
                label_4 = np.where(target_2d == 4, 1, 0)

                # stack 3 ch
                stacked_target = np.dstack([label_1, label_2, label_4])

                self.X_val_target.append(stacked_target)

            del all_input_data
            gc.collect()

            print("finished {}".format(i))

        self.X_val_input = np.asarray(self.X_val_input, dtype=np.float32)
        self.X_val_target = np.asarray(self.X_val_target)

        np.save(val_input_path, self.X_val_input)
        np.save(val_target_path, self.X_val_target)

        print("Saved Validation data")

        print("Loading Training data...")

        for i in self.train_name_list:
            all_input_data = []
            for j in self.data_types:
                img_path = os.path.join(train_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_data()
                img = (img - mean_std_dict[j]['mean']) / mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_input_data.append(img)

            target_path = os.path.join(train_data_path, i, i + '_seg.nii.gz')
            target_img = nib.load(target_path).get_data()
            target_img = np.transpose(target_img, (1, 0, 2))

            for j in range(all_input_data[0].shape[2]):
                # Stack input - 4 channels
                stacked_input = np.dstack(
                    [all_input_data[0][:, :, j], all_input_data[1][:, :, j], all_input_data[2][:, :, j], all_input_data[3][:, :, j]])
                stacked_input = np.transpose(stacked_input, (1, 0, 2))  # .tolist()
                stacked_input.astype(np.float32)
                self.X_train_input.append(stacked_input)

                #np.save(train_input_path + '_' + str(i), combined_array)

                # Stack target - 3 channels
                target_2d = target_img[:, :, j]
                target_2d.astype(int)

                # Loss func - Binary-cross Entropy
                # Make all labels binary class
                # label 1 - necrotic and non - enhancing tumor
                # label 2 - edema
                # label 4 - enhancing tumor
                label_1 = np.where(target_2d == 1, 1, 0)
                label_2 = np.where(target_2d == 2, 1, 0)
                label_4 = np.where(target_2d == 4, 1, 0)

                # stack 3 ch
                stacked_target = np.dstack([label_1, label_2, label_4])

                self.X_train_target.append(stacked_target)

            del all_input_data
            print("finished {}".format(i))

        self.X_train_input = np.asarray(self.X_train_input, dtype=np.float32)
        self.X_train_target = np.asarray(self.X_train_target)

        np.save(train_input_path, self.X_train_input)
        np.save(train_target_path, self.X_train_target )
        print("Saved Training data")

    def load_image_test(self):
        mean_std_dict = self.mean_std_dict

        print("Loading test data...")

        for i in self.test_name_list:
            all_input_data = []
            for j in self.data_types:
                img_path = os.path.join(test_data_path, i, i + '_' + j + '.nii.gz')
                img = nib.load(img_path).get_fdata()
                img = (img - mean_std_dict[j]['mean']) / mean_std_dict[j]['std']
                img = img.astype(np.float32)
                all_input_data.append(img)

            target_path = os.path.join(test_data_path, i, i + '_seg.nii.gz')
            target_img = nib.load(target_path).get_fdata()
            target_img = np.transpose(target_img, (1, 0, 2))

            for j in range(all_input_data[0].shape[2]):
                stacked_input = np.dstack(
                    [all_input_data[0][:, :, j], all_input_data[1][:, :, j], all_input_data[2][:, :, j], all_input_data[3][:, :, j]])
                stacked_input = np.transpose(stacked_input, (1, 0, 2))  # .tolist()
                stacked_input.astype(np.float32)

                self.X_test_input.append(stacked_input)

                target_2d = target_img[:, :, j]
                target_2d.astype(int)

                # Loss func - Binary-cross Entropy
                # Make all labels binary class
                # label 1 - necrotic and non - enhancing tumor
                # label 2 - edema
                # label 4 - enhancing tumor
                label_1 = np.where(target_2d == 1, 1, 0)
                label_2 = np.where(target_2d == 2, 1, 0)
                label_4 = np.where(target_2d == 4, 1, 0)

                # 3 channels
                stacked_target = np.dstack([label_1, label_2, label_4])
                self.X_test_target.append(stacked_target)

            del all_input_data

            gc.collect()
            print("finished {}".format(i))

        self.X_test_input = np.asarray(self.X_test_input, dtype=np.float32)
        self.X_test_target = np.asarray(self.X_test_target)

        # save in np (Num of slices, h, w, ch)
        np.save(test_input_path, self.X_test_input)
        np.save(test_target_path, self.X_test_target)


class BraTS:

    def __init__(self, mode):

        os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # Select GPU device

        train_input_np = 'D:/KHK/data/2020010629/Train/Train_input_3label.npy'
        train_target_np = 'D:/KHK/data/2020010629/Train/Train_target_3label.npy'
        test_input_np = 'D:/KHK/data/2020010629/Test/Test_input_3label.npy'
        test_target_np = 'D:/KHK/data/2020010629/Test/Test_target_3label.npy'
        val_input_np = 'D:/KHK/data/2020010629/Val/val_input_3label.npy'
        val_target_np = 'D:/KHK/data/2020010629/Val/val_target_3label.npy'

        #
        # Load data
        #
        if mode == 'train' :
                                                    #If no saved npy
                                                    #dataset = Dataset('train')
            self.X_train = np.load(train_input_np)  #dataset.X_train_input
            self.y_train = np.load(train_target_np) #dataset.X_train_target
            self.X_val = np.load(val_input_np)      #dataset.X_val_input
            self.y_val = np.load(val_target_np)     #dataset.X_val_target

        else:
                                                    #dataset = Dataset('test')
            self.X_test = np.load(test_input_np)    #dataset.X_test_input
            self.y_test = np.load(test_target_np)   #dataset.X_test_target
        #
        # Model param
        #
        self.batch_size = 10
        self.lr = 0.0001
        self.beta1 = 0.9
        self.n_epoch = 100

        #
        # Get Model input shape
        #
        if mode == 'train':
            x = self.X_train.shape[1]
            y = self.X_train.shape[2]
            z = self.X_train.shape[3]
        else:
            x = self.X_test.shape[1]
            y = self.X_test.shape[2]
            z = self.X_test.shape[3]
        self.image_shape = (x, y, z)

        #
        # Build model and compile
        #
        # Loss - binary crossentropy
        # Metrics - Dice coefficient and accuracy
        self.Unet = self.build_unet()
        self.Unet.compile(optimizer=Adam(lr=self.lr, beta_1=self.beta1), loss='binary_crossentropy', metrics= [self.dice_coef, 'accuracy'])

        self.ckpt_path = 'Brain_Tumor_Segmentation/checkpoint/unet_3ch_early_stop.hdf5'
        cp_callback = ModelCheckpoint(filepath=self.ckpt_path ,save_weights_only=True, verbose=1)

        # Allocate GPU as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        if mode =='train':

            print(self.X_train.shape, self.y_train.shape)

            # Enable early stop
            earlystopper = EarlyStopping(patience=8, verbose=1)
            # Decay lr
            decay_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001, verbose=1, cooldown=1)

            # load model if weights exist
            #self.Unet.load_weights(self.ckpt_path)

            # train
            history = self.Unet.fit(self.X_train, self.y_train, epochs=self.n_epoch, batch_size = self.batch_size, validation_data=( self.X_val , self.y_val), callbacks=[earlystopper, cp_callback, decay_lr])

            # save history
            self.draw_history( history)

        else: # test

            # Load model weights
            self.Unet.load_weights(self.ckpt_path)

            # Test img dir
            output_dir = "./Brain_Tumor_Segmentation/output_/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save test result
            for idx in range(self.X_test.shape[0]):
                result = self.Unet.predict(self.X_test[idx][np.newaxis])
                self.save_pred_images( self.X_test[idx], self.y_test[idx], result, "{}/test_{}.png".format(output_dir, idx))

            # Model evaluate
            evaluate = self.Unet.evaluate(self.X_test, self.y_test, batch_size=32)
            print('## loss ## Dice coefficient ## Accuracy ')
            print(evaluate)


##################################################
# Loss / Metrics function
##################################################

    def dice_coef(self, y_true, y_pred):

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    # For minimizing loss func
    # def dice_coef_loss(self, y_true, y_pred):
    #     return 1 - self.dice_coef(y_true, y_pred)

##################################################
# Build U-net model
##################################################

    def build_unet(self):

        inputs = Input(shape= self.image_shape)

        # down1
        conv1_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1_1')(inputs)
        conv1_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1_2')(conv1_1)

        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1_2)

        # down2
        conv2_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_1')(pool1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_2')(conv2_1)

        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2_2)

        # down3
        conv3_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_1')(pool2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_2')(conv3_1)

        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_2)

        # down4
        conv4_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_1')(pool3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_2')(conv4_1)

        pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4_2)

        # down5
        conv5_1 = Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu', name='conv5_1')(pool4)
        conv5_2 = Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu', name='conv5_2')(conv5_1)

        # up4
        up4 = Conv2DTranspose(512, 3, strides=2, padding='same',  name='deconv4')(conv5_2)
        up4 = Concatenate(axis=3, name='concat4')([up4, conv4_2])

        uconv4_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv4_1')(up4 )
        uconv4_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv4_2')(uconv4_1)

        # up3
        up3 = Conv2DTranspose(256, 3, strides=2, padding='same',  name='deconv3')( uconv4_2 )
        up3 = Concatenate(axis=3, name='concat3')([up3, conv3_2])

        uconv3_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv3_1')(up3)
        uconv3_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv3_2')(uconv3_1)

        # up2
        up2 = Conv2DTranspose(128, 3, strides=2, padding='same',  name='deconv2')(uconv3_2)
        up2 = Concatenate(axis=3, name='concat2')([up2, conv2_2])

        uconv2_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv2_1')(up2)
        uconv2_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv2_2')(uconv2_1)

        # up1
        up1 = Conv2DTranspose(64, 3, strides=2, padding='same',  name='deconv1')(uconv2_2)
        up1 = Concatenate(axis=3, name='concat1')([up1, conv1_2])

        uconv1_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv1_1')(up1)
        uconv1_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='uconv1_2')(uconv1_1)

        # Ouput 3 ch
        outputs = Conv2D(filters=3, kernel_size=1, strides=1, padding='same', activation='sigmoid', name='outConv')(uconv1_2)

        model = Model(inputs=inputs, name='u_net', outputs=outputs)

        return model

##################################################
#  Plot loss history
#################################################

    def draw_history(self, history):
        # list all data in history
        if (not type(history) == dict):
            history = history.history
        print(history.keys())

        plt.figure(figsize=(7, 4))

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])

        plt.title('model loss')

        plt.ylabel('loss')
        plt.xlabel('epoch')

        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('history.png')

##################################################
# Save predicted images
##################################################

    def save_pred_images(self, X, label, pred, path, size=(1, 10)):

        if pred.ndim == 2:
            pred = pred[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if pred.ndim == 4:
            pred = pred[0]
        if label.ndim == 4:
            label = label[0]

        images = np.asarray(
            [X[:, :, 0, np.newaxis], X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis],
             label[:, :, 0, np.newaxis], label[:, :, 1, np.newaxis], label[:, :, 2, np.newaxis],
             pred[:, :, 0, np.newaxis], pred[:, :, 1, np.newaxis], pred[:, :, 2, np.newaxis]])

        if np.max(images) <= 1 and (-1 <= np.min(images) < 0):
            images = ((images + 1) * 127.5).astype(np.uint8)

        if np.max(images) <= 1 and np.min(images) >= 0:
            images = (images * 255).astype(np.uint8)

        img = np.zeros((images.shape[1] * size[0], images.shape[2] * size[1], 3), dtype=images.dtype)
        for idx, tmp in enumerate(images):
            r = idx % size[1]
            c = idx // size[1]

            img[r * images.shape[1]:r * images.shape[1] + images.shape[1],
            c * images.shape[2]:c * images.shape[2] + images.shape[2], :] = tmp[:, :, :]

        imageio.imwrite(path, img)




# Define Path
train_data_path = 'D:/KHK/data/2020010629/Training Data/'
test_data_path = 'D:\KHK\data/2020010629/Test Data/'
save_dir = "./Brain_Tumor_Segmentation/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint = "./Brain_Tumor_Segmentation/checkpoint/"
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)


train_input_path = 'D:/KHK/data/2020010629/Train/Train_input_3label'
train_target_path = 'D:/KHK/data/2020010629/Train/Train_target_3label'
test_input_path = 'D:/KHK/data/2020010629/Test/Test_input_3label'
test_target_path = 'D:/KHK/data/2020010629/Test/Test_target_3label'
val_input_path = 'D:/KHK/data/2020010629/Val/val_input_3label'
val_target_path = 'D:/KHK/data/2020010629/Val/val_target_3label'

# train or test
#BraTS('train')
BraTS('test')