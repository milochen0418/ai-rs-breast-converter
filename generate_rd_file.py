
from __future__ import print_function, division

# Step 1.
# collect all data in Frankie_Dataset
def test_case_get_process_inputs():
    print('Run test case of get_process_inputs')
    root_folder = 'Frankie_Dataset'
    inputs = get_process_inputs(root_folder)
    print('len(inputs) = ', len(inputs))
    for idx in range(len(inputs)):
        f_dict = inputs[idx]
        print('[idx = {}]'.format(idx))
        print('.keys() = {}'.format(f_dict.keys()))
        for key in f_dict.keys():
            val = f_dict[key]
            if type(val) == list:
                print('{} (with len={}) -> '.format(key, len(val)))
                itemlist = []
                if len(val) > 5:
                    itemlist = val[:5]
                else:
                    itemlist = val[:]
                for item in itemlist:
                    print('\t{}'.format(item))
                print('\t...')
            else:
                print('{} -> {}'.format(key, val))
def get_process_inputs(root_folder):
    # The function will return list of dict
    # one of key in dict is ct_filelist, whose value is list of ct filepath(string)
    # another keys in dict are rs_filepath, dose_filepath and plan_filepath, whose value is filepath(string)
    import os
    ret_list = []
    if not os.path.isdir(root_folder):
        return ret_list
    for tmp_file_d1 in os.listdir(root_folder):
        tmp_folder_path_d1 = r"{}/{}".format(root_folder, tmp_file_d1)
        # example tmp_folder_path_d1 = Frankie_Dataset/10043024
        if not os.path.isdir(tmp_folder_path_d1):
            continue
        for tmp_file_d2 in os.listdir(tmp_folder_path_d1):
            tmp_folder_path_d2 = r"{}/{}".format(tmp_folder_path_d1, tmp_file_d2)
            # example tmp_folder_path_d2 = Frankie_Dataset/10043024/C1_20180809
            if not os.path.isdir(tmp_folder_path_d2):
                continue
            for tmp_file_d3 in os.listdir(tmp_folder_path_d2):
                data_folder_path = r"{}/{}".format(tmp_folder_path_d2, tmp_file_d3)
                # example data_folder_path = Frankie_Dataset/10043024/C1_20180809/C1NPC7000/
                if not os.path.isdir(data_folder_path):
                    continue
                # In data_folder_path, we have
                # ./CT , ./Dose, ./Plan and ./Structure
                ct_folder_path = r"{}/{}".format(data_folder_path, 'CT')
                dose_folder_path = r"{}/{}".format(data_folder_path, 'Dose')
                plan_folder_path = r"{}/{}".format(data_folder_path, 'Plan')
                rs_folder_path = r"{}/{}".format(data_folder_path, 'Structure')
                if not os.path.isdir(ct_folder_path):
                    continue
                if not os.path.isdir(dose_folder_path):
                    continue
                if not os.path.isdir(plan_folder_path):
                    continue
                if not os.path.isdir(rs_folder_path):
                    continue
                try:
                    ct_filelist = []
                    for ct_file in os.listdir(ct_folder_path):
                        ct_filepath = r"{}/{}".format(ct_folder_path, ct_file)
                        ct_filelist.append(ct_filepath)
                    dose_filepath = r"{}/{}".format(dose_folder_path, os.listdir(dose_folder_path)[0])
                    plan_filepath = r"{}/{}".format(plan_folder_path, os.listdir(plan_folder_path)[0])
                    rs_filepath = r"{}/{}".format(rs_folder_path, os.listdir(rs_folder_path)[0])
                    f_dict = {}
                    f_dict['dose_filepath'] = dose_filepath
                    f_dict['plan_filepath'] = plan_filepath
                    f_dict['rs_filepath'] = rs_filepath
                    f_dict['ct_filelist'] = ct_filelist
                    ret_list.append(f_dict)

                except:
                    print(r"Exception of {} ,when call get_process_inputs().".format(data_folder_path))

    # ret_list.append(f_dict)
    ret_list = sorted(ret_list, key=lambda f_dict: f_dict['rs_filepath'], reverse=False)
    return ret_list



# Step 2.
# AI predict from of RS & CT to the output dose volume python object
# Step 2.1.
# Save the AI output into xxx.bytes file if you need it (You can avoid computing repeatly
# will save to output/pyobj.bytes
# Basic stuff element for implement AI_predict_for_rd_pixel_array


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import pydicom
import numpy
import scipy
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adamax
import datetime
import matplotlib.image as mpimg
import sys
from data_loader import DataLoader
import json
import tensorflow as tf
from skimage import transform, data
class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        # self.initial_epoch = 5
        self.path_Epoch = 1610
        self.path_name = 'DGX_GAN_256_3CH'
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64 #64
        self.df = 64 #64

        optimizer = Adamax(0.00002, 0.5) # Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # save_model_path = 'saved_model/{}/predict_weights_{}.hdf5'.format(self.path_name, self.path_Epoch)
        save_model_path = 'predict_weights_635.hdf5'
        self.generator.load_weights(save_model_path)

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        # save_model_path = 'saved_model/generator_weights_96_BS8.hdf5'
        # self.combined.load_weights(save_model_path)
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                #fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                csvfile = "Epoch,%d,%d,Dloss,%f,acc,%3d%%,Gloss,%f\n" % (epoch, epochs, d_loss[0], 100 * d_loss[1],
                                                                         g_loss[0])
                file = open('Loss.csv', 'a+')
                file.write(csvfile)
                file.close()

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.save_model(epoch)
                    #self.sample_images(epoch, batch_i)


    # def sample_images(self, epoch, batch_i):
    #     os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
    #     r, c = 3, 3
    #
    #     imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
    #     fake_A = self.generator.predict(imgs_B)
    #
    #     gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
    #
    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #
    #     titles = ['Condition', 'Generated', 'Original']
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt])
    #             axs[i, j].set_title(titles[i])
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
    #     plt.close()


    def pred_image_fake_a(self, imgs_B):
        imgs_B_new = []
        imgs_B_new.append(imgs_B)
        # print(imgs_B_new.shape)
        imgs_B_new = np.array(imgs_B_new) / 50. - 1.
        # imgs_B_new = np.reshape(imgs_B, (1, self.img_rows, self.img_cols, 3))
        fake_A = self.generator.predict(imgs_B_new)
        print(fake_A.dtype)
        fake_Aa = np.reshape(fake_A, (self.img_rows, self.img_cols, 3))
        # fake_AaA = 0.5 * fake_Aa + 0.5
        return fake_Aa
gan = Pix2Pix()
path="Frankie_Dataset"
# output=path + "\\JPG_RG_3ch\\"
Structure_list = path + "\\structure.csv"
target_list = path + "\\target.csv"



# python object dump tool for you to save all temp python object file of AI output
def python_object_dump(obj, filename):
    import os
    import time
    import pickle
    file_w = open(filename, "wb")
    pickle.dump(obj, file_w)
    file_w.close()
def python_object_load(filename):
    import os
    import time
    import pickle
    file_r = open(filename, "rb")
    obj2 = pickle.load(file_r)
    file_r.close()
    return obj2

def AI_predict_for_rd_pixel_array(rs_filepath, ct_filelist):
    import os

    Structure_list = path + "\\structure.csv"
    target_list = path + "\\target.csv"
    print("CALL AI_predict_for_rd_pixel_array() start ")
    print("Structure_list = {}".format(Structure_list))
    print("target_list = {}".format(target_list))


    RS = rs_filepath
    dicom_rt_structure = pydicom.dcmread(RS)
    RS_RefUID = dicom_rt_structure.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

    structures = [line.rstrip('\n') for line in open(Structure_list)]
    structure_array = []
    to_delete = []
    for st in structures:
        structure_array.append(st.split(","))

    for item in range(len(dicom_rt_structure.RTROIObservationsSequence)):
        if (dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel == "Cochlea_R" or
                dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel == "CN_VIII_R"):
            dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel = "R-ear"
        if (dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel == "Cochlea_L" or
                dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel == "CN_VIII_L"):
            dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel = "L-ear"
        for st in range(len(structures)):
            if (dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel == str(structure_array[st][0])):
                structure_array[st][2] = item
    for item in structure_array:
        if (item[2] == ''):
            to_delete.append(item)
    for delete in to_delete:
        structure_array.remove(delete)

    targets = [line.rstrip('\n') for line in open(target_list)]
    target_array = []
    to_delete2 = []
    for st in targets:
        target_array.append(st.split(","))
    for item in range(len(dicom_rt_structure.RTROIObservationsSequence)):
        for st in range(len(targets)):
            if (dicom_rt_structure.RTROIObservationsSequence[item].ROIObservationLabel == str(target_array[st][0])):
                target_array[st][2] = item
    for item in target_array:
        if (item[2] == ''):
            to_delete2.append(item)
    for delete in to_delete2:
        target_array.remove(delete)

    ct_name_z_list = []
    ct_location_list = []

    for ct_filepath in ct_filelist:
        dicom_ct_image = pydicom.read_file(ct_filepath)
        ct_name_z_list.append(int(dicom_ct_image.SliceLocation))
        ct_location_list.append(ct_filepath + "," + str(dicom_ct_image.SliceLocation))

    # dicom_ct_image = pydicom.dcmread(CT_list[0])
    CT_origin_x = float(dicom_ct_image.ImagePositionPatient[0])
    CT_origin_y = float(dicom_ct_image.ImagePositionPatient[1])
    CT_origin_z = float(dicom_ct_image.ImagePositionPatient[2])
    Table_H = float(dicom_ct_image.TableHeight)
    CT_ps_x = float(dicom_ct_image.PixelSpacing[0])
    CT_ps_y = float(dicom_ct_image.PixelSpacing[1])
    CT_columns = int(dicom_ct_image.Columns)
    CT_rows = int(dicom_ct_image.Rows)
    CT_FOV_X = CT_columns * CT_ps_x
    CT_FOV_Y = CT_rows * CT_ps_y
    CT_Slope = float(dicom_ct_image.RescaleSlope)
    CT_Intercept = float(dicom_ct_image.RescaleIntercept)

    x, y = np.meshgrid(np.linspace(0, CT_columns, CT_columns), np.linspace(0, CT_rows, CT_rows))
    x, y = x.flatten(), y.flatten()
    dosegridpoints = np.vstack((x, y)).T

    print("436 FFFF")
    ########################################################################################
    try:
        mask100 = numpy.zeros((CT_columns, CT_rows, len(ct_name_z_list)), numpy.float32)
        num = 0
        for slice_location in tqdm.tqdm(ct_name_z_list):
            for ctname in ct_location_list:
                if (ctname.split(',')[1]) == str(slice_location):
                    dicom_ct_image = pydicom.dcmread(ctname.split(',')[0])
            mask97 = dicom_ct_image.pixel_array * CT_Slope + CT_Intercept
            # mask97[np.where(mask97 > 128)] = 128
            # mask97[np.where(mask97 < -127)] = -127
            # mask97 = mask97 + 127
            # mask97 = mask97.round(0).astype(int)
            ###########################################################################################################
            mask98 = numpy.zeros((CT_columns, CT_rows), numpy.uint8)
            mask98t = numpy.zeros((CT_columns, CT_rows), numpy.uint8)
            # body_mask = numpy.zeros((CT_columns, CT_rows), numpy.uint8)

            for ROIclass in target_array:
                if len(dicom_rt_structure.ROIContourSequence[ROIclass[2]]) == 3:
                    for Contour in dicom_rt_structure.ROIContourSequence[ROIclass[2]].ContourSequence:
                        if ((Contour.ContourData[2]) == slice_location):
                            xyarray = np.array(Contour.ContourData, numpy.float)
                            xyarray = xyarray.reshape((int(len(Contour.ContourData) / 3)), 3)
                            xyarray[:, 0] = ((xyarray[:, 0] - CT_origin_x) / CT_ps_x).round(0)
                            xyarray[:, 1] = ((xyarray[:, 1] - CT_origin_y) / CT_ps_y).round(0)
                            xyarray[:, 2] = ct_name_z_list.index(int(xyarray[0, 2]))
                            xyarray = xyarray.astype(int)
                            temp = matplotlib.path.Path(list(xyarray[:, :2]))
                            temp = temp.contains_points(dosegridpoints)
                            temp = temp.reshape(CT_columns, CT_rows)
                            mask98t[temp == True] = int(ROIclass[1])
            for ROIclass in structure_array:
                if len(dicom_rt_structure.ROIContourSequence[ROIclass[2]]) == 3:
                    for Contour in dicom_rt_structure.ROIContourSequence[ROIclass[2]].ContourSequence:
                        if ((Contour.ContourData[2]) == slice_location):
                            xyarray = np.array(Contour.ContourData, numpy.float)
                            xyarray = xyarray.reshape((int(len(Contour.ContourData) / 3)), 3)
                            xyarray[:, 0] = (xyarray[:, 0] - CT_origin_x) / CT_ps_x
                            xyarray[:, 1] = (xyarray[:, 1] - CT_origin_y) / CT_ps_y
                            xyarray[:, 2] = ct_name_z_list.index(int(xyarray[0, 2]))
                            xyarray = xyarray.astype(int)

                            temp = matplotlib.path.Path(list(xyarray[:, :2]))
                            temp = temp.contains_points(dosegridpoints)
                            temp = temp.reshape(CT_columns, CT_rows)
                            mask98[temp == True] = int(ROIclass[1])
                            # if(ROIclass[0]=="BODY"):
                            #     body_mask[temp==True]=1
            # print('486 FFFF')
            # if np.max(mask98) > 80 or np.max(mask98t) > 0:  # or  np.max(mask98t) > 0
            # print('488 FFFF')
            mask98_rgb = np.dstack((mask97, mask98t, mask98))

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)

            # ax1.title.set_text("Prediction")
            # ax1.imshow(fake_Aa, cmap='jet')
            #
            # ax2.title.set_text("Ground-Truth")
            # ax2.imshow(img_Ass, cmap='jet')
            #
            #
            # ax3.title.set_text("CT")
            # ax3.imshow(CT[:,:,0], cmap='bone')

            ax1.title.set_text("CT")
            #ax1.imshow(CT[:, :, 0])
            ax1.imshow(mask97, cmap='jet')

            ax2.title.set_text("T")
            #ax2.imshow(CT[:, :, 1], cmap='jet')
            ax2.imshow(mask98t, cmap='jet')

            ax3.title.set_text("S")
            #ax3.imshow(CT[:, :, 2], cmap='jet')
            ax3.imshow(mask98, cmap='jet')

            #plt.savefig('pred_image/{}/test/{}_{}_{}.png'.format(E, E, test_image_name[:-4], u))
            plt.savefig('pred_image/{}.png'.format(num))
            plt.close()

            # plt.imshow(mask98_rgb[:,:,2])
            # plt.show()

            # n = n + 1
            # '''GAN Result'''

            # imgs_B = np.array(mask98_rgb) / 50. - 1.

            result_image = gan.pred_image_fake_a(mask98_rgb)

            # generated_volume = ((result_image[:, :, 2] * 65536 * 255) + (
            #             result_image[:, :, 1] * 256 * 255) + result_image[:, :, 0] * 255) / 1500
            generated_volume_1 = result_image + 1
            generated_volume_2 = generated_volume_1 * 50.
            generated_volume_2[generated_volume_2 > 45] = 45
            generated_volume = generated_volume_2
            # 1500 is 15 million
            # 255 is because 0 to 1
            # mask100[:, :, num] = generated_volume[:, :] * 0.74
            mask100[:, :, num] = generated_volume[:, :, 0]
            # Because maximum dose is 7400


            # print(np.max(generated_volume))
            # plt.imshow(generated_volume[:,:,0])
            # plt.show()

            num = num + 1

        print(mask100.shape)
        print(mask100.dtype)
        print(np.max(mask100))
        print(np.min(mask100))
        #np.save('GAN_Test_Frankie.npy', mask100)
        np.save(r"TestCase_Breast_Input_CtFolder/GAN_Test_Frankie.npy", mask100)

    except:

        raise
        #print(id, str(n))
        #fail.append(id + '_sn:' + str(n) + ',')
        #return None
        pass
    return mask100
    pass


# TEST CASE for AI_predict_for_rd_pixel_array
def test_case_AI_predict_for_rd_pixel_array():
    root_folder = 'Frankie_Dataset'
    inputs = get_process_inputs(root_folder)
    f_dict = inputs[0]
    rs_filepath = f_dict['rs_filepath']
    ct_filelist = f_dict['ct_filelist']
    pyobj = AI_predict_for_rd_pixel_array(rs_filepath, ct_filelist)
    python_object_dump(pyobj, "new-mask100.bytes")
    print('pyobj.shape = ', pyobj.shape)
    print('pyobj.dtype = ', pyobj.dtype)
    pass

def get_process_list(root_folder):
    process_list = []
    import os
    root_folder = 'Frankie_Dataset'
    inputs = get_process_inputs(root_folder)
    for f_dict in inputs:
        rs_filepath = f_dict['rs_filepath']
        print(rs_filepath)
        p_dict = {}  # will be major element for process_list. it will set output key and input key
        o_dict = {}  # will be p_dict['output']
        basedir = os.path.dirname(os.path.dirname(rs_filepath))
        output_folder = r"{}/{}".format(basedir, "output")

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, mode=0o777)
        o_dict['rdai_filepath'] = r"{}/{}".format(output_folder, 'rd.ai.output.dcm')  #
        o_dict['bytes_filepath'] = r"{}/{}".format(output_folder, 'pyobj.bytes')  # bytes file made by
        p_dict['input'] = f_dict
        p_dict['output'] = o_dict
        process_list.append(p_dict)
    return process_list

def test_case_outputAll_AI_predict_for_rd_pixel_array():
    process_list = []
    # each p_dict in process_list has
    # Step 1. set basic path for process list
    import os
    root_folder = 'Frankie_Dataset'
    # inputs = get_process_inputs(root_folder)
    process_list = get_process_list(root_folder)
    """
    for f_dict in inputs:
        rs_filepath = f_dict['rs_filepath']
        print(rs_filepath)
        p_dict = {} # will be major element for process_list. it will set output key and input key 
        o_dict = {} # will be p_dict['output']
        basedir = os.path.dirname(os.path.dirname(rs_filepath))
        output_folder = r"{}/{}".format(basedir, "output")

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, mode=0o777)    
        o_dict['rdai_filepath'] = r"{}/{}".format(output_folder, 'rd.ai.output.dcm') #
        o_dict['bytes_filepath'] = r"{}/{}".format(output_folder, 'pyobj.bytes') # bytes file made by 
        p_dict['input'] = f_dict
        p_dict['output'] = o_dict
        process_list.append(p_dict)
        #print(p_dict['output'])
    """

    # Step 2. Start to ai process for each element
    for p_dict in process_list:
        rs_filepath = p_dict['input']['rs_filepath']
        ct_filelist = p_dict['input']['ct_filelist']
        bytes_filepath = p_dict['output']['bytes_filepath']
        rdai_filepath = p_dict['output']['rdai_filepath']
        print('call AI_predict_for_rd_pixel_array')
        pyobj = AI_predict_for_rd_pixel_array(rs_filepath, ct_filelist)
        print('end of call AI_predict_for_rd_pixel_array')
        python_object_dump(pyobj, bytes_filepath)
    pass


# Step 3.
# wrap dose volume python object with RD template into AI RD.dcm file (save to correct place ?)
# Implementation of def wrap_prediction_rd_file(rd_template_filepath, ai_ouput_pyobj, rd_output_filepath):
def wrap_prediction_rd_file(rd_template_filepath, ai_output_pyobj, rd_output_filepath, ct_filelist):
    import pydicom
    import os
    import numpy as np
    import copy
    import datetime

    def debug(*args, **kwargs):
        is_debug = False
        if is_debug == True:
            print(*args, **kwargs)

    z_dict = {}
    sorted_ct_filelist = []
    for ct_filepath in ct_filelist:
        ct_fp = pydicom.read_file(ct_filepath)
        z = ct_fp.SliceLocation
        z_dict[z] = ct_filepath

    for z in sorted(z_dict.keys()):
        sorted_ct_filelist.append(z_dict[z])

    ct_fp = None
    # ct_fp = pydicom.read_file(ct_filelist[0])
    ct_fp = pydicom.read_file(sorted_ct_filelist[0])

    rd_tmp_fp = pydicom.read_file(rd_template_filepath)

    debug(rd_tmp_fp.dir())

    slice_num = len(sorted_ct_filelist)
    origin_z = 0
    end_z = origin_z + (slice_num - 1) * ct_fp.SliceThickness
    GridFrame = np.arange(origin_z, end_z + ct_fp.SliceThickness, ct_fp.SliceThickness)

    rd_tmp_fp.Columns = 512
    rd_tmp_fp.Rows = 512
    rd_tmp_fp.NumberOfFrames = len(sorted_ct_filelist)

    rd_tmp_fp.PixelSpacing = ct_fp.PixelSpacing
    rd_tmp_fp.PixelSpacing = ct_fp.PixelSpacing

    # Process GridFrameOffsetVector
    gridF = []
    for v in GridFrame:
        gridF.append(v)
    rd_tmp_fp.GridFrameOffsetVector = gridF

    rd_tmp_fp.DoseGridScaling = 1

    rd_tmp_fp.PatientID = ct_fp.PatientID
    rd_tmp_fp.PatientName = ct_fp.PatientName
    rd_tmp_fp.PatientBirthDate = ct_fp.PatientBirthDate
    rd_tmp_fp.PatientBirthTime = ct_fp.PatientBirthTime
    rd_tmp_fp.PatientSex = ct_fp.PatientSex
    rd_tmp_fp.PhysiciansOfRecord = ct_fp.PhysiciansOfRecord

    now = datetime.datetime.now()
    rd_tmp_fp.SOPInstanceUID = "1.2.246.352.71.7.417454940236.2863736.{}".format(now.strftime("%Y%m%d%H%M%S"))
    # template example of SOPIntanceUID  : 1.2.246.352.71.7.417454940236.2863736.20180322135900

    rd_tmp_fp.StudyDate = ct_fp.StudyDate
    rd_tmp_fp.StudyTime = ct_fp.StudyTime
    rd_tmp_fp.Manufacturer = "clickLab"
    rd_tmp_fp.SeriesDescription = "AI_project"
    rd_tmp_fp.SliceThickness = ct_fp.SliceThickness
    rd_tmp_fp.StudyInstanceUID = ct_fp.StudyInstanceUID
    #rd_tmp_fp.SeriesInstanceUID = "1.2.826.0.1.3680043.2.200.1276805877.135.65597.2107.{}".format(now.strftime("%Y%m%d%H%M%S"))
    rd_tmp_fp.SeriesInstanceUID = "1.2.826.0.1.3680043.2.200.1276805877.135.65597.2107.{}".format(
        now.strftime("%S"))
    # template example of SeriesInstnaceUID = 1.2.826.0.1.3680043.2.200.1276805877.135.65597.2107.1
    rd_tmp_fp.StudyID = ct_fp.StudyID

    rd_tmp_fp.ImagePositionPatient[0] = ct_fp.ImagePositionPatient[0]
    rd_tmp_fp.ImagePositionPatient[1] = ct_fp.ImagePositionPatient[1]
    rd_tmp_fp.ImagePositionPatient[2] = ct_fp.ImagePositionPatient[2]
    rd_tmp_fp.ImageOrientationPatient[0] = ct_fp.ImageOrientationPatient[0]
    rd_tmp_fp.ImageOrientationPatient[1] = ct_fp.ImageOrientationPatient[1]
    rd_tmp_fp.ImageOrientationPatient[2] = ct_fp.ImageOrientationPatient[2]
    rd_tmp_fp.ImageOrientationPatient[3] = ct_fp.ImageOrientationPatient[3]
    rd_tmp_fp.ImageOrientationPatient[4] = ct_fp.ImageOrientationPatient[4]
    rd_tmp_fp.ImageOrientationPatient[5] = ct_fp.ImageOrientationPatient[5]

    rd_tmp_fp.FrameOfReferenceUID = ct_fp.FrameOfReferenceUID

    # rd_tmp_fp.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID="1.2.246.352.71.5.417454940236.1790316.20180322093133"
    # template example "1.2.246.352.71.5.417454940236.1790316.20180322093127"
    rd_tmp_fp.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = "1.2.246.352.71.5.417454940236.1790316.{}".format(
        now.strftime("%Y%m%d%H%M%S"))
    # print(rd_tmp_fp.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID)

    # debug('type(rd_tmp_fp.pixel_array) = ', type(rd_tmp_fp.pixel_array))
    # debug('type(rd_tmp_fp.pixel_array) = ', type(rd_tmp_fp.pixel_array))

    # pixel_array = np.zeros( dtype=np.uint32, shape=(ai_output_pyobj.shape[2], ai_output_pyobj.shape[0], ai_output_pyobj.shape[1] ))
    pixel_array = np.zeros(dtype=np.uint32,
                           shape=(ai_output_pyobj.shape[2], ai_output_pyobj.shape[0], ai_output_pyobj.shape[1]))
    depth_max = ai_output_pyobj.shape[2]

    # reverse order of AI's output with depth
    # for depth in range(ai_output_pyobj.shape[2]):
    for depth in range(depth_max):
        # pixel_array[depth,:,:] = ai_output_pyobj[:,:,depth]
        pixel_array[(depth_max - depth - 1), :, :] = ai_output_pyobj[:, :, depth]

        # On Does Units, We use cGY in instead for fit AI output
    # rd_tmp_fp.DoseUnits = 'cGY'

    # On DoseGridScaling, We use 0.01 to instread 1.0 for fit AI output
    rd_tmp_fp.DoseGridScaling = 0.01

    debug('ai_output_pyobj.shape = ', ai_output_pyobj.shape)
    debug('ai_output_pyobj.dtype = ', ai_output_pyobj.dtype)
    debug('pixel_array.shape = ', pixel_array.shape)
    debug('pixel_array.dtype = ', pixel_array.dtype)  # expected as uint32
    debug('type(rd_tmp_fp.PixelData) = ', type(rd_tmp_fp.PixelData))
    debug('len(rd_tmp_fp.PixelData) = ', len(rd_tmp_fp.PixelData))
    debug('prepare to assign PixelData')
    theBytes = pixel_array.tobytes()
    debug('type(theBytes) = ', type(theBytes))
    debug('len(theBytes) = ', len(theBytes))
    # rd_tmp_fp.PixelData = pixel_array.tostring()
    # rd_tmp_fp.PixelData = pixel_array.tobytes()
    rd_tmp_fp.PixelData = theBytes
    # print("rd_tmp_fp.pixel_array.shape = ", rd_tmp_fp.pixel_array.shape)
    # refer https://github.com/pydicom/pydicom/issues/808
    # rd_tmp_fp[(0x0028,0x0006)]._value = 0

    # pydicom.write_file(rd_output_filepath, rd_tmp_fp,  write_like_original=True)
    pydicom.write_file(rd_output_filepath, rd_tmp_fp)

    qq_fp = pydicom.read_file(rd_output_filepath)
    debug(sorted(qq_fp.dir()))
    debug("Columns = ", qq_fp.Columns, "Rows = ", qq_fp.Rows)
    debug(qq_fp.pixel_array.shape)
    pass
def test_case_wrap_prediction_rd_file():
    rd_output_filepath = 'rd.ai.output.dcm'
    rd_template_filepath = r"RD.template.dcm"
    #rs_filepath = 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/Structure/RS.1.2.246.352.71.4.417454940236.244087.20180824102353.dcm'
    rs_filepath = r"Frankie_Dataset\25051982\C1_20180827\C1Bre4256\Structure\RS.1.2.246.352.71.4.417454940236.244247.20190418132000.dcm"
    rd_original_filepath = 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/Dose/RD.1.2.246.352.71.7.417454940236.2989177.20180824102340.dcm'
    #ct_folder_filepath = 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT'
    ct_folder_filepath = r"Frankie_Dataset\25051982\C1_20180827\C1Bre4256\CT"
    # ct_filelist = ['Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007721.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007722.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007723.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007724.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007725.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007726.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007727.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007728.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007729.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007730.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007731.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007732.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007733.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007734.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007735.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007736.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007737.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007738.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007739.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007740.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007741.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007742.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007743.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007744.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007745.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007746.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007747.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007748.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007749.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007750.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007751.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007752.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007753.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007754.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007755.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007756.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007757.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007758.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007759.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007760.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007761.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007762.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007763.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007764.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007765.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007766.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007767.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007768.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007769.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007770.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007771.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007772.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007773.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007774.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007775.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007776.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007777.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007778.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007779.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007780.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007781.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007782.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007783.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007784.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007785.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007786.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007787.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007788.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007789.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007790.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007791.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007792.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007793.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007794.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007795.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007796.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007797.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007798.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007799.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007800.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007801.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007802.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007803.dcm', 'Frankie_Dataset/10043024/C1M1_20180824/C1M1NPC5200/CT/CT.1.3.12.2.1107.5.1.4.95999.30000018082300093120600007804.dcm']
    ct_filelist = []
    for file in os.listdir(ct_folder_filepath):
        filepath = os.path.join(ct_folder_filepath, file)
        try:
            fp = pydicom.read_file(filepath)
            if fp.Modality == 'CT':
                ct_filelist.append(filepath)
        except:
            continue
    #ai_output_pyobj = AI_predict_rd_for_pixel_array(rs_filepath, ct_filelist):
    #ai_output_pyobj = python_object_load('mask100.bytes')
    ai_output_pyobj = python_object_load('new-mask100.bytes')
    wrap_prediction_rd_file(rd_template_filepath, ai_output_pyobj, rd_output_filepath, ct_filelist)
    pass

#test_case_AI_predict_for_rd_pixel_array()
#test_case_wrap_prediction_rd_file()

def generate_rd_by_ct_rs(rs_filepath, ct_filelist, output_rd_filepath, is_recreate=True, bytes_filepath="temp-mask.bytes"):
    rd_template_filepath = r"RD.template.dcm"
    pyobj = AI_predict_for_rd_pixel_array(rs_filepath, ct_filelist)
    #python_object_dump(pyobj, bytes_filepath)
    #ai_output_pyobj = python_object_load(bytes_filepath)
    #wrap_prediction_rd_file(rd_template_filepath, ai_output_pyobj, output_rd_filepath, ct_filelist)
    wrap_prediction_rd_file(rd_template_filepath, pyobj, output_rd_filepath, ct_filelist)
    pass
def example_of_generate_rd():
    rs_filepath = r"Frankie_Dataset\24120779\C1_20181031\C1Bre4256\Structure\RS.1.2.246.352.71.4.417454940236.250260.20190418131658.dcm"
    ct_folder_filepath = r"Frankie_Dataset\24120779\C1_20181031\C1Bre4256\CT"
    bytes_filepath = r"new-mask100.bytes"
    output_rd_filepath = r"rd.ai.output.dcm"
    ct_filelist = []
    for file in os.listdir(ct_folder_filepath):
        filepath = os.path.join(ct_folder_filepath, file)
        try:
            fp = pydicom.read_file(filepath)
            if fp.Modality == 'CT':
                ct_filelist.append(filepath)
        except:
            continue
    generate_rd_by_ct_rs(rs_filepath, ct_filelist, output_rd_filepath, bytes_filepath)



if __name__ == "__main__":
    example_of_generate_rd()
