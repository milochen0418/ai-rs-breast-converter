import os, sys, pydicom
from decimal import Decimal
from collections import defaultdict
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
def AI_process_get_predict_result(filelist, model_name):
    """
    Check correctness of CT filelist and dispatch suitable model by model_name
    :param filelist:
        CT filelist
    :param model_name:
        The name of AI model
    :return:
        return AI predict result of filelist
    """
    print("AI_process is calling...")
    print("with model_name = {} and filelist = {}".format(model_name, filelist) )
    # PS: Assume input CT files folder that only put one Study Case for one patient
    # filelist check
    if filelist == None:
        print("filelist is None")
        return None
    if len(filelist) == 0:
        print("filelist is Empty")
        return None

    if model_name == "MRCNN_Brachy":
        ret = MRCNN_Brachy_AI_process(filelist)
        return ret
    elif model_name == "MRCNN_Breast":
        ret = MRCNN_Breast_AI_process(filelist)
        return ret
    else:
        return None
def MRCNN_Breast_AI_process(filelist):
    #TODO
    def get_dataset():
        import Mult_Class_Breast
        dataset = Mult_Class_Breast.NeckDataset()
        CLASS_DIR = os.path.join("datasets_dicom")
        dataset.load_Neck(CLASS_DIR, "val")
        dataset.prepare()
        # print(dataset)
        return dataset
    def get_model():
        # Root directory of the project
        ROOT_DIR = os.path.abspath(".")

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        DEVICE = "/gpu:0"
        #MASKRCNN_MODEL_WEIGHT_FILEPATH = r"../ModelsAndRSTemplates/Brachy/MaskRCNN_ModelWeight/mask_rcnn_neck_0082.h5"
        MASKRCNN_MODEL_WEIGHT_FILEPATH = r"./ModelsAndRSTemplates/Breast/MaskRCNN_ModelWeight/mask_rcnn_neck_0020.h5"
        import tensorflow as tf
        import Mult_Class_Breast
        import mrcnn.model as modellib
        config = Mult_Class_Breast.NeckConfig()
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        weights_path = MASKRCNN_MODEL_WEIGHT_FILEPATH
        # Load weights
        print('model.load_weight start')
        model.load_weights(weights_path, by_name=True)
        print('model.load_weight end')
        # Test
        model.detect([np.zeros([512, 512, 3])])
        return model
    dataset = get_dataset()
    model = get_model()
    print('get_model() done')
    label_id_mask = get_label_id_mask(dataset, model, filelist)
    print('Show label_id_mask')

    return label_id_mask


def MRCNN_Brachy_AI_process(filelist):
    print("MRCNN_Brachy_AI_process is calling with filelist = {}".format(filelist) )
    def get_dataset():
        import Mult_Class_Brachy
        dataset = Mult_Class_Brachy.NeckDataset()
        CLASS_DIR = os.path.join("datasets_dicom")
        dataset.load_Neck(CLASS_DIR, "val")
        dataset.prepare()
        return dataset
    def get_model():
        ROOT_DIR = os.path.abspath(".")

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # config.display()
        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.
        # DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        DEVICE = "/gpu:0"

        # MASKRCNN_MODEL_WEIGHT_FILEPATH = r"C:/Users/Milo/Desktop/Milo/ModelsAndRSTemplates/Brachy/MaskRCNN_ModelWeight/mask_rcnn_neck_0082.h5"
        #MASKRCNN_MODEL_WEIGHT_FILEPATH = r"../ModelsAndRSTemplates/Brachy/MaskRCNN_ModelWeight/mask_rcnn_neck_0082.h5"
        MASKRCNN_MODEL_WEIGHT_FILEPATH = r"./ModelsAndRSTemplates/Brachy/MaskRCNN_ModelWeight/mask_rcnn_neck_0082.h5"
        import tensorflow as tf
        import Mult_Class_Brachy
        import mrcnn.model as modellib
        config = Mult_Class_Brachy.NeckConfig()
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        weights_path = MASKRCNN_MODEL_WEIGHT_FILEPATH
        # Load weights
        print('model.load_weight start')
        model.load_weights(weights_path, by_name=True)
        print('model.load_weight end')
        # Test
        model.detect([np.zeros([512, 512, 3])])
        return model
    dataset = get_dataset()
    model = get_model()
    label_id_mask = get_label_id_mask(dataset, model, filelist)
    return label_id_mask
def get_label_id_mask(dataset, model, ct_filelist):
    """
    :param dataset:
    :param model:
    :param ct_filelist:
    :return:
    """
    def to_json(class_idsss, masksss, dataset):
        cn_list = []
        dict_value_newT = []
        for i in range(len(class_idsss)):
            mask = masksss[:, :, i]
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            dict_value = []
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                dict_value.append(verts)
            dict_value_new = [dict_value[0].tolist()]
            dict_value_newT.append(dict_value_new)
            ci = class_idsss[i]
            cn = dataset.class_names[ci]  # ci is id in class_ids, and cn is mean class name. e.g, ci=27 -> cn=GTV-T
            # The id->class_name mapping is set by Multi_Class.py NeckDataset -> load_Neck -> self.add_class()
            # for example, self.add_class("Neck", 27, "GTV-T")
            cn_list.append(cn)
        dict_json = dict(zip(cn_list, dict_value_newT))
        return dict_json

    print("get_label_id_mask is calling")
    label_id_mask = defaultdict(dict)
    for idx,filepath in enumerate(ct_filelist):
        # filepath is a absolute filepath for some ct file
        print(filepath)

        ct_fp = pydicom.read_file(filepath)
        #image = ct_fp.pixel_array
        # Only Case for Breast right now, (but not Brachy)
        rescale_slope = ct_fp.RescaleSlope
        rescale_intercept = ct_fp.RescaleIntercept
        image = ct_fp.pixel_array * rescale_slope + rescale_intercept


        tmp_image = np.zeros([512, 512, 3])
        for ii in range(512):
            for jj in range(512):
                for kk in range(3):
                    tmp_image[ii][jj][kk] = image[ii][jj]
        image = tmp_image
        print('image.dtype = {}'.format(image.dtype))
        print('image.shape = {}'.format(image.shape))
        #plt.imshow(image[:,:,0])
        #plt.show()



        results = model.detect([image])


        print('OUTPUT of model.detect() idx={}'.format(idx))
        r = results[0]
        print("results[0] = {}".format(results[0]))

        mask = to_json(r['class_ids'], r['masks'], dataset)

        x_spacing, y_spacing = float(ct_fp.PixelSpacing[0]), float(ct_fp.PixelSpacing[1])
        origin_x, origin_y, _ = ct_fp.ImagePositionPatient

        for k, v in mask.items():
            pixel_coords = list()
            for x, y in mask[k][0]:
                # pixel_coords.append(float(Decimal(str((x + origin_x - 7) * x_spacing)).quantize(Decimal('0.00'))))
                # pixel_coords.append(float(Decimal(str((y + origin_y - 7) * y_spacing)).quantize(Decimal('0.00'))))
                tmpX = x * x_spacing + origin_x
                tmpY = y * y_spacing + origin_y
                theX = float(Decimal(str(tmpX)).quantize(Decimal('0.00')))  # Some format transfer stuff
                theY = float(Decimal(str(tmpY)).quantize(Decimal('0.00')))  # Some format transfer stuff
                pixel_coords.append(theX)
                pixel_coords.append(theY)

                pixel_coords.append(float(Decimal(str(ct_fp.SliceLocation)).quantize(Decimal('0.00'))))
            label_id_mask[k][ct_fp.SOPInstanceUID] = pixel_coords

    return label_id_mask


