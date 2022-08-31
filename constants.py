import cv2
import tensorflow as tf



# class Effusion_constants:
#
#     data_dir = r"/home/hungrycucumber/Documents/HCL/train"
#     test_img = cv2.imread("/home/hungrycucumber/PycharmProjects/pneumonia_classification/dataset/chest_data/train/Effusion")
#     img_size = (256,256)
#     data = tf.keras.utils.image_dataset_from_directory(data_dir)
#
#
# class Edema_constants:
#
#
#     data_dir = r"C:\\Users\\tyagi\HCL PROJ\edema\train"
#     test_img = cv2.imread("C:\\Users\\tyagi\HCL PROJ\edema\train\\Normal\IM-0111-0001.jpeg")
#     img_size = (256, 256)
#     data = tf.keras.utils.image_dataset_from_directory(data_dir)
#
#
#
# class Pneumonia_constants:
#
#     data_dir = r"/home/hungrycucumber/Documents/HCL"
#     test_img = cv2.imread("/home/hungrycucumber/PycharmProjects/pneumonia_classification/dataset/chest_data/train/Effusion")
#     img_size = (256, 256)
#     data = tf.keras.utils.image_dataset_from_directory(data_dir)

class Models:

    effusion_model = r"C:\\Users\\tyagi\Downloads\HCL\models\models\imageclassifier.h5"
    edema_model = r"C:\\Users\\tyagi\Downloads\Disease_Classification\models\edema_classifier.h5"

class Streamlit_var:

    file_name = "Please upload an x-ray  file"
    uplaod_file = "Please upload an X-ray file"
    title = "Disease Classification"
    selectbox = "You selected: "
    spinner = "Model is being loaded.."
    Normal = "Normal"
    Effusion = "Effusion"
    Edema = "Edema"
    Select_disease = "Choose the medical Condition "