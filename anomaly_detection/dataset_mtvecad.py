import tensorflow as tf
import os
import matplotlib.pyplot as plt

RESHAPE_SIZE = 256
CROP_SIZE = 224


def preprocessing(reshape_size=RESHAPE_SIZE, crop_size=CROP_SIZE):

    def preprocess_fn(image, label):

        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, (reshape_size, reshape_size))
        central_fraction = crop_size / reshape_size
        image = tf.image.central_crop(image, central_fraction=central_fraction)
        image = tf.keras.applications.resnet.preprocess_input(image)

        return image, label

    return preprocess_fn


def create_mtvecad_data(data_path, batch_size):

    images = []
    names = []

    for folder in os.listdir(data_path):

        folder_path = os.path.join(data_path, folder)

        if os.path.isdir(folder_path):

            image_name_list = os.listdir(folder_path)

            for image_name in image_name_list:

                if image_name.endswith('.png'):

                    with open(os.path.join(folder_path, image_name), "rb") as f:
                        image = f.read()

                    images.append(image)

                    names.append(os.path.join(folder, image_name))

    dataset = tf.data.Dataset.from_tensor_slices((images, names))
    preprocess_fn = preprocessing()
    dataset = dataset.map(preprocess_fn).batch(batch_size)

    return dataset


# dataset = create_mtvecad_data(data_path='/media/guillaume/Data/data/mvtec_anomaly_detection/bottle/train')
#
# for ex in dataset.take(5):
#
#     print(ex[0].shape)
#     print(ex[1])
#
#     plt.imshow(ex[0].numpy()/255.)
#     plt.show()