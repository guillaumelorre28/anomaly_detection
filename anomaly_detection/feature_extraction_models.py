import tensorflow as tf
from tensorflow import keras

layers = tf.keras.layers


def create_multiscale_model(backbone='resnet50', image_size=224):

    input = tf.keras.Input(shape=(image_size, image_size, 3))

    if backbone == 'resnet50':
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input,
                                               input_shape=(image_size, image_size, 3), pooling=None)
        layer_names = ['conv2_block3_out', 'conv3_block3_out', 'conv4_block3_out']

    outputs = []
    for name in layer_names:
        outputs.append(layers.experimental.preprocessing.Resizing(56, 56)(model.get_layer(name).output))

    final_output = layers.Concatenate(axis=-1)(outputs)

    multiscale_model = keras.Model(inputs=model.input, outputs=final_output)

    return multiscale_model


def create_model(backbone='resnet50', image_size=224):

    if backbone == 'resnet50':
        return tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                               input_shape=(image_size, image_size, 3), pooling=None)

    else:
        return None
