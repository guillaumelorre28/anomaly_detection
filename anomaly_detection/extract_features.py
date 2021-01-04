import tensorflow as tf
from anomaly_detection.dataset_mtvecad import create_mtvecad_data
from anomaly_detection.feature_extraction_models import create_multiscale_model, create_model
import os
import pickle
import numpy as np
import gc


params = {
    'data_path': '/media/guillaume/Data/data/mvtec_anomaly_detection',
    'output_path': '/media/guillaume/Data/data/multiscale_features/resnet50',
    'backbone': 'resnet50',
    'multiscale': True,
    'image_size': 224,
    'batch_size': 32
}

if params['multiscale']:
    model = create_multiscale_model(params['backbone'], params['image_size'])
else:
    model = create_model(params['backbone'], params['image_size'])

for class_folder in os.listdir(params['data_path']):

    if os.path.isdir(os.path.join(params['data_path'], class_folder)):

        print(f"Exctracting features for {class_folder}")

        dataset_train = create_mtvecad_data(data_path=os.path.join(params['data_path'], class_folder, 'train'),
                                            batch_size=params['batch_size'])

        dataset_test = create_mtvecad_data(data_path=os.path.join(params['data_path'], class_folder, 'test'),
                                           batch_size=params['batch_size'])

        train_features = []
        train_names = []

        for ex in dataset_train:

            features = model(ex[0])

            train_features.append(features.numpy())
            train_names.append(ex[1].numpy())

        train_features = np.concatenate(train_features, axis=0)

        train_names = np.concatenate(train_names, axis=0)

        train_output = {'names': train_names, 'features': train_features}

        if not os.path.isdir(os.path.join(params['output_path'], class_folder)):
            os.mkdir(os.path.join(params['output_path'], class_folder))

        with open(os.path.join(params['output_path'], class_folder, 'output_train.p'), "wb") as f:

            pickle.dump(train_output, f, protocol=4)

        del train_features
        del train_names

        gc.collect()

        test_features = []
        test_names = []

        for ex in dataset_test:
            features = model(ex[0])

            test_features.append(features.numpy())
            test_names.append(ex[1].numpy())

        test_features = np.concatenate(test_features, axis=0)

        test_names = np.concatenate(test_names, axis=0)

        test_output = {'names': test_names, 'features': test_features}

        with open(os.path.join(params['output_path'], class_folder, 'output_test.p'), "wb") as f:

            pickle.dump(test_output, f, protocol=4)

        del test_features
        del test_names

        del dataset_train
        del dataset_test

        gc.collect()






