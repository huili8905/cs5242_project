from __future__ import print_function
import numpy as np

from keras import optimizers

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

import os
from datetime import datetime
from math import sqrt


def data():
    
    def image_generator(split_file_to_load, mode="train", aug=None):
        from read_datasetBreakfast import load_data, read_mapping_dict
        from matplotlib import pyplot as plt
        import cv2

        COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
        loadfile_output_file = os.path.join(COMP_PATH, 'loadfile_output.txt')

        split = 'training'
        # split = 'test'

        file_split  =  os.path.join(COMP_PATH, split_file_to_load) #Split
        GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video 
        DATA_folder =  os.path.join(COMP_PATH, 'data/') #Frame I3D features for all videos
        mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt') 

        actions_dict = read_mapping_dict(mapping_loc)
        
        startTime = datetime.now()
        data_feat, data_labels = load_data( file_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
        endTime = datetime.now()
        
        log_f = open( loadfile_output_file, 'w' )
        log_f.write("******** Time to load the train and test data:"+str(endTime - startTime))
        log_f.close()
        print("===== Tuning_fit_gen.cmd.py =====")
        print("******** Time to load the train and test data:"+str(endTime - startTime))
        print("===== Tuning_fit_gen.cmd.py =====")

        timesteps = 1        
        # X_data shape is (batch/num of files, samples/images/frames per files, features/pixels)
        X_data_tmp = [data_feat[i].cpu().detach().numpy() for i in range(len(data_feat))]
        X_data = [np.reshape(arr, (arr.shape[0], timesteps, arr.shape[1])) for arr in X_data_tmp]
        
        # X_data = X_data / 255.0 # preprocess the data

        Y_data = np.array([np_utils.to_categorical(np.array(data_labels[i]), num_classes=48) for i in range(len(data_labels))])
        
        print("===== Tuning_fit_gen.cmd.py =====")
        print("******** Time: "+str(datetime.now())+"; mode: "+mode+"; len(X_data):", len(X_data))
        print("===== Tuning_fit_gen.cmd.py =====")

        idx = 0
        while True:

            if idx >= len(X_data):
                idx = 0
                # Load the next batch of files
                # break

            # if the data augmentation object is provided, then apply data augmentation before returning
            if aug is not None:
                X_converted = []
                # TODO: apply random data augmentation to the current data
                # X_data[idx], Y_data[idx]
                # https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
                # E.g. aug = ImageDataGenerator() object
                # X_converted = aug.flow(X_data[idx])
        
            yield (X_data[idx], Y_data[idx])
            idx += 1
                

    # train_split =  'splits/train.split1 - P16_48.bundle' #Train Split
    # valid_split =  'splits/valid.split1 - P49_54.bundle' #Valid Split
    train_split =  'splits/900 train.split1 - P16_48.bundle' #Train Split
    valid_split =  'splits/150 valid.split1 - P49_54.bundle' #Valid Split
    # train_split =  'splits/300 train.split1 - P16_48.bundle' # 300 random files Train Split
    # valid_split =  'splits/90 valid.split1 - P49_54.bundle' # 90 random files Valid Split
    # train_split =  'splits/2 train.split1.bundle' # Short only 2 files Train Split, for code testing
    # valid_split =  'splits/2 valid.split1.bundle' # Short only 2 files Valid Split, for code testing

    train_generator = image_generator(train_split, "train")
    validation_generator = image_generator(valid_split, "eval")

    print("===== Tuning_fit_gen.cmd.py =====")
    print("******** Returning from generators")
    print("===== Tuning_fit_gen.cmd.py =====")
    return train_generator, validation_generator


def create_model():
    
#     tf.keras.layers.Conv2D(
#     filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
#     dilation_rate=(1, 1), activation=None, use_bias=True,
#     kernel_initializer='glorot_uniform', bias_initializer='zeros',
#     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None, **kwargs
# )

    n_timesteps = 1
    n_features = 400
    n_outputs = 48

    print("Creating a 400-(200-200)-48 deep LSTM \n")
    model = Sequential()
    model.add(LSTM(400, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    
    return model

if __name__ == '__main__':

    COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
    tuning_output_file = os.path.join(COMP_PATH, 'tuning_LSTM.output.txt')

    startTime = datetime.now()

    print("===== Tuning_fit_gen.cmd.py =====")
    print("******** Calling data() to get generators")
    print("===== Tuning_fit_gen.cmd.py =====")
    train_generator, validation_generator = data()
    
    adam = optimizers.Adam(lr=10**-3)
    # rmsprop = optimizers.RMSprop(lr=10**-3)
    # sgd = optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
    optim = adam
    
    model = create_model()
    
    for i, layer in enumerate(model._layers):
        print(i, layer.name, layer.get_config())

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)

    print("===== Tuning_fit_gen.cmd.py =====")
    print("******** Starting fit_generator")
    print("===== Tuning_fit_gen.cmd.py =====")
    model.fit_generator(train_generator, 
                        steps_per_epoch=64,
                        epochs=250,
                        # epochs={{choice([10, 20, 25, 50, 75])}},
                        validation_data=validation_generator, 
                        validation_steps=64)
    
    print("===== Tuning_fit_gen.cmd.py =====")
    print("******** Starting evaluate_generator")
    print("===== Tuning_fit_gen.cmd.py =====")
    print(model.evaluate_generator(validation_generator, steps=64))
    print(model.metrics_names)

    model.save('Tuning.LSTM.h5')

    endTime = datetime.now()

    log_f = open( tuning_output_file, 'w' )
    log_f.write("******** Total Run Time:"+str(endTime - startTime)+"\n")
    log_f.close()
    print("******** Total Run Time:",(endTime - startTime))


