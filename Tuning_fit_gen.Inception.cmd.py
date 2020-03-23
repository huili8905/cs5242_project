from __future__ import print_function
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Dropout, Activation
# from tensorflow.keras.models import Sequential

from keras import optimizers
# from keras.layers.core import Dense, Dropout, Activation
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D, Input
# from keras.models import Sequential
from keras.utils import np_utils
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Flatten
# from keras.layers.normalization import BatchNormalization

# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet_v2 import ResNet50V2

# from tfkerassurgeon import delete_layer, insert_layer
# from kerassurgeon.operations import delete_layer, insert_layer, delete_channels

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten
    
import os
from datetime import datetime
from math import sqrt


def data():
    
    def image_generator(split_file_to_load, mode="train", aug=None):
        # import main
        from read_datasetBreakfast import load_data, read_mapping_dict
        from matplotlib import pyplot as plt
        import cv2


        COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
        loadfile_output_file = os.path.join(COMP_PATH, 'loadfile_output.txt')

        split = 'training'
        # split = 'test'

        # train_split =  os.path.join(COMP_PATH, 'splits/Copy of train.split1 - P16_48.bundle') #Train Split
        # valid_split =  os.path.join(COMP_PATH, 'splits/Copy of valid.split1 - P49_54.bundle') #Valid Split
        # train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle') #Train Split
        # valid_split =  os.path.join(COMP_PATH, 'splits/valid.split1.bundle') #Valid Split
        # test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
        file_split  =  os.path.join(COMP_PATH, split_file_to_load) #Split
        GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video 
        DATA_folder =  os.path.join(COMP_PATH, 'data/') #Frame I3D features for all videos
        mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt') 

        actions_dict = read_mapping_dict(mapping_loc)
        startTime = datetime.now()
        # if split == 'training':
        #     data_feat, data_labels = load_data( file_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
        #     valid_feat, valid_labels = load_data( valid_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
        # if  split == 'test':
        #     data_feat = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features only
        data_feat, data_labels = load_data( file_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
        endTime = datetime.now()
        
        log_f = open( loadfile_output_file, 'w' )
        log_f.write("Diff Time:"+str(endTime - startTime))
        log_f.close()
        print("===== Tuning_fit_gen.cmd.py =====")
        print("Diff Time:"+str(endTime - startTime))
        print("===== Tuning_fit_gen.cmd.py =====")
        
        X_data_tmp = [data_feat[i].cpu().detach().numpy() for i in range(len(data_feat))]
        X_data = np.array([np.reshape(arr, (arr.shape[0], 20, 20, 1)) for arr in X_data_tmp])
        X_data = X_data / 255.0 # preprocess the data

        # # X_data = []
        # # for i, arr in enumerate(X_data_tmp):
        # #     X_data.append(np.reshape(arr, (arr.shape[0], 20, 20, 1))/255.0)
        Y_data = np.array([np_utils.to_categorical(np.array(data_labels[i]), num_classes=48) for i in range(len(data_labels))])
        
        # print(type(X_data))
        # print(type(Y_data))

        # file_ptr = open(file_split, 'r')
        # content_all = file_ptr.read().split('\n')[1:-1]
        # content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
        
        # max_num_files = len(content_all)
        # print("file_split:", file_split)
        # print("max_num_files:", max_num_files)

        print("===== Tuning_fit_gen.cmd.py =====")
        # print("Time: "+str(datetime.now())+"; mode: "+mode+"; max_num_files:", max_num_files)
        print("Time: "+str(datetime.now())+"; mode: "+mode+"; len(X_data):", len(X_data))
        print("===== Tuning_fit_gen.cmd.py =====")
        # max_load=250
        # curr_start=0
        # curr_end=max_load
        # while True:
            
        #     startTime = datetime.now()
        #     data_feat, data_labels = load_data_range( file_split, actions_dict, GT_folder, DATA_folder, 
        #                                             datatype = split, start=curr_start, end=curr_end) #Get features and labels
        #     endTime = datetime.now()
            
        #     log_f = open( loadfile_output_file, 'w' )
        #     log_f.write("Diff Time:"+str(endTime - startTime))
        #     log_f.close()
        #     print("===== Tuning_fit_gen.cmd.py =====")
        #     print("Diff Time:"+str(endTime - startTime))
        #     print("===== Tuning_fit_gen.cmd.py =====")
            
        #     X_data_tmp = [data_feat[i].cpu().detach().numpy() for i in range(len(data_feat))]
        #     X_data = np.array([np.reshape(arr, (arr.shape[0], 20, 20, 1)) for arr in X_data_tmp])
        #     X_data = X_data / 255.0 # preprocess the data
    
        #     Y_data = np.array([np_utils.to_categorical(np.array(data_labels[i]), num_classes=48) for i in range(len(data_labels))])

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
                
            # # Now, calculate the next batch of the files to load
            # curr_start += max_load
            # curr_end += max_load
            
            # if curr_start > max_num_files:
            #     # Reset to the beginning
            #     curr_start = 0
            #     curr_end = max_load
                
            # if curr_end >= max_num_files:
            #     curr_end = max_num_files
                

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
    print("returning generators")
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

    def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
        
        conv_1x1 = Conv2D(filters_1x1, (1,1), strides=(1, 1), padding='same', activation='relu')(x)
        
        conv_3x3 = Conv2D(filters_3x3_reduce, (1,1), strides=(1, 1), padding='same', activation='relu')(x)
        conv_3x3 = Conv2D(filters_3x3, (3,3), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
        
        conv_5x5 = Conv2D(filters_5x5_reduce, (1,1), strides=(1, 1), padding='same', activation='relu')(x)
        conv_5x5 = Conv2D(filters_5x5, (5,5), strides=(1, 1), padding='same', activation='relu')(conv_5x5)
        
        pool_proj = MaxPool2D((2,2), strides=(1,1), padding='same')(x)
        pool_proj = Conv2D(filters_pool_proj, (1,1), strides=(1, 1), padding='same', activation='relu')(pool_proj)
        
        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
        
        return output
        
        
        
    COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
    traindata_output_file = os.path.join(COMP_PATH, 'traindata_output.txt')

    input_layer = Input(shape=(20, 20, 1))
    
    x = Conv2D(16, (3,3), strides=(1,1), padding='same', activation='relu', name='conv_1a_3x3')(input_layer)
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', name='conv_1b_3x3')(input_layer)
    x = MaxPool2D((2,2), strides=(2,2), padding='same', name='max_pool_1_2x2')(x)
    x = inception_module(x, 
                        filters_1x1=32,
                        filters_3x3_reduce=32,
                        filters_3x3=48,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_2')
    x = MaxPool2D((2,2), strides=(2,2), padding='same', name='max_pool_2_2x2')(x)
    x = inception_module(x, 
                        filters_1x1=48,
                        filters_3x3_reduce=48,
                        filters_3x3=64,
                        filters_5x5_reduce=24,
                        filters_5x5=48,
                        filters_pool_proj=48,
                        name='inception_3')
    x = AveragePooling2D((2,2), strides=(2,2), padding='valid', name='avg_pooling_3_2x2')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(48, activation='softmax', name='output')(x)
    
    model = Model(input_layer, x, name='inception_v1')
    
    return model

if __name__ == '__main__':

    COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
    tuning_output_file = os.path.join(COMP_PATH, 'tuning_output.txt')

    startTime = datetime.now()

    print("===== Tuning_fit_gen.cmd.py =====")
    print("Calling data() to get generators")
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
    print("Starting fit_generator")
    print("===== Tuning_fit_gen.cmd.py =====")
    model.fit_generator(train_generator, 
                        steps_per_epoch=64,
                        epochs=250,
                        # epochs={{choice([10, 20, 25, 50, 75])}},
                        validation_data=validation_generator, 
                        validation_steps=64)
    
    print("===== Tuning_fit_gen.cmd.py =====")
    print("Starting evaluate_generator")
    print("===== Tuning_fit_gen.cmd.py =====")
    print(model.evaluate_generator(train_generator, steps=32))
    print(model.metrics_names)

    model.save('Tuning.h5')

    endTime = datetime.now()

    log_f = open( tuning_output_file, 'w' )
    log_f.write("Diff Time:"+str(endTime - startTime)+"\n")
    log_f.close()
    print("Diff Time:",(endTime - startTime))


