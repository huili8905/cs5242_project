from __future__ import print_function
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Dropout, Activation
# from tensorflow.keras.models import Sequential

from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

import os
from datetime import datetime
from math import sqrt


def data():
    
    def image_generator(split_file_to_load, mode="train"):
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
        X_data = []
        for i, arr in enumerate(X_data_tmp):
            X_data.append(np.reshape(arr, (arr.shape[0], 20, 20, 1)))
        Y_data = [np_utils.to_categorical(np.array(data_labels[i]), num_classes=48) for i in range(len(data_labels))]

        print("===== Tuning_fit_gen.cmd.py =====")
        print("Time: "+str(datetime.now())+"; mode: "+mode+"; len(X_data): "+str(len(X_data)))
        print("===== Tuning_fit_gen.cmd.py =====")
        idx = 0
        while True:
            if idx >= len(X_data):
                idx = 0
                # if mode == "eval":
                #     # stop if we are doing evaluation
                #     break
            # print("Time: "+str(datetime.now())+"; mode: "+mode+"; X_data["+str(idx)+"] len: "+str(len(X_data[idx]))+"\n")
            yield (X_data[idx], Y_data[idx])
            idx += 1

    # train_split =  'splits/train.split1 - P16_48.bundle' #Train Split
    # valid_split =  'splits/valid.split1 - P49_54.bundle' #Valid Split
    # train_split =  'splits/900 train.split1 - P16_48.bundle' #Train Split
    # valid_split =  'splits/150 valid.split1 - P49_54.bundle' #Valid Split
    # train_split =  'splits/300 train.split1 - P16_48.bundle' #Train Split
    # valid_split =  'splits/90 valid.split1 - P49_54.bundle' #Valid Split
    train_split =  'splits/2 train.split1.bundle' # Short only 2 files Train Split, for code testing
    valid_split =  'splits/2 valid.split1.bundle' # Short only 2 files Valid Split, for code testing

    train_generator = image_generator(train_split, "train")
    validation_generator = image_generator(valid_split, "eval")

    print("===== Tuning_fit_gen.cmd.py =====")
    print("returning generators")
    print("===== Tuning_fit_gen.cmd.py =====")
    return train_generator, validation_generator


def create_model(train_generator, validation_generator):
    
    COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
    traindata_output_file = os.path.join(COMP_PATH, 'traindata_output.txt')

    ks1_first = {{choice([2, 3])}} # Conv2D 1 kernel size
    ks1_second = {{choice([2, 3])}}

    ss1_first = {{choice([1, 2])}} # Conv2D 1 stride step
    ss1_second = {{choice([1, 2])}}
    
    ps1_first = {{choice([1, 2])}} # layer 1 pool stride
    ps1_second = {{choice([1, 2])}}

    ss2_first = {{choice([1, 2])}} # Conv2D 2 stride step
    ss2_second = {{choice([1, 2])}}
    
    ps2_first = {{choice([1, 2])}} # Where ss stands for stride_step
    ps2_second = {{choice([1, 2])}}

    f1 = {{quniform(4, 16, 4)}}  # quniform(min, max, q) means uniform(min, max) with step size q
    f2 = {{quniform(20, 32, 4)}}  # quniform(min, max, q) means uniform(min, max) with step size q

    fc1 = {{quniform(64, 128, 8)}}  # quniform(min, max, q) means uniform(min, max) with step size q
    # fc2 = {{quniform(128, 168, 8)}}  # quniform(min, max, q) means uniform(min, max) with step size q

    d1 = {{uniform(0.1, 0.3)}}
    d2 = {{uniform(0.3, 0.8)}}

    model = Sequential()
    
    model.add(Conv2D(filters=16,
                     kernel_size=(4, 4),
                     strides=(1, 1),
                     input_shape=(20,20,1), 
                     activation='relu',
                     padding='valid'))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Dropout(int(d1)))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1), # 1, 2
                     activation='relu',
                     padding='valid'))
    # model.add(Dropout(int(d1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.1))

    # model.add(Conv2D(filters=int(f2), 
    #                  kernel_size=(ks2_first, ks2_second),
    #                  strides=(ss3_first, ss3_second),
    #                  activation='relu',
    #                  padding='same''))
    # # model.add(Dropout(int(d2)))
    # model.add(Conv2D(filters=int(f2), 
    #                  kernel_size=(ks2_first, ks2_second),
    #                  strides=(ss3_first, ss3_second),
    #                  activation='relu',
    #                  padding='same'))
    # # model.add(Dropout(int(d2)))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(ps2_first, ps2_second)))
    # model.add(Dropout(int(d2)))

    # if conditional({{choice(['two', 'three'])}}) == 'three':
    # if {{choice(['two', 'three'])}} == 'three':
    #     model.add(Conv2D(filters=int(f3), 
    #                     kernel_size=(2, 2),
    #                     strides=(1, 1),
    #                     activation='relu',
    #                     padding='valid',
    #                     kernel_initializer='TruncatedNormal'))
    #     model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    #     model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(108))
    # model.add(Activation='relu')
    # model.add(Dense(108), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(48))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    rmsprop = optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    # sgd = optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
  
    choiceval = {{choice(['adam', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    else:
        optim = rmsprop
    # elif choiceval == 'rmsprop':
    #     optim = rmsprop
    # else:
    #     optim = sgd
        
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)

    # nb_train_samples = 1855
    # nb_validation_samples = 745
    # nb_test_samples = 2750

    # epochs=20
    # batch_size=16

    print("===== Tuning_fit_gen.cmd.py =====")
    print("Starting fit_generator")
    print("===== Tuning_fit_gen.cmd.py =====")
    model.fit_generator(train_generator, 
                        steps_per_epoch={{choice([32,64])}},
                        epochs={{choice([10, 20, 25])}},
                        # epochs={{choice([10, 20, 25, 50, 75])}},
                        validation_data=validation_generator, 
                        validation_steps=32)
    
    print("===== Tuning_fit_gen.cmd.py =====")
    print("Starting evaluate_generator")
    print("===== Tuning_fit_gen.cmd.py =====")
    score, acc = model.evaluate_generator(generator=validation_generator, steps=32)

    # model.fit_generator(train_generator, 
    #                     steps_per_epoch=nb_train_samples // batch_size, 
    #                     epochs=epochs, 
    #                     validation_data=validation_generator, 
    #                     validation_steps=nb_validation_samples // batch_size)

    # score, acc = model.evaluate_generator(generator=validation_generator, 
    #                                       steps=nb_validation_samples // batch_size)

    # score, acc = model.evaluate(X_test[0], Y_test[0], verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    COMP_PATH = "/content/drive/My Drive/Colab Notebooks/NUS/CS5242 Neural Networks and Deep Learning/Project"
    tuning_output_file = os.path.join(COMP_PATH, 'tuning_output.txt')

    startTime = datetime.now()

    print("===== Tuning_fit_gen.cmd.py =====")
    print("Starting optim.minimize")
    print("===== Tuning_fit_gen.cmd.py =====")
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          notebook_name='Tuning',
                                          trials=Trials())
    
    print("===== Tuning_fit_gen.cmd.py =====")
    print("Calling data() to get generators")
    print("===== Tuning_fit_gen.cmd.py =====")
    train_generator, validation_generator = data()

    # X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate_generator(validation_generator, steps=32))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model.metrics_names)

    best_model.save('Tuning.h5')

    endTime = datetime.now()

    log_f = open( tuning_output_file, 'w' )
    log_f.write("Diff Time:"+str(endTime - startTime)+"\n")
    log_f.write("Best performing model chosen hyper-parameters:\n")
    log_f.write(str(best_run))
    log_f.close()
    print("Diff Time:",(endTime - startTime))


