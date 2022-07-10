#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainCNN
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
import tensorflow as tf
import keras.backend as k
import os
from efficientnet.tfkeras import EfficientNetB0

#######################################################################################################################
# GPU Settings
#######################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#######################################################################################################################
# Additional function definitions
#######################################################################################################################
def lossMetric(y_true, y_pred):
    return 1 - k.sum(k.abs(y_pred - y_true)) / (k.sum(y_true) + k.epsilon()) / 2


#######################################################################################################################
# Models
#######################################################################################################################
def createCNNmdl(X_train, outputdim):
    mdl = tf.keras.models.Sequential()

    mdl.add(tf.keras.layers.BatchNormalization(input_shape=X_train.shape[-3:]))
    mdl.add(tf.keras.layers.Conv2D(filters=30, kernel_size=(1, 10), activation='relu', padding="same", strides=(1, 1)))
    mdl.add(tf.keras.layers.BatchNormalization())
    mdl.add(tf.keras.layers.Conv2D(filters=30, kernel_size=(1, 8), activation='relu', padding="same", strides=(1, 1)))
    mdl.add(tf.keras.layers.BatchNormalization())
    mdl.add(tf.keras.layers.Conv2D(filters=40, kernel_size=(1, 6), activation='relu', padding="same", strides=(1, 1)))
    mdl.add(tf.keras.layers.BatchNormalization())
    mdl.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(1, 5), activation='relu', padding="same", strides=(1, 1)))
    mdl.add(tf.keras.layers.BatchNormalization())
    mdl.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(1, 5), activation='relu', padding="same", strides=(1, 1)))
    mdl.add(tf.keras.layers.BatchNormalization())

    mdl.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(512, activation='relu'))
    mdl.add(tf.keras.layers.Dense(512, activation='relu'))
    mdl.add(tf.keras.layers.Dense(512, activation='relu'))

    mdl.add(tf.keras.layers.Dense(outputdim, activation='linear'))
    mdl.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=[lossMetric])

    return mdl


def createPreCNNmdl(X_train, outputdim):
    effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(X_train.shape[1], X_train.shape[2], 3))

    mdl = tf.keras.models.Sequential()
    mdl.add(effnet)
    mdl.add(tf.keras.layers.GlobalAveragePooling2D())
    mdl.add(tf.keras.layers.Dropout(0.5))
    mdl.add(tf.keras.layers.BatchNormalization())
    mdl.add(tf.keras.layers.Dense(outputdim, activation="linear"))

    mdl.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=[lossMetric])

    return mdl


#######################################################################################################################
# Function
#######################################################################################################################
def trainCNN(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    BATCH_SIZE = setup_Mdl['batch_size']
    BUFFER_SIZE = XTrain.shape[0]
    EVALUATION_INTERVAL = int(np.floor(BUFFER_SIZE/BATCH_SIZE))
    EPOCHS = setup_Mdl['epochs']
    VALSTEPS = setup_Mdl['valsteps']
    VERBOSE = setup_Mdl['verbose']
    SHUFFLE = setup_Mdl['shuffle']
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=setup_Mdl['patience'])

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    if len(XTrain.shape) == 2:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], 1, 1))
        XVal = XVal.reshape((XVal.shape[0], XVal.shape[1], 1, 1))
    elif len(XTrain.shape) == 3:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], XTrain.shape[2], 1))
        XVal = XVal.reshape((XVal.shape[0], XVal.shape[1], XVal.shape[2], 1))
    else:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], XTrain.shape[2], XTrain.shape[3]))
        XVal = XVal.reshape((XVal.shape[0], XVal.shape[1], XVal.shape[2], XVal.shape[3]))

    # ------------------------------------------
    # Build CNN Model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        outputdim = 1
    else:
        outputdim = setup_Data['numApp']
    mdl = createCNNmdl(XTrain, outputdim)
    # mdl.summary()

    # ------------------------------------------
    # Save initial weights
    # ------------------------------------------
    os.chdir(mdlPath)
    mdl.save_weights('initMdl.h5')
    os.chdir(path)

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.h5'
            try:
                mdl.load_weights(mdlName)
                print("Running NILM tool: Model exist and will be retrained!")
            except:
                mdl.save_weights(mdlName)
                print("Running NILM tool: Model does not exist and will be created!")
            os.chdir(path)

            # Create Data
            train_data = tf.data.Dataset.from_tensor_slices((XTrain, np.squeeze(YTrain[:, i])))
            train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
            val_data = tf.data.Dataset.from_tensor_slices((XVal, np.squeeze(YVal[:, i])))
            val_data = val_data.batch(BATCH_SIZE).repeat()

            # Train
            mdl.fit(train_data, epochs=EPOCHS,
                    steps_per_epoch=EVALUATION_INTERVAL,
                    validation_data=val_data,
                    validation_steps=VALSTEPS,
                    use_multiprocessing=True,
                    verbose=VERBOSE,
                    shuffle=SHUFFLE,
                    callbacks=[stop_early])

            # Save model
            os.chdir(mdlPath)
            mdl.save_weights(mdlName)
            os.chdir(path)

    elif setup_Para['multiClass'] == 1:
        # Load Model
        os.chdir(mdlPath)
        mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.h5'
        try:
            mdl.load_weights(mdlName)
            print("Running NILM tool: Model exist and will be retrained!")
        except:
            mdl.save_weights(mdlName)
            print("Running NILM tool: Model does not exist and will be created!")
        os.chdir(path)

        # Create Data
        train_data = tf.data.Dataset.from_tensor_slices((XTrain, YTrain))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((XVal, YVal))
        val_data = val_data.batch(BATCH_SIZE).repeat()

        # Train
        mdl.fit(train_data, epochs=EPOCHS,
                steps_per_epoch=EVALUATION_INTERVAL,
                validation_data=val_data,
                validation_steps=VALSTEPS,
                use_multiprocessing=True,
                verbose=VERBOSE,
                shuffle=SHUFFLE,
                callbacks=[stop_early])

        # Save model
        os.chdir(mdlPath)
        mdl.save_weights(mdlName)
        os.chdir(path)
