#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: loadData
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from os.path import join as pjoin
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold


#######################################################################################################################
# Functions
#######################################################################################################################
def loadDataLF(setup_Data, path):
    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses) + 'LF'
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataRaw = dataRaw[name]

    # Limit data
    dataRaw = dataRaw[0:setup_Data['limit'] - 1, :]

    # Split train test val
    dataTrain, dataTest = train_test_split(dataRaw, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)
    _, dataVal = train_test_split(dataTest, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

    # Extract Time
    timeTrain = dataTrain[:, 0]
    timeTest = dataTest[:, 0]
    timeVal = dataVal[:, 0]
    dataTrain = dataTrain[:, 1:]
    dataTest = dataTest[:, 1:]
    dataVal = dataVal[:, 1:]

    # Norm data
    if setup_Data['normData'] == 4:
        normX = np.max(dataTrain[:, 0])
        dataTrain[:, 0] = dataTrain[:, 0] / normX
        dataTest[:, 0] = dataTest[:, 0] / normX
        dataVal[:, 0] = dataVal[:, 0] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal


def loadDataHF(setup_Data, path):
    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses) + 'HF'
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataRaw = dataRaw[name]

    # Norm data
    if setup_Data['normData'] == 5:
        dataRaw[:, :, 0] = dataRaw[:, :, 0] / 180
        dataRaw[:, :, 1] = dataRaw[:, :, 1] / 135

    # Limit data
    dataRaw = dataRaw[0:setup_Data['limit']-1, :, :]

    # Split train test val
    dataTrain, dataTest = train_test_split(dataRaw, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)
    _, dataVal = train_test_split(dataTest, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

    # Extract Time
    timeTrain = dataTrain[:, 0, 0]
    timeTest = dataTest[:, 0, 0]
    timeVal = dataVal[:, 0, 0]
    dataTrain = dataTrain[:, 1:, :]
    dataTest = dataTest[:, 1:, :]
    dataVal = dataVal[:, 1:, :]

    # Norm data
    if setup_Data['normData'] == 4:
        for i in range(0, dataTrain.shape[2]):
            normX = np.max(dataTrain[:, 0, i])
            dataTrain[:, 0, i] = dataTrain[:, 0, i] / normX
            dataTest[:, 0, i] = dataTest[:, 0, i] / normX
            dataVal[:, 0, i] = dataVal[:, 0, i] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal


def loadDataKfoldLF(setup_Data, path):
    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses) + 'LF'
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataRaw = dataRaw[name]

    # Limit data
    dataRaw = dataRaw[0:setup_Data['limit'] - 1, :]

    # Splitting
    kf = KFold(n_splits=setup_Data['kfold'])
    kf.get_n_splits(dataRaw)
    iii = 0
    for train_index, test_index in kf.split(dataRaw):
        iii = iii + 1
        dataTrain, dataTest = dataRaw[train_index], dataRaw[test_index]
        if iii == setup_Data['numkfold']:
            break

    # Get val
    _, dataVal = train_test_split(dataTest, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

    # Extract Time
    timeTrain = dataTrain[:, 0]
    timeTest = dataTest[:, 0]
    timeVal = dataVal[:, 0]
    dataTrain = dataTrain[:, 1:]
    dataTest = dataTest[:, 1:]
    dataVal = dataVal[:, 1:]

    # Norm data
    if setup_Data['normData'] == 4:
        normX = np.max(dataTrain[:, 0])
        dataTrain[:, 0] = dataTrain[:, 0] / normX
        dataTest[:, 0] = dataTest[:, 0] / normX
        dataVal[:, 0] = dataVal[:, 0] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal


def loadDataKfoldHF(setup_Data, path):
    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses) + 'HF'
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataRaw = dataRaw[name]

    # Norm data
    if setup_Data['normData'] == 5:
        dataRaw[:, :, 0] = dataRaw[:, :, 0] / 180
        dataRaw[:, :, 1] = dataRaw[:, :, 1] / 135

    # Limit data
    dataRaw = dataRaw[0:setup_Data['limit']-1, :, :]

    # Splitting
    kf = KFold(n_splits=setup_Data['kfold'])
    kf.get_n_splits(dataRaw)
    iii = 0
    for train_index, test_index in kf.split(dataRaw):
        iii = iii + 1
        dataTrain, dataTest = dataRaw[train_index], dataRaw[test_index]
        if iii == setup_Data['numkfold']:
            break

    # Get val
    _, dataVal = train_test_split(dataTest, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

    # Extract Time
    timeTrain = dataTrain[:, 0, 0]
    timeTest = dataTest[:, 0, 0]
    timeVal = dataVal[:, 0, 0]
    dataTrain = dataTrain[:, 1:, :]
    dataTest = dataTest[:, 1:, :]
    dataVal = dataVal[:, 1:, :]

    # Norm data
    if setup_Data['normData'] == 4:
        for i in range(0, dataTrain.shape[2]):
            normX = np.max(dataTrain[:, 0, i])
            dataTrain[:, 0, i] = dataTrain[:, 0, i] / normX
            dataTest[:, 0, i] = dataTest[:, 0, i] / normX
            dataVal[:, 0, i] = dataVal[:, 0, i] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal
