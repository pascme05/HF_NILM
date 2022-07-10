#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: preprocessing
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
from numpy import inf
from lib.fnc.normData import normData
import scipy
from scipy import signal


#######################################################################################################################
# Function
#######################################################################################################################
def preprocessingLF(data, setup_Data):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Preprocessing LF data")

    ####################################################################################################################
    # Function
    ####################################################################################################################

    # ------------------------------------------
    # Filtering
    # ------------------------------------------
    if setup_Data['filt'] >= 1:
        for i in range(0, data.shape[1]):
            data[:, i] = scipy.signal.medfilt(data[:, i], kernel_size=setup_Data['filt_len'])

    # ------------------------------------------
    # Down-sample
    # ------------------------------------------
    data = data[::setup_Data['downsample'], :]

    # ------------------------------------------
    # Remove Negative Values
    # ------------------------------------------
    if setup_Data['neg'] == 1:
        data[data < 0] = 0

    # ------------------------------------------
    # Split data X/Y
    # ------------------------------------------
    Xdata = data[:, 0]
    Ydata = data[:, 1:]

    del data

    # ------------------------------------------
    # Select appliances
    # ------------------------------------------
    if len(setup_Data['selApp']) == 0:
        setup_Data['numApp'] = Ydata.shape[1]
    else:
        Ydata = Ydata[:, setup_Data['selApp']]
        setup_Data['numApp'] = len(setup_Data['selApp'])

    # ------------------------------------------
    # Calculate Ghost Power
    # ------------------------------------------
    if setup_Data['ghost'] == 1:
        ghostData = np.zeros((Ydata.shape[0], 1))
        ghostData[:, 0] = Xdata - np.sum(Ydata, axis=1)
        ghostData[ghostData < 0] = 0
        Ydata = np.concatenate((Ydata, ghostData), axis=1)
        setup_Data['numApp'] = setup_Data['numApp'] + 1

        del ghostData

    elif setup_Data['ghost'] == 2:
        ghostData = Xdata - np.sum(Ydata, axis=1)
        ghostData[ghostData < 0] = 0
        Xdata = Xdata - ghostData

        del ghostData

    # ------------------------------------------
    # norm data
    # ------------------------------------------
    if setup_Data['normData'] >= 1:
        [Xdata, Ydata] = normData(Xdata, Ydata, setup_Data)

    # ------------------------------------------
    # Remove NaNs and Inf
    # ------------------------------------------
    Xdata = np.nan_to_num(Xdata)
    Xdata[abs(Xdata) == inf] = 0
    Ydata = np.nan_to_num(Ydata)
    Ydata[abs(Ydata) == inf] = 0

    ####################################################################################################################
    # Output
    ####################################################################################################################

    return [Xdata, Ydata, setup_Data]


def preprocessingHF(data, setup_Data, setup_Para):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Preprocessing HF data")

    ####################################################################################################################
    # Function
    ####################################################################################################################

    # ------------------------------------------
    # Filtering
    # ------------------------------------------
    if setup_Data['filt'] == 2:
        for i in range(0, data.shape[1]):
            for ii in range(0, data.shape[2]):
                data[:, i, ii] = scipy.signal.medfilt(data[:, i, ii], kernel_size=setup_Data['filt_len'])

    # ------------------------------------------
    # Down-sample
    # ------------------------------------------
    data = data[::setup_Data['downsample'], :, :]

    # ------------------------------------------
    # Size Adaption
    # ------------------------------------------
    down = int(np.floor(data.shape[1]/setup_Para['framelength']))
    if isinstance(data.shape[1]/setup_Para['framelength'], int):
        data = data[:, ::down, :]
    else:
        data = signal.resample(data, setup_Para['framelength'], axis=1)

    # ------------------------------------------
    # Remove Negative Values
    # ------------------------------------------
    if setup_Data['neg'] == 1:
        data[data < 0] = 0

    # ------------------------------------------
    # norm data
    # ------------------------------------------
    #if setup_Data['normData'] >= 1:
        #[_, data] = normData(data, data, setup_Data)

    # ------------------------------------------
    # Remove NaNs and Inf
    # ------------------------------------------
    data = np.nan_to_num(data)
    data[abs(data) == inf] = 0

    ####################################################################################################################
    # Output
    ####################################################################################################################

    return [data, setup_Data]
