#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: featuresMul
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from numpy import inf
import numpy as np
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField


#######################################################################################################################
# Function
#######################################################################################################################
def featuresMul(data, setup_Feat, setup_Para):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Feature extraction")

    ####################################################################################################################
    # Pre-Processing
    ####################################################################################################################
    feat_vec = []

    ####################################################################################################################
    # Pre-Processing
    ####################################################################################################################
    # ------------------------------------------
    # Frzye Power Theory
    # ------------------------------------------
    if setup_Para['feat_pre'] == 1:
        i_f = np.zeros((data.shape[0], data.shape[1]))
        p = np.zeros((data.shape[0]))
        V_rms = np.zeros((data.shape[0]))
        for i in range(0, data.shape[0]):
            p[i] = np.mean(data[i, :, 0] * data[i, :, 1])
            V_rms[i] = np.sqrt(np.mean(data[i, :, 0] ** 2))
            i_f[i, :] = data[i, :, 1] - (data[i, :, 0] / (V_rms[i] ** 2)) * p[i]
        data[:, :, 1] = i_f

    ####################################################################################################################
    # Features
    ####################################################################################################################
    # ------------------------------------------
    # I
    # ------------------------------------------
    if setup_Feat['I'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], 1))
        feat_vec[:, :, 0] = data[:, :, 1]

    # ------------------------------------------
    # VI-Trajectory
    # ------------------------------------------
    if setup_Feat['VI'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        i_max = 0.01
        v_max = 1
        d_i = i_max / data.shape[1]
        d_v = v_max / data.shape[1]
        for i in range(0, data.shape[0]):
            for ii in range(0, data.shape[1]):
                n_i = np.ceil((np.ceil(data[i, :, 1] / d_i) + data.shape[1])/2)
                n_v = np.ceil((np.ceil(data[i, :, 0] / d_v) + data.shape[1])/2)
                if 0 < n_i[ii] < data.shape[1] and 0 < n_v[ii] < data.shape[1]:
                    feat_vec[i, int(n_i[ii]), int(n_v[ii])] = 1

    # ------------------------------------------
    # REC
    # ------------------------------------------
    if setup_Feat['REC'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = RecurrencePlot()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = transformer.transform(data[i, :, 0].reshape(1, -1))

    if setup_Feat['REC'] == 2:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = RecurrencePlot()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = transformer.transform(data[i, :, 1].reshape(1, -1))

    if setup_Feat['REC'] == 3:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1], 2))
        transformer = RecurrencePlot()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :, 0] = transformer.transform(data[i, :, 0].reshape(1, -1))
            feat_vec[i, :, :, 1] = transformer.transform(data[i, :, 1].reshape(1, -1))

    # ------------------------------------------
    # PQ
    # ------------------------------------------
    if setup_Feat['PQ'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = np.outer(data[i, :, 0], data[i, :, 1])

    # ------------------------------------------
    # GAF
    # ------------------------------------------
    if setup_Feat['GAF'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = GramianAngularField()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = transformer.transform(data[i, :, 0].reshape(1, -1))

    if setup_Feat['GAF'] == 2:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = GramianAngularField()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = transformer.transform(data[i, :, 1].reshape(1, -1))

    if setup_Feat['GAF'] == 3:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1], 2))
        transformer = GramianAngularField()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :, 0] = transformer.transform(data[i, :, 0].reshape(1, -1))
            feat_vec[i, :, :, 1] = transformer.transform(data[i, :, 1].reshape(1, -1))

    # ------------------------------------------
    # MKF
    # ------------------------------------------
    if setup_Feat['MKF'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = MarkovTransitionField()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = transformer.transform(data[i, :, 0].reshape(1, -1))

    if setup_Feat['MKF'] == 2:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = MarkovTransitionField()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = transformer.transform(data[i, :, 1].reshape(1, -1))

    if setup_Feat['MKF'] == 3:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1], 2))
        transformer = MarkovTransitionField()
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :, 0] = transformer.transform(data[i, :, 0].reshape(1, -1))
            feat_vec[i, :, :, 1] = transformer.transform(data[i, :, 1].reshape(1, -1))

    # ------------------------------------------
    # DFIA
    # ------------------------------------------
    if setup_Feat['DFIA'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = np.fft.fftshift(np.abs(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))

    elif setup_Feat['DFIA'] == 2:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = np.fft.fftshift(np.angle(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))

    elif setup_Feat['DFIA'] == 3:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1], 3))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :, 0] = np.outer(data[i, :, 0], data[i, :, 1])
            feat_vec[i, :, :, 1] = np.fft.fftshift(np.abs(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))
            feat_vec[i, :, :, 2] = np.fft.fftshift(np.angle(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    # ------------------------------------------
    # Replacing NaNs and Inf
    # ------------------------------------------
    feat_vec = np.nan_to_num(feat_vec)
    feat_vec[feat_vec == inf] = 0

    return [feat_vec, setup_Feat]
