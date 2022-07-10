#######################################################################################################################
#######################################################################################################################
# Title: HF-CNN NILM Architecture
# Topic: High Frequency Non-intrusive load monitoring based on pre-trained models
# File: start
# Date: 10.07.2022
# Author: Dr. Pascal A. Schirmer
# Version: V.1.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from os.path import dirname, join as pjoin
import os
from main import main
from kfold import kfold
import warnings

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings('ignore')                                                                                        # suppressing all warning

#######################################################################################################################
# Paths
#######################################################################################################################
basePath = pjoin(dirname(os.getcwd()), 'HF_NILM')
dataPath = pjoin(dirname(os.getcwd()), 'HF_NILM', 'data')
mdlPath = pjoin(dirname(os.getcwd()), 'HF_NILM', 'mdl')
libPath = pjoin(dirname(os.getcwd()), 'HF_NILM', 'lib')
resultPath = pjoin(dirname(os.getcwd()), 'HF_NILM', 'results')

#######################################################################################################################
# Configuration
#######################################################################################################################

# Experiment
setup_Exp = {'experiment_name': "test",                                                                                  # name of the experiment (name of files that will be saved)
             'author': "Pascal",                                                                                         # name of the person running the experiment
             'configuration_name': "hfNILM",                                                                             # name of the experiment configuration
             'train': 0,                                                                                                 # if 1 training will be performed (if 'experiment_name' exist the mdl will be retrained)
             'test': 1,                                                                                                  # if 1 testing will be performed
             'plotting': 0,                                                                                              # if 1 results will be plotted
             'saveResults': 0}                                                                                           # if 1 results will be saved

# Dataset
setup_Data = {'dataset': "redd",                                                                                         # name of the dataset: 1) redd, 2) ampds, 3) refit, 4)...
              'granularity': 1/60,                                                                                       # granularity of the data in Hz
              'downsample': 1,                                                                                           # down-samples the data with an integer value, use 1 for base sample rate
              'limit': 0,                                                                                                # limit number of samples for training, if 0 no limit
              'houseTrain': [1, 3],                                                                                      # houses used for training, e.g. [1, 3, 4, 5, 6]
              'houseTest': 3,                                                                                            # house used for testing, e.g. 2
              'houseVal': 5,                                                                                             # house used for validation, e.g. 2
              'kfold': 5,                                                                                               # number of kfold validations
              'testRatio': 0.1,                                                                                          # if only one house is used 'testRatio' defines the split of 'houseTrain'
              'selApp': [],                                                                                              # appliances to be evaluated (note first appliance is '0')
              'ghost': 0,                                                                                                # if 0) ghost data will not be used, 1) ghost data will be treated as own appliance, 2) ideal data will be used
              'normData': 0,                                                                                             # normalize data, if 0) none, 1) min-max (in this case meanX/meanY are interpreted as max values), 2) min/max one common value (meanX), 3) mean-std, 4) min/max using train-data
              'noise': 0,                                                                                                # adding noise to testing data (%)
              'meanX': 0,                                                                                                # normalization value (mean) for the aggregated signal
              'meanY': [8, 7, 47, 82, 53],                                                                               # normalization values (mean) for the appliance signals
              'stdX': 1,                                                                                                 # normalization value (std) for the aggregated signal
              'stdY': [102, 66, 68, 593, 92],                                                                            # normalization values (std) for the aggregated signals
              'neg': 0,                                                                                                  # if 1 negative data will be removed during pre-processing
              'filt': 1,                                                                                                 # if 0) no filter is used if 1) Output data is filtered if 2) both input and output data is filtered (median filter)
              'filt_len': 21}                                                                                            # length of the filter (must be an odd number)

# Architecture Parameters
setup_Para = {'algorithm': 1,                                                                                            # if 0 classification is used, if 1 regression is used
              'feat_pre': 0,                                                                                             # if 1 current is pre-processed according to Fryze
              'classifier': "CNN",                                                                                       # possible classifier: 1) ML: RF, CNN, LSTM \ 2) PM: DTW, MVM \ 3) SS: NMF, SCA
              'framelength': 55,                                                                                         # frame-length of the time-frames
              'p_Threshold': 50,                                                                                         # threshold for binary distinction of On/Off states
              'multiClass': 1}                                                                                           # if 0 one model per appliance is used, if 1 one model for all appliances is used

# Mdl Parameters
setup_Mdl = {'batch_size': 50,                                                                                           # batch size for DNN based approaches
             'epochs': 50,                                                                                               # number of epochs for training
             'patience': 15,                                                                                             # patience for early stopping
             'valsteps': 50,                                                                                             # number of validation steps
             'shuffle': "True",                                                                                         # either True or False for shuffling data
             'verbose': 2}                                                                                               # settings for displaying mdl progress                                                                                               # scale parameter SVM

#######################################################################################################################
# Select Features
#######################################################################################################################
setup_Feat = {'I': 0,                                                                                                    # if 1) raw current is used
              'PQ': 1,                                                                                                   # if 1) raw pq values are used
              'VI': 0,                                                                                                   # if 1) VI-Trajectory is used
              'REC': 0,                                                                                                  # if 1) Recurrent plot is used
              'GAF': 0,                                                                                                  # if 1) Gramian Angular Field is used
              'MKF': 0,                                                                                                  # if 1) Markov Transition Field is used
              'DFIA': 0}                                                                                                 # if 1) FFT amplitudes, 2) FFT phases (only for shape 3 data with P/Q as features)

#######################################################################################################################
# Non Transfer
#######################################################################################################################
if setup_Data['kfold'] <= 1:
    main(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat)
else:
    kfold(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat)
