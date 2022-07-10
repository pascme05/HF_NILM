#######################################################################################################################
#######################################################################################################################
# Title: HF-CNN NILM Architecture
# Topic: High Frequency Non-intrusive load monitoring based on pre-trained models
# File: main
# Date: 10.07.2022
# Author: Dr. Pascal A. Schirmer
# Version: V.1.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.loadData import loadDataLF
from lib.fnc.loadData import loadDataHF
from lib.mdl.trainCNN import trainCNN
from lib.fnc.plotting import plotting
from lib.fnc.printResults import printResults
from lib.postprocessing import postprocessing
from lib.fnc.performanceMeasure import performanceMeasure
from lib.fnc.featuresMul import featuresMul
from lib.preprocessing import preprocessingLF
from lib.preprocessing import preprocessingHF
from lib.mdl.testCNN import testCNN
from lib.fnc.save import save
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def main(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    print("Welcome to the HF-CNN NILM tool!")
    print("Author:     Dr. Pascal Alexander Schirmer")
    print("Copyright:  University of Hertfordshire")
    print("Date:       10.07.2022 \n \n")
    print("Running NILM tool: Conventional Mode")
    print("Algorithm:       " + str(setup_Para['algorithm']))
    print("Classifier:      " + setup_Para['classifier'])
    print("Dataset:         " + setup_Data['dataset'])
    print("House Train:     " + str(setup_Data['houseTrain']))
    print("House Test:      " + str(setup_Data['houseTest']))
    print("House Val:       " + str(setup_Data['houseVal']))
    print("Configuration:   " + setup_Exp['configuration_name'])
    print("Experiment name: " + setup_Exp['experiment_name'])
    print("Plotting:        " + str(setup_Exp['plotting']))
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")

    ####################################################################################################################
    # Training
    ####################################################################################################################
    if setup_Exp['train'] == 1:
        # ------------------------------------------
        # Welcome Message
        # ------------------------------------------
        print("Running NILM tool: Training Model")

        # ------------------------------------------
        # Load data
        # ------------------------------------------
        [dataTrainLF, _, dataValLF, _, _, _] = loadDataLF(setup_Data, dataPath)
        [dataTrainHF, _, dataValHF, _, _, _] = loadDataHF(setup_Data, dataPath)

        # ------------------------------------------
        # Pre-Processing
        # ------------------------------------------
        [_, YTrain, setup_Data] = preprocessingLF(dataTrainLF, setup_Data)
        [_, YVal, setup_Data] = preprocessingLF(dataValLF, setup_Data)
        [XTrain, setup_Data] = preprocessingHF(dataTrainHF, setup_Data, setup_Para)
        [XVal, setup_Data] = preprocessingHF(dataValHF, setup_Data, setup_Para)

        # ------------------------------------------
        # Specific Features
        # ------------------------------------------
        [XTrain, _] = featuresMul(XTrain, setup_Feat, setup_Para)
        [XVal, _] = featuresMul(XVal, setup_Feat, setup_Para)

        # ------------------------------------------
        # Classification or Regression
        # ------------------------------------------
        if setup_Para['algorithm'] == 0:
            YTrain[YTrain < setup_Para['p_Threshold']] = 0
            YTrain[YTrain >= setup_Para['p_Threshold']] = 1
            YVal[YVal < setup_Para['p_Threshold']] = 0
            YVal[YVal >= setup_Para['p_Threshold']] = 1

        # ------------------------------------------
        # Model
        # ------------------------------------------
        if setup_Para['classifier'] == "CNN":
            trainCNN(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    ####################################################################################################################
    # Testing
    ####################################################################################################################
    if setup_Exp['test'] == 1:
        # ------------------------------------------
        # Welcome Message
        # ------------------------------------------
        print("Running NILM tool: Testing Model")

        # ------------------------------------------
        # Init
        # ------------------------------------------
        XPred = []
        YPred = []

        # ------------------------------------------
        # Load data
        # ------------------------------------------
        [_, dataTestLF, _, _, _, _] = loadDataLF(setup_Data, dataPath)
        [_, dataTestHF, _, _, _, _] = loadDataHF(setup_Data, dataPath)

        # ------------------------------------------
        # Pre-Processing
        # ------------------------------------------
        [_, YTest, setup_Data] = preprocessingLF(dataTestLF, setup_Data)
        [XTest, setup_Data] = preprocessingHF(dataTestHF, setup_Data, setup_Para)

        # ------------------------------------------
        # Adding Noise
        # ------------------------------------------
        noise = setup_Data['noise'] / 100 * np.random.normal(0, 1, len(XTest))
        XTest = XTest * (1 + noise)

        # ------------------------------------------
        # Specific Features
        # ------------------------------------------
        [XTest, _] = featuresMul(XTest, setup_Feat, setup_Para)

        # ------------------------------------------
        # Classification or Regression
        # ------------------------------------------
        if setup_Para['algorithm'] == 0:
            YTest[YTest < setup_Para['p_Threshold']] = 0
            YTest[YTest >= setup_Para['p_Threshold']] = 1

        # ------------------------------------------
        # Model
        # ------------------------------------------
        if setup_Para['classifier'] == "CNN":
            [XPred, YPred] = testCNN(XTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

        # ------------------------------------------
        # Post-Processing
        # ------------------------------------------
        [_, YPred, _, YTest, YTestLabel, YPredLabel] = postprocessing(XPred, YPred, XTest, YTest, setup_Para, setup_Data)

        # ------------------------------------------
        # Performance Measurements
        # ------------------------------------------
        [resultsApp, resultsAvg] = performanceMeasure(YTest, YPred, YTestLabel, YPredLabel, setup_Data)

        # ------------------------------------------
        # Plotting
        # ------------------------------------------
        if setup_Exp['plotting'] == 1:
            plotting(YTest, YPred, YTestLabel, YPredLabel, setup_Data)

        # ------------------------------------------
        # Output
        # ------------------------------------------
        printResults(resultsApp, resultsAvg, setup_Data)

        # ------------------------------------------
        # Saving results
        # ------------------------------------------
        if setup_Exp['saveResults'] == 1:
            save(resultsApp, resultsAvg, YTest, YPred, setup_Exp, setup_Data, resultPath)
