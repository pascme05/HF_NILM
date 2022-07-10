# Introduction
HFNILM is a tool for solving the energy dissagregation problem using high-frequency (HF) data. 
It aims to provide a baseline systems for both new and experienced researchers within 
the area of energy disaggregation and Non-Intrusive Load Monitoring using HF data.

# Publication
The HFNILM toolkit is part of the following NILM paper and tries to Please cite the following paper when using 
the HFNILM toolkit:

Pascal A. Schirmer and Iosif Mporas: High-Frequency Energy Disaggregation using Time Series Embedding 

# Datasets
An overview of a few datasets and their locations:

1) REDD:   http://redd.csail.mit.edu/

# Dependencies
The BaseNILM Toolkit was implemented using the following dependencies:
- Python 3.8
- Tensorflow 2.5.0
- Keras 2.4.3

For GPU based calculations CUDA in combination with cuDNN has been used, utilizing the Nvidia RTX 3000 series for
calculation. The following versions have been tested and proven to work with the BaseNILM toolkit:
- CUDA 11.4
- DNN 8.2.4
- Driver 472.39

# Usage
For a first test run use start.py to train, test and plot a 
5-fold cross validation using the REDD-3 HF dataset with all loads. 
If you don't want to train simply set 'setup_Exp['train']=0' as the models 
for the example test run are already stored in BaseNILM \mdl.

# Results
For the setup described in usage the results can be found below. 

	|          |    FINITE STATES   |          POWER ESTIMATION         |   PERCENT OF TOTAL  |
	| item ID  | ACCURACY | F-SCORE | E-ACCURACY |   RMSE   |    MAE    |    EST    |  TRUTH  |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 1        |   98.30% |  98.14% |   59.61%   |   12.91% |    2.40%  |    0.47%  |   0.71% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 2        |  100.00% | 100.00% |   60.08%   |    0.54% |    0.16%  |    0.06%  |   0.05% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 3        |   99.93% |  99.90% |   64.14%   |   11.68% |    0.81%  |    0.07%  |   0.21% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 4        |   99.98% |  99.99% |   90.23%   |   50.62% |   20.34%  |   24.95%  |  27.87% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 5        |   93.54% |  93.51% |   87.71%   |   36.04% |   11.61%  |   11.52%  |  12.79% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 6        |  100.00% | 100.00% |   36.08%   |    0.80% |    0.19%  |    0.06%  |   0.04% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 7        |   98.54% |  98.24% |   52.30%   |   64.90% |    7.10%  |    0.47%  |   2.03% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 8        |   98.76% |  99.04% |   78.89%   |   40.93% |    5.87%  |    3.94%  |   4.26% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 9        |   90.83% |  91.06% |   66.44%   |   45.64% |   18.41%  |    8.53%  |   9.54% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 10       |   95.74% |  94.12% |   45.87%   |   12.05% |    3.80%  |    0.23%  |   0.87% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 11       |   99.88% |  99.88% |   81.26%   |   54.85% |    3.86%  |    7.32%  |   7.76% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 12       |   99.54% |  99.55% |   82.34%   |   62.93% |    5.99%  |    9.53%  |  10.16% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 13       |  100.00% | 100.00% |   59.91%   |    0.70% |    0.23%  |    0.10%  |   0.07% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 14       |   99.75% |  99.72% |   87.49%   |   38.54% |    1.99%  |    2.16%  |   2.20% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 15       |   98.11% |  98.12% |   90.56%   |   16.32% |    6.33%  |    8.33%  |   8.84% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 16       |  100.00% | 100.00% |   87.26%   |    0.55% |    0.27%  |    0.29%  |   0.28% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 17       |   91.27% |  91.47% |   75.91%   |   21.59% |    7.69%  |    3.59%  |   4.13% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 18       |   99.55% |  99.59% |   73.89%   |   63.32% |    6.30%  |    3.98%  |   4.37% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 19       |   99.99% |  99.99% |   20.99%   |    3.91% |    2.58%  |    0.20%  |   0.75% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| 20       |   95.06% |  93.69% |   60.47%   |   68.71% |   10.88%  |    1.20%  |   3.06% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	|    AVG   |   97.94% |  97.80% |   85.04%   |  190.70% |    5.84%  |   87.01%  | 100.00% |

# Development
As failure and mistakes are inextricably linked to human nature, the toolkit is obviously not perfect, 
thus suggestions and constructive feedback are always welcome. If you want to contribute to the HFNILM 
toolkit or spotted any mistake, please contact me via: p.schirmer@herts.ac.uk

# License
The software framework is provided under the MIT license.

# Notes
Please note that the resolution of the provided data for REDD-3 HF has been limited to one decimal point
in order to reduce the files size for upload. However, the error is neglectable for many experiments.