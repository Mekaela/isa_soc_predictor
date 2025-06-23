# isa_soc_predictor

## Use
To run this predictor on your computer, clone this repository and run main.py. Open the local environment Flask provides, input the fields and get an estimate. 
To train it on your own data, use regression_model.py with your data (make sure it has the required fields). Note there is no imputer, so add it to the pipeline if your data needs it. After running this file, it will replace soc_classifier_model.pkl and you can run main.py with your data. 

## Project category
Tabular data, with user input

## Problem Statement
Unsustainable farming practices and mismanagement of forestry and urban land has resulted in soil with low carbon levels, presenting an opportunity to find practices to improve this soil. Climate change has exemplified these issues and made management changes increasingly important, and so landowners are increasingly under pressure to change their management practices. Increasing soil carbon in all types of agriculture, forestry, and urban land holds potential for sequestering large amounts of carbon dioxide. This tool will allow landowners and policymakers to estimate the carbon levels of the land they manage by inputting the management techniques and site details.

## Challenges
While the dataset chosen has a large amount of data, there are a large amount of records which do not have data on soil organic carbon, so there may not be enough data from the dataset. If more data cannot be found, this might impact on the accuracy of the predictions. 
The dataset covers 42 countries, excluding Portugal, so the predictions may be less accurate in the countries without data.

## Dataset
The SoilHealthDB created by Jian et. al (2020), which provides data collated from studies across 354 sites across 42 countries.
This dataset includes several different types of soil organic carbon measurements. https://github.com/jinshijian/SoilHealthDB/tree/master/data

The other option, that will be used if it is more suitable, is using a pre-trained model, such as DL-SOC: https://github.com/PEESEgroup/DL-SOC

## Method or Algorithm
Several methods will be tried and the most accurate will be used with user input with the interface. The methods tried will be random forest, linear regression, and potentially deep learning. 
The pre-trained model DL-SOC uses deep learning.

## Evaluation
Results will be plotted for visual analysis, and statistical measures will be calculated. Mean absolute percentage error will be calculated to display the average error to the user with the result.
