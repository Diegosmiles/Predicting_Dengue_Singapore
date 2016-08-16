# Singapore_Dengue
### Predicting Dengue Fever outbreaks in Singapore
#### Machine Learning Project for the Center of Urban Science and Progress at New York University
##### Contributors:
##### - Yuan Lai (https://github.com/ylurban)
##### - Diego Garzòn (https://github.com/Diegosmiles)
##### - Bilguun Turboli (https://github.com/bilguun)
##### - Lucas Chizzali (https://github.com/chizzinho)

#### Files:

This repo includes all relevant files for the Machine Learning Project: Predicting Dengue Fever in Singapore

Four notebooks comprise the total methodology: 


Origin_Destination_Simulation.ipynb - based on population and land use (commercial vs residencial areas) of Singapore, create estimates for commuting population between Singapore areas.


Phase 1.ipynb - Using temperature, humidity and wind speed to verify if there is a correlation with this predictors and number of cases per neighborhood in Singapore 


Phase 2.ipynb - Taking raster images of each variable (density of bus stops, street density, trash bins, park density, population density, water areas, and lot density) as inputs and the observed location of dengue cases as labels, we used a Random Forest classifier to predict if there is going to be a dengue case in each square kilometer of Singapore.


gridsearch.py - Python script for finding the best combination of attributes for the Random Forest algorithm. 

Dengue_Fever_Singapore.pdf - the complete report of the project, including goals, methodology, results, conclusions and social impact


![Alt tag](Prediction.png)
- Visualization of predicted areas prone to dengue fever, by ylurban
