import pandas as pd
import numpy as np
import math
import time
import sys
from multiprocessing import Pool

start_time = time.time()
specialNumber = 10
# open training.txt and read into DataFrame
df = pd.read_csv(sys.argv[1], header = None)
# add column names to DataFrame df
df.columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
'Temp3pm', 'RainToday', 'RainTomorrow']
#drop MaxTemp
df = df.drop('MaxTemp', 1)

#transform directions to numerical values
df["WindGustDir"].replace({' N': 1, ' E': 2, ' S': 3, ' W': 4, ' NE': 5, ' NW': 6, ' SE': 7, ' SW': 8, ' NNE': 9, ' ENE': 10, ' NNW': 11, ' WNW': 12, ' SSE': 13, ' ESE': 14, ' SSW': 15, ' WSW': 16}, inplace=True)
df["WindDir9am"].replace({' N': 1, ' E': 2, ' S': 3, ' W': 4, ' NE': 5, ' NW': 6, ' SE': 7, ' SW': 8, ' NNE': 9, ' ENE': 10, ' NNW': 11, ' WNW': 12, ' SSE': 13, ' ESE': 14, ' SSW': 15, ' WSW': 16}, inplace=True)
df["WindDir3pm"].replace({' N': 1, ' E': 2, ' S': 3, ' W': 4, ' NE': 5, ' NW': 6, ' SE': 7, ' SW': 8, ' NNE': 9, ' ENE': 10, ' NNW': 11, ' WNW': 12, ' SSE': 13, ' ESE': 14, ' SSW': 15, ' WSW': 16}, inplace=True)

# get quantile values for a given column
def discretization(df, columnName):
    data = np.array(df[columnName])
    deciles = np.percentile(data, np.arange(0, 100, specialNumber))
    length = len(deciles)
    for index in range(1,length+1):
        if(index == 1):
            df.loc[(df[columnName] < deciles[1]), columnName] = 1
        elif(index == length):
            df.loc[(df[columnName] >= deciles[length - 1]), columnName] = length
        else:
            df.loc[(df[columnName] >= deciles[index - 1]) & (df[columnName] < deciles[index]), columnName] = index

# discretize continous float values
# get percentile values for all columns that are float type and group the values as 1 - 8
for column in df:
    if(df[column].dtype == float):
        discretization(df, column)

# split dataFrame into two dataframes
# yesRainTomorrow is a DataFrame where RainTomorrow only equals Yes
# noRainTomorrow is a DataFrame where RainTomorrow only equals No
yesRainTomorrow = df[df["RainTomorrow"] == " Yes"]
noRainTomorrow = df[df["RainTomorrow"] == " No"]

# open testing.txt and read into DataFrame
testingdf = pd.read_csv(sys.argv[2], header = None)
# add column names to DataFrame testingdf
testingdf.columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
'Temp3pm', 'RainToday', 'RainTomorrow']
#drop MaxTemp
testingdf = testingdf.drop('MaxTemp', 1)

#transform directions to numerical values
testingdf["WindGustDir"].replace({' N': 1, ' E': 2, ' S': 3, ' W': 4, ' NE': 5, ' NW': 6, ' SE': 7, ' SW': 8, ' NNE': 9, ' ENE': 10, ' NNW': 11, ' WNW': 12, ' SSE': 13, ' ESE': 14, ' SSW': 15, ' WSW': 16}, inplace=True)
testingdf["WindDir9am"].replace({' N': 1, ' E': 2, ' S': 3, ' W': 4, ' NE': 5, ' NW': 6, ' SE': 7, ' SW': 8, ' NNE': 9, ' ENE': 10, ' NNW': 11, ' WNW': 12, ' SSE': 13, ' ESE': 14, ' SSW': 15, ' WSW': 16}, inplace=True)
testingdf["WindDir3pm"].replace({' N': 1, ' E': 2, ' S': 3, ' W': 4, ' NE': 5, ' NW': 6, ' SE': 7, ' SW': 8, ' NNE': 9, ' ENE': 10, ' NNW': 11, ' WNW': 12, ' SSE': 13, ' ESE': 14, ' SSW': 15, ' WSW': 16}, inplace=True)

# discretize continous float values
# get percentile values for all columns that are float type and group the values as 1 - 8
for column in testingdf:
    if(testingdf[column].dtype == float):
        discretization(testingdf, column) 

# get bayes probability for a given value in a column given whether it rains tomorrow
def getBayesProbability(value, columnName, rainTomorrow):
    count = rainTomorrow[columnName].value_counts().get(value)
    if(count == None):
        count = 1
    else:
        count = count + 1
    return count/9941

def trainModel(rainTomorrowModel, rainTomorrow):
    for column in range(19):
        valueDict = {}
        if((column == 4) or (column == 6) or (column == 7)):
            for num in range(16):
                valueDict[num + 1] = getBayesProbability(num + 1, rainTomorrow.columns[column + 1], rainTomorrow)
        elif(column == 18):
                valueDict[' Yes'] = getBayesProbability(' Yes', rainTomorrow.columns[column + 1], rainTomorrow)
                valueDict[' No'] = getBayesProbability(' No', rainTomorrow.columns[column + 1], rainTomorrow)
        else:
            for num in range(int(100/specialNumber)):
                valueDict[num + 1] = getBayesProbability(num + 1, rainTomorrow.columns[column + 1], rainTomorrow)
        rainTomorrowModel.append(valueDict)

# Train Models
yesRainTomorrowModel = []
noRainTomorrowModel = []
trainModel(yesRainTomorrowModel, yesRainTomorrow)
trainModel(noRainTomorrowModel, noRainTomorrow)
            
def compute(row,rainTomorrowModel):
    probability = math.log(.5,2)
    for column in range(19):
        if((column != 10)):
            probability = probability - math.log(rainTomorrowModel[column][row[column + 2]],2)
    return probability

# use data to compute probabliities
for row in testingdf.itertuples():
    yesRain = compute(row,yesRainTomorrowModel)
    noRain = compute(row,noRainTomorrowModel)
    if((yesRain < noRain)):
        print(1)
    else:
        print(0)

# count = 0
# total = 0
# pool = Pool()
# for row in testingdf.itertuples():
#     yesRain = compute(row,yesRainTomorrowModel)
#     noRain = compute(row,noRainTomorrowModel)
#     if((yesRain < noRain) and (row[21] == " Yes")):
#         count = count + 1
#     elif((yesRain > noRain) and (row[21] == " No")):
#         count = count + 1
#     total = total + 1
# print(count / total)
# print("--- %s seconds ---" % (time.time() - start_time))
