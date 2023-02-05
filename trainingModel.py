import math
import pandas as pd
import numpy as np
from scipy import optimize

def calc_vo2_if_bike(row, max_hr, resting_hr, weight) :

    if row['WorkoutType'] == 'Bike':
        percent_vo2 = (row['HeartRateAverage'] - resting_hr) / (max_hr - resting_hr)
        vo2_power = row['PowerAverage'] / percent_vo2
        vo2_estimated = (((vo2_power)/75)*1000)/weight
        return vo2_estimated

def optimize_banister(params):
    data = pd.read_csv('workouts.csv')
    data['day_TSS'] = data['TSS'].groupby(data['WorkoutDay']).transform('sum') #Fill in any missing days with zero
    data['day_TSS'] = data['day_TSS'].fillna(0) #Fill in any missing days with zero
    data['bike_V02'] = data.apply(lambda row: calc_vo2_if_bike(row, 196, 50, 74), axis=1) #Calculate VO2 for each bike workout
    data = data[['WorkoutDay', 'day_TSS', 'bike_V02']] #Keep only the columns we need
    data = data.groupby('WorkoutDay').mean() #Group by day and take the mean of the TSS and VO2
    data['WorkoutDate'] = data.index #Add a column for the date
    data['WorkoutDate'] = pd.to_datetime(data['WorkoutDate']) #Convert the date column to a datetime object
    data = data.sort_values(by=['WorkoutDate']) #Sort the data by date
    data.index = pd.DatetimeIndex(data['WorkoutDate']) #Set the index to the date
    missing_dates = pd.date_range(start=data.index.min(), end=data.index.max()) #Create a list of all the dates in the range
    data = data.reindex(missing_dates, fill_value=0) #Add missing dates
    data['bike_V02'] = data['bike_V02'].fillna(method="ffill") #FiLL missing VOZ values with previous value
    data = data.dropna() #Drop any remaining rows with missing values TSS= data['day_TSS'].to_list() #Convert the TSS column to a list
    TSS = data['day_TSS'].to_list() #Convert the TSS column to a list
    Performance = data['bike_V02'].to_list() #Convert the V02 column to a list
    losses = [] #Create an empty list to store the loss values from our model
    ctls = [0] #Create an empty list to store the CTL values from our model with starting CTL of 0
    atls = [0] #Create an empty list to store the ATL values from our model with starting ATL of 0 for i in range(len(TSS)):
    print(params[3], params[4])
    for i in range(len(TSS)):
        ctl = (TSS[i] * (1-math.exp(-1/params[3]))) + (ctls[i] * (math.exp(-1/params[3]))) #Calculate the CTL for the day
        atl = (TSS[i] * (1-math.exp(-1/params[4]))) + (atls[i] * (math.exp(-1/params[4]))) #Calculate the ATL for the day
        ctls.append(ctl) #Add the CTL to the list
        atls.append(atl) #Add the ATL to the List
        Banister_Prediction = params[2] + params[0]*ctl - params[1]*atl #Calculate the Banister Prediction for the day
        loss = abs(Performance[i]- Banister_Prediction)
        losses.append(loss) #Add the loss to the list
    # print(f"CTLs: {ctls}") 
    # print(f"ATLS: {atls}")
    MAE= np.mean(losses) #Calculate the mean absolute error
    # print(f"MAE: {MAE}")
    return MAE

initial_guess = [0.1, 0.5, 50, 40, 15]
individual_banister_model = optimize.minimize(optimize_banister,  x0 = initial_guess, bounds=[(0,1),(0,1),(20,50),(20,60),(10,20)])
print(individual_banister_model)