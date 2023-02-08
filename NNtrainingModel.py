import math
import pandas as pd
import numpy as np
from scipy import optimize 
from sklearn.neural_network import MLPRegressor


data = pd.read_csv('workouts.csv')
data.describe()
Actual_Performance = data['IF']
Offset_Performance = []
for i in range(len(Actual_Performance)):
    Offset_Performance.append(np.mean (Actual_Performance[i: (i+28)]))
TSS = data['TSS']
TSS.dropna(inplace=True)
Block_TSS = [] 
for i in range(len(TSS)):
    avg_TSS = np.mean(TSS[i:(i+28)])
    Block_TSS.append(avg_TSS)

losses = []
#Individual Banister Model
# def Banister(params):
def Banister(k1, k2, P0, CTLC, ATLC):
    # for i in range(len(TSS)):
    #     fitness = TSS[i] * 1-math.exp(-1/params[3])
    #     fatigue = TSS[i] * 1-math.exp(-1/params[4])
    #     Banister_Prediction = params[0]* fitness - params[1]* fatigue + params[2] 
    #     loss = abs(Actual_Performance[i] - Banister_Prediction) 
    #     losses.append(loss)
    # MAE= np.mean(losses)
    # return MAE
    for i in range(len(TSS)):
        fitness = TSS[i] * 1-math.exp(-1/CTLC)
        fatigue = TSS[i] * 1-math.exp(-1/ATLC)
        Banister_Prediction = k1* fitness - k2* fatigue + P0    
        loss = abs(Actual_Performance[i] - Banister_Prediction) 
        losses.append(loss)
    MAE= np.mean(losses)
    return MAE

initial_guess = [0.1, 0.5, 50, 45, 15]
individual_banister_model = optimize.minimize(Banister, initial_guess)
print(individual_banister_model)

#Individual Neural Network Model
# individual_neural_net_model = MLPRegressor(solver='lbfgs', activation='relu', hidden_layer_sizes=[50], random_state=42)
# individual_neural_net_model.fit(Block_TSS, Offset_Performance)