import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

predictors = np.loadtxt('hourly_wages.csv', delimiter=',',skiprows=1)
n_cols = predictors.shape[1]

# MODEL SETUP

model = Sequential() #'MODEL TYPE: CLASSIC TYPE (SEQUENTIAL)'
model.add(Dense(50, activation='relu', input_shape = (n_cols,))) #'INPUT LAYER 50 NODES, INPUT SHAPE DATA COLUMN VALUES, ACTIVATION: RELU (IF LAYER VALUE < 0 THEN 0 ELSE VALUE)'
model.add(Dense(32,activation='relu')) #'HIDDEN LAYER 32 NODES, ACTIVATION: RELU (IF LAYER VALUE < 0 THEN 0 ELSE VALUE)'
model.add(Dense(1)) #'OUTPUT LAYER WITH ONE NODE'

# COMPILING AND FITTING

model.compile(optimizer = 'adam', loss='mean_squared_error') #learning rate optimization function: ADAM , error function optimization: MSE
model.fit(predictors,target)