import numpy as np
import pandas as pd
from datetime import datetime
from create_train_data import train_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)


''' TRAIN PROCESS'''

# Define features and target
X = train_data.drop(columns=['Real (m/s)', 'video_name'])
y = train_data['Real (m/s)']

# Define the model
model = RandomForestRegressor(n_estimators= 200, random_state=42)

print("Start of train process: ", datetime.now())

# Cross Validation 
loo = LeaveOneOut()

y_true= []
y_pred= []

for train_inx, test_inx in loo.split(X):

    X_train = X.iloc[train_inx]
    X_test = X.iloc[test_inx]

    y_train = y.iloc[train_inx]
    y_test = y.iloc[test_inx]

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(prediction[0])

# Evaluation
mae = mean_absolute_error(y_true, y_pred)  

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

r2 = r2_score(y_true, y_pred) 

print(f"Evaluation Metrics\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR2-SCORE: {r2:.3f}")

# Train and predict on whole dataset
model.fit(X, y)
pred = model.predict(X) 

print("End of train process: ", datetime.now())

# Fill the table جدول
results_table = pd.DataFrame({
    "dataset name": train_data['video_name'],
    "real value": y,
    "predicted value": pred,
    "absolute error": abs(y-pred)
})

results_table.to_csv("outputs/جدول.csv")







