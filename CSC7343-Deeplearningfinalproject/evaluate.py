
from project2 import Model
import pandas as pd
import numpy as np

data=pd.read_excel('AllData.xlsx')


# Use pandas to convert the excel sheet into a 2D numpy array.
data=pd.DataFrame.to_numpy(data)

# Suppose the array is in a variable named “data”.
err = 0
for i in set(data[:, 0]):
    train = data[data[:, 0] != i]
    test = data[data[:, 0] == i]
    X_train, y_train = train[:, 2:], train[:, 1]
    X_test, y_test = test[:, 2:], test[:, 1]
    m = Model() # a new model is used for each CV iteration
    # m = ModelPyTorch() # a new model is used for each CV iteration
    m.train(X_train, y_train)
    y_pred = m.predict(X_test)
    err += np.sum(y_pred != y_test)

total_accuracy = 1 - err/len(data)

print("Total Accuracy: "+ str(round(total_accuracy*100,1))+"%")