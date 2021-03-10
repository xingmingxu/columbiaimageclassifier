# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:56:25 2020

@author: User
"""



from PIL import Image
import numpy as np
import pandas as pd
import statistics
import statsmodels.api as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import regularizers
import random
import copy
from datetime import datetime as dt
## logistic regression
from sklearn.linear_model import LogisticRegression
## randomforest and decision trees
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
## xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.svm import LinearSVR
from numpy.random import seed
import warnings
warnings.filterwarnings("ignore")



fold = ("C:\\Users\\User\\Documents\\Xingming\\Programming\\"
    + "Projects\\columbiaImages\\") ## insert folder
pm = pd.read_csv(fold + "photoMetaData.csv")
n = len(pm)

# identify "outdoor-day"
y = [int(pm.category[i] == "outdoor-day") for i in range(n)]

# insert into pandas dataframe
xx = pd.DataFrame(columns = ["R", "G", "B"]) 

for i in range(n):
    
    path = fold + "\\columbiaImages\\" + pm.name[i]
    imframe = Image.open(path)
    #im.show()
    npframe = np.array(imframe.getdata())
    pdpixels = pd.DataFrame(npframe / 255)

    ms = [statistics.median(pdpixels[j]) for j in range(3)]
    rgb = pd.DataFrame([ms], columns = ["R", "G", "B"])
    xx = xx.append(rgb, ignore_index = True)
    print(i)

    
x = xx ##backup
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clflm = linear_model.LinearRegression()
clflm.fit(x_train, y_train)

y_predlm = clflm.predict_proba(x_test)
for i in range(len(y_predlm)):
    if y_predlm[i] > 0.5:
        y_predlm[i] = 1
    else:
        y_predlm[i] = 0
        
## ADD TABLE


##y_predglm = predict(x_test)

#Create a Gaussian Classifier
clfrf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfrf.fit(x_train,y_train)
y_predrf = clfrf.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_predrf))



xgbr = xgb.XGBRegressor(base_score = 0.5, objective='binary:logistic') 
print(xgbr)
xgbr.fit(x_train, y_train)
xgb_score = xgbr.score(x_train, y_train)
print("Training score: ", xgb_score) ## 0.85 !!

y_predxgbr = xgbr.predict(x_test)
for i in range(len(y_predxgbr)):
    if y_predxgbr[i] > 0.5:
        y_predxgbr[i] = 1
    else:
        y_predxgbr[i] = 0

## graph
"""
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_predxbgr, label="predicted")"""