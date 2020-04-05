#!/usr/bin/env python
# coding: utf-8

# Test the Generalized Linear Model  
# Reference: https://qiita.com/ground0state/items/38123b70c152253befe4

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
display(df.head(10))

# Split data
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size =0.8)

# Create train data
data = pd.concat([X_train, y_train], axis=1)

# Create linear predictor
formula = "PRICE ~  1 + INDUS + CHAS + RM + RAD"

# Chose link function
link = sm.genmod.families.links.log

# Chose distribution
family = sm.families.Poisson(link=link)

# Fit model
model = smf.glm(formula=formula, data=data, family=family )
result = model.fit() 
display(result.summary())

# Evaluate AIC
print(result.aic)

# Predict
y_pred = result.predict(X_test)
df_test = pd.concat([X_test, y_test.rename("ACTUAL"), y_pred.rename("PREDICT")], axis=1).reset_index(drop = True)
display(df_test.head(10))

# Caluclate Accuracy
df_test["AE"] = np.abs(df_test["ACTUAL"] - df_test["PREDICT"])
display(df_test["AE"].mean())

# Another model (Use all valiables)
# Or use follow code
X_train.const = sm.add_constant(X_train)
model2 = sm.GLM(y_train, X_train.const, family=sm.families.Poisson())
result2 = model2.fit() 
display(result2.summary())

# Evaluate AIC
result2.aic

# Predict
X_test.const = sm.add_constant(X_test)
y_pred2 = result2.predict(X_test.const)
df_test2 = pd.concat([X_test, y_test.rename("ACTUAL"), y_pred2.rename("PREDICT")], axis=1).reset_index(drop = True)
display(df_test2.head(10))

# Caluclate Accuracy
df_test2["AE"] = np.abs(df_test2["ACTUAL"] - df_test2["PREDICT"])
display(df_test2["AE"].mean())



