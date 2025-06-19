#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Project 3: 
# 1) Build ML Models to predict Bank of America's next day price using historcial data
# peer financial stocks( JPM, MS,C,WFC) and key macro variables
# oil prices and gold prices

#2) Feature engineering

# 3) Apply different ML algorithms such as Deciion trees, Random forest,SVM,K nearest neighbour

# 4) will evaluate the model based on R square,RMSE,MSE,MAE & other metrics


# In[2]:


# import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf


# In[3]:


#'BAC'- Bank of America
#'JPM'-JP Morgan
#'MS'- Morgan Stanley
#'C'- Citi Group
#'WFC'- Wells Fargo 
#'SPY'-S&P500
#'^VIX'- CBOE Volatility Index
#'^TNX'- 10Y US Treasury yeild
#'DX-Y.NYB'- US Dollar Index
#'CL=F'- Crude oil
#'GC=F'- Gold futures


# In[4]:


# Download the data from yahoo finance

tickers=['BAC','JPM','MS','C','WFC','SPY','^VIX','^TNX','DX-Y.NYB','CL=F','GC=F']
data=yf.download(tickers,start='2002-01-01',end='2025-01-01')['Close']
data


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


# missing data
#1) Drop the values
#2) forward fill - carries forward the last known value
#3) backward fill- filling the value backward
#4) average 
#4) Interpolation - linear or cubic spline or monotone convex 


# In[9]:


# here we will do forward fill
data=data.ffill()
data.head()


# In[10]:


data.isnull().sum()


# In[11]:


#correlation in our data
data.corr()


# In[12]:


import matplotlib.pyplot as plt

# create a larger picture
plt.figure(figsize=(16,9))

#plot each stock
plt.plot(data.index,data['BAC'],label='BAC:Bank of America',linewidth=2)
plt.plot(data.index,data['JPM'],label='JPM:JP Morgan',linewidth=2)
plt.plot(data.index,data['MS'],label='MS: Morgan Stanley',linewidth=2)
plt.plot(data.index,data['C'],label='C: Citi Group',linewidth=2)
plt.plot(data.index,data['WFC'],label='WFC: Wells Fargo',linewidth=2)
plt.plot(data.index,data['SPY'],label='SPY:S&P500',linewidth=2)

# Title,label for X & Y Axis
plt.title('Stock price comparison : Major bank vs S&P500')
plt.xlabel('Date',fontsize=16)
plt.ylabel('Close Price($)',fontsize=16)

plt.grid(True,linestyle='--',alpha=0.5)

#customize legend
plt.legend(fontsize=12,loc='upper left')

#show the plot
plt.tight_layout()
plt.show()


# In[13]:


# feature engineering
df=pd.DataFrame(index=data.index)

# create lag features-stock data
df['JPM(t-1)']=data['JPM'].shift(1)
df['BAC(t-1)']=data['BAC'].shift(1)
df['MS(t-1)']=data['MS'].shift(1)
df['C(t-1)']=data['C'].shift(1)
df['WFC(t-1)']=data['WFC'].shift(1)
df['SPY(t-1)']=data['SPY'].shift(1)

# create lag features-macro data
df['^VIX(t-1)']=data['^VIX'].shift(1)
df['^TNX(t-1)']=data['^TNX'].shift(1)
df['GCF(t-1)']=data['GC=F'].shift(1)
df['DX-Y.NYB(t-1)']=data['DX-Y.NYB'].shift(1)
df['CL=F(t-1)']=data['CL=F'].shift(1)

# Technincal Indicators= moving average & rolling volatility
df['BAC_MA5']=data['BAC'].rolling(window=5).mean().shift(1)
df['BAC_MA10']=data['BAC'].rolling(window=10).mean().shift(1)
df['BAC_Vola5']=data['BAC'].pct_change(5).shift(1)

# Target variable
df['Target']=data['BAC']
df.dropna(inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


# Train our ML Algo
# a) Tell what is X & Y variables
#b) split our data into training and test
# c) apply ML Algorithm
# d) do the prediction
# e) evaluate the model based on RMSE, MSE,R2
#f) visualization


# In[16]:


# a) Tell what is X & Y variables
X=df.drop('Target',axis=1)
Y=df['Target']


# In[17]:


#b) split our data into training and test(80:20)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,shuffle=False,test_size=0.10)


# In[18]:


# c) apply ML Algorithm : Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(max_depth=4)
dt_model.fit(X_train,Y_train) #train my dt


# In[19]:


#d) do prediction

dt_pred=dt_model.predict(X_test)
dt_pred


# In[20]:


result=pd.DataFrame(Y_test.index)
result['Actual']=Y_test.values
result['Predicted']=dt_pred
result


# In[21]:


# evaluate the model based on R2, rmse,mse
from sklearn.metrics import r2_score,mean_squared_error
def evaluate_model(y_true,y_pred,model_name):
    r2=r2_score(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    print("Model Name",model_name)
    print("R2 Value", r2)
    print("MSE",mse)
    print("RMSE",rmse)
    
evaluate_model(Y_test,dt_pred,"Decision Tree")


# In[22]:


# f) Visualtion => Actual vs Forecasted
# Plot figure => tell figure size
plt.figure(figsize = (14,8))
#Plot Actual Value and Predicted Value (X = Date, Y = Stock Price)
plt.plot(Y_test.index, Y_test, label = 'Actual Bank of America Stock Price',linewidth=2,color='red')
plt.plot(Y_test.index, dt_pred, label = 'Decision Tree Prediction', linewidth=2,color='black',linestyle='--')

# Highlight title, xlabel, and ylabel
plt.title("Actual vs Predicted Bank of America Stock Price")
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(alpha = 0.5)
plt.legend()
plt.show()


# In[23]:


# What all features are important?
importance = dt_model.feature_importances_
features_name = X_train.columns
df_features = pd.DataFrame({'Feature':features_name, 'Importance': importance})
df_features = df_features.sort_values(by = 'Importance', ascending = False)
df_features


# In[24]:


from sklearn.tree import plot_tree
plt.figure(figsize = (20,10))
plot_tree(dt_model, feature_names = X.columns, filled = True, rounded = True,precision=2)
plt.title("Decision Tree Flowchart")
plt.show()


# In[ ]:




