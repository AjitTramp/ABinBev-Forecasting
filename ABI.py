from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
import numpy

import matplotlib.pyplot as plt
from pandas import read_csv
import os
import glob
import pandas as pd

# For train Data
path_Train = "D:/Job/R/Practices/AV/ABI/Train/"
##########################
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import dateutil.parser
from  datetime import date
parser = lambda x : pd.datetime.strptime(x,'%Y%m')
df = pd.read_csv("D:/Job/R/Practices/AV/ABI/Train/Data1.csv", parse_dates=[2], index_col='YearMonth', date_parser=parser)
###BaseLine ##################################################################################
#Average at the Agency and SKU level
vf=pd.read_csv("D:/Job/R/Practices/AV/ABI/volume_forecast.csv")
L=[]
A=list(vf.Agency)
S=list(vf.SKU)
Ag=[]
Sk=[]
for i in range(len(A)):
    agency=A[i]
    sku=S[i]
    y=df[(df['Agency'] ==agency) & (df['SKU'] == sku)]
    if(len(y.index) > 0):
        y=y[['Volume']]
        Ag.append(agency)
        Sk.append(sku)
        L.append(y.mean()[0])

first=pd.DataFrame({'Agency':Ag,'SKU':Sk,'Volume':L})

sub=pd.merge(vf[['Agency','SKU']],first[['Agency','SKU','Volume']],on=['Agency','SKU'],how='left')
sub['Volume']=sub['Volume'].fillna(0)
sub[sub<0]=0
sub.to_csv("D:/Job/R/Practices/AV/ABI/volume_forecast00.csv",index=False)


#Average of last year
L=[]
A=list(vf.Agency)
S=list(vf.SKU)
Ag=[]
Sk=[]
for i in range(len(A)):
    agency=A[i]
    sku=S[i]
    y=df[(df['Agency'] ==agency) & (df['SKU'] == sku)]
    if(len(y.index) > 0):
        y = y[['Volume']]
        y = y['2017-01-01':]
        Ag.append(agency)
        Sk.append(sku)
        L.append(y.mean()[0])

first=pd.DataFrame({'Agency':Ag,'SKU':Sk,'Volume':L})

sub=pd.merge(vf[['Agency','SKU']],first[['Agency','SKU','Volume']],on=['Agency','SKU'],how='left')
sub['Volume']=sub['Volume'].fillna(0)
sub.to_csv("D:/Job/R/Practices/AV/ABI/volume_forecast01.csv",index=False)

########ARIMA MODEL
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    #X = X.astype('float32')
    train_size = int(len(X) * 0.80)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        
        model = ARIMA(history,order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

def best_model(dataset):
    best_score, best_cfg = float("inf"), None
    for p in [0,1,2,3,4,5]:
        for d in [0,1,2,3]:
            for q in [0,1,2,3]:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s RMSE=%.3f' % (order,mse))
                except:
                    continue
    if best_cfg == None:
        best_cfg = (5,1,0)
    return best_cfg


L=[]
A=list(vf.Agency)
S=list(vf.SKU)
Ag=[]
Sk=[]
for i in range(len(A)):
    agency=A[i]
    sku=S[i]
    y=df[(df['Agency'] ==agency) & (df['SKU'] == sku)]
    if(len(y.index) > 0):
        y = y[['Volume']]
        Ag.append(agency)
        Sk.append(sku)
        X = y['Volume'].values
        model = ARIMA(X, order=best_model(X))
        model_fit = model.fit(disp=0,trend='nc')
        output = model_fit.forecast()[0]
        L.append(output)
        
#######################################################################################################
import time
start = time.time()
for agency in A:
    for sku in S:
        y=df[(df['Agency'] ==agency) & (df['SKU'] == sku)]
        if(len(y.index) > 0):
            y.sort_index(inplace=True)
            y=y[['Volume']]
            Ag.append(agency)
            Sk.append(sku)
            # Define the p, d and q parameters to take any value between 0 and 2
            p = d= q =range(0,2)
            # Generate all different combinations of p, q and q triplets
            pdq = list(itertools.product(p, d, q))
            # Generate all different combinations of seasonal p, q and q triplets
            seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]   
            warnings.filterwarnings("ignore") # specify to ignore warning messages
            best_score=float("inf")
            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(y,
                                                        order=param,
                                                        seasonal_order=param_seasonal,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False)
            
                        results = mod.fit()
                        pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
                        y_forecasted = pred.predicted_mean
                        y_truth = y['2015-01-01':]
                        y_truth['forecasted']=y_forecasted
                        y_truth['Error']=(y_truth['forecasted']-y_truth['Volume'])**2
                        # Compute the mean square error
                        mse = y_truth['Error'].mean()
                        if mse < best_score:
                            best_score=mse
                            para=param
                            pas=param_seasonal
                            #print("MSE",best_score,":",param,param_seasonal)
                    except:
                        continue   


            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=para,
                                            seasonal_order=pas,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            
            results = mod.fit()
            '''
            pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
            y_forecasted = pred.predicted_mean
            y_truth = y['2015-01-01':]
            y_truth['forecasted']=y_forecasted
            y_truth['Error']=(y_truth['forecasted']-y_truth['Volume'])**2
            # Compute the mean square error
            mse = y_truth['Error'].mean()
            print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
            '''
            pred_uc = results.get_forecast(steps=1)
            # Get confidence intervals of forecasts
            pred_ci = pred_uc.conf_int()  
            v=pred_uc.predicted_mean
            L.append(v)
            end = time.time()
            print(end - start)


#Assuming res is a flat list
import csv
csvfile="D:/Job/R/Practices/AV/ABI/Train/L.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in L:
        writer.writerow([val])    


first=pd.DataFrame(Ag,columns=['Agency'])
first['SKU']=Sku
m=pd.read_csv("D:/Job/R/Practices/AV/ABI/Train/L.csv")
first['Volume']=pd.DataFrame(m,columns=['Volume'])

#first=read_csv("D:/Job/R/Practices/AV/ABI/Train/first.csv")
sub=pd.merge(volume_forecast[['Agency','SKU']],first[['Agency','SKU','Volume']],on=['Agency','SKU'],how='left')
sub.to_csv('D:/Job/R/Practices/AV/ABI/Train/sub_2.csv')

#SKU recommonedation 
###Which two skus are selling the most
result_1.to_csv('D:/Job/R/Practices/AV/ABI/Train/result_1.csv',index=False)
event_calendar.to_csv('D:/Job/R/Practices/AV/ABI/Train/Event.csv',index=False)
x=pd.merge(volume_forecast,result_1[['Agency','SKU','Volume']], on=['Agency','SKU'], how='left')
result_1.to_csv('D:/Job/R/Practices/AV/ABI/Train/Data1.csv',index=False)


