import pickle
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix as sparse_matrix
from sklearn.metrics import mean_squared_error
import datetime
import math

def create_user_item_matrix(ratings,user_key="user",item_key="item"):

    n = len(set(ratings[user_key]))
    d = len(set(ratings[item_key]))

    user_mapper = dict(zip(np.unique(ratings[user_key]), list(range(n))))
    item_mapper = dict(zip(np.unique(ratings[item_key]), list(range(d))))

    user_inverse_mapper = dict(zip(list(range(n)), np.unique(ratings[user_key])))
    item_inverse_mapper = dict(zip(list(range(d)), np.unique(ratings[item_key])))

    user_ind = [user_mapper[i] for i in ratings[user_key]]
    item_ind = [item_mapper[i] for i in ratings[item_key]]

    X = sparse_matrix((ratings["rating"], (user_ind, item_ind)), shape=(n,d))
    
    return X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind        

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')

def classification_error(y, yhat):
    return np.mean(y!=yhat)


def test_and_plot(model,X,y,Xtest=None,ytest=None,title=None,filename=None):

    # Compute training error
    yhat = model.predict(X)
    #trainError = np.mean((yhat - y)**2)
    trainError = mean_squared_error(y,yhat)
    print("Training mse = %.10f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        # testError = np.mean((yhat - ytest)**2)
        testError = mean_squared_error(ytest,yhat)
        print("Test mse    = %.10f" % testError)

    # Plot model
    plt.figure()
    plt.plot(X,y,'b.')

    # Choose points to evaluate the function
    if(len(X[0])==1):
        Xgrid = np.linspace(np.min(X),np.max(X),1000)[:,None]
        ygrid = model.predict(Xgrid)
        plt.plot(Xgrid, ygrid, 'g')

        if title is not None:
            plt.title(title)

        if filename is not None:
            filename = os.path.join("..", "figs", filename)
            print("Saving", filename)
            plt.savefig(filename)

def linear_model_test(model,X,y,Xtest=None,ytest=None):

    # Compute training error
    yhat = model.predict(X)
    #trainError = np.mean((yhat - y)**2)
    trainError = mean_squared_error(y,yhat)
    print("Training mse = %.10f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        # testError = np.mean((yhat - ytest)**2)
        testError = mean_squared_error(ytest,yhat)
        print("Test mse    = %.10f" % testError)

def linear_model_test_with_plot(model,X,y,Xtest=None,ytest=None):

    # Compute training error
    yhat = model.predict(X)
    #trainError = np.mean((yhat - y)**2)
    trainError = mean_squared_error(y,yhat)
    print("Training mse = %.10f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        # testError = np.mean((yhat - ytest)**2)
        testError = mean_squared_error(ytest,yhat)
        print("Test mse    = %.10f" % testError)

def predictions_plot(y_pred,y_real,exp_ct_name,exp_fea_name,phase,n_neigh=None):
    rmse = np.sqrt(mean_squared_error(y_pred,y_real))
    l=len(y_pred)
    plt.plot(np.arange(l),y_pred,marker="*",label="pred")
    plt.plot(np.arange(l),y_real,marker=".",label="real")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    title="Pred_"+exp_fea_name+"_rmse is "+str(rmse)
    plt.title(title)
    plt.legend()
    plotname="Pred_"+phase+"_plot_"+exp_fea_name+"_d"+str(l)+"_"+exp_ct_name+".pdf"
    if n_neigh !=None:
        plotname="Pred_"+phase+"_plot_"+exp_fea_name+"_d"+str(l)+"_n"+str(n_neigh)+"_"+exp_ct_name+".pdf"
    fname = os.path.join("..", "figs", plotname)
    if fname is not None:
        fname = os.path.join("..", "figs", fname)
        print("Saving", fname)
    plt.savefig(fname)
    plt.clf()
    plt.close('all')   
    
    
    
def dif_series(x,day,refine=None):
    y=x-x.shift(day)
    if refine == 1:
        y=y[day:]
    return y
        
def s2m(series):
    X = np.zeros((1,len(series)))
    X[0] = series
    return X

def s2v(series):
    return s2m(series).T
   
def make_mapper(dataset):

    ct_key="country_id"
    dataset_ct = dataset[ct_key]
    n_ct=len(set(dataset_ct))
    ct_mapper = dict(zip(list(set(dataset_ct)),list(range(n_ct))))
    ct_inv_mapper = dict(zip(list(range(n_ct)),list(set(dataset_ct))))
    ct_ind = [ct_mapper[i] for i in dataset_ct]
    return n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind
    

def shift_fill0(series,lag): 
    series=series.shift(lag)
    series[:lag]=0.0
    return series     
    
def shift_fill0_array(array,lag): 
    int_lag =int(lag)
    array=np.roll(array,int_lag)
    array[:int_lag]=0.0
    return array
# from original dataset generate the death_100k column and return the new dataset
def get_death_100k(dataset):
    temp = dataset["cases"]
    temp[temp == 0] = 1
    deaths_100k_dataset=dataset["deaths"]*dataset["cases_100k"]/temp
    #print(deaths_100k_dataset)
    dataset["deaths_100k"]=pd.Series(deaths_100k_dataset, index=dataset.index)
    return dataset
    

def test_death_100k(dataset,n):
    deaths=dataset["deaths"]
    cases=dataset["cases"]
    cases_100k=dataset["cases_100k"]
    cases[cases==0]=1
    return deaths[n]*cases_100k[n]/cases[n]
    
def get_num_people_100k(dataset,ct_name):
    #choose the newest data to compute population
    #Here's some assumption that population wont change during the whole process
    n_row=dataset[dataset["country_id"]==ct_name]["deaths"].index[-1]
    cases_100k = dataset["cases_100k"]
    cases_100k[cases_100k ==0] =1
    num_people_100k = dataset["cases"][n_row]/cases_100k[n_row]
    return num_people_100k

def get_all(dataset):
    n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind = make_mapper(dataset)
    dataset=get_death_100k(dataset)
    dataset["daily_cases"] = 0.0
    dataset["daily_deaths"] = 0.0
    dataset["daily_deaths_100k"] = 0.0
    for nct in range(1,n_ct):     
        dataset1=dataset[dataset[ct_key]==ct_inv_mapper[nct]]
        l=len(dataset1)
        daily_cases=np.zeros(l)
        daily_deaths=np.zeros(l)
        daily_deaths_100k=np.zeros(l)
        for i in range(1,l):
            daily_deaths[i]= dataset1.iloc[i,3] - dataset1.iloc[i-1,3]
            daily_cases[i] = dataset1.iloc[i,2] - dataset1.iloc[i-1,2]
            daily_deaths_100k[i] =dataset1.iloc[i,6] - dataset1.iloc[i-1,6]
        dataset1["daily_cases"]=daily_cases
        dataset1["daily_deaths"]=daily_deaths
        dataset1["daily_deaths_100k"]=daily_deaths_100k
        dataset[dataset[ct_key]==ct_inv_mapper[nct]] = dataset1
    return dataset
#Estimator to average the value between ensamble predictions from other coutries and auto regression    
def estimator(MLR_pred,self_pred,distances,method="cross"):
    if type(self_pred) == pd.core.series.Series:
        self_pred=self_pred.reset_index(drop = True)
    if type(MLR_pred) == pd.core.series.Series:
        MLR_pred=MLR_pred.reset_index(drop = True)

    if method == "cross":
        #The maximum of weight depends on the mean correlation of n_neighbor countreis
        #The n_neighbors are selected strictly to provide long term information to the prediction
        #Especially those countries with earlier outbreaks
        weight_corr = np.mean(distances)
        weight_self = 1.0
        w_max = weight_corr
        w_min = 0.0
        #The weight of the MLR part is made from calculating the mean correlations
        #Between spec country
        #They are selected N_neighbor countries 
        #so they have higher weight if they are more similar to spec coutnrey       
        predictions=np.zeros(len(MLR_pred))
        for day in range(len(MLR_pred)):  
            #The error of auto regression increase as time increases
            #The curve will tend to go to the averaged predictions made by other countries
            #This is an linear scaling over the weights of other countries according to time
            weight_MLR=w_function(day,len(MLR_pred),w_min,w_max)
            w_MLR = weight_MLR / (weight_corr + weight_self)
            predictions[day] = w_MLR * MLR_pred[day] + (1-w_MLR) * self_pred[day]
        return predictions
    else:
        print("Error: Method not defined! ")
    
def w_function(x,length,w_min,w_max):
    return w_min+(w_max-w_min)/length*x



# def read_data(filename):
#     with open(os.path.join("data", filename), "rb") as f:
#         OriginDataSet = pd.read_csv(f,keep_default_na=False, na_values='_')
#     TrainingSet = OriginDataSet.copy()
#     data = TrainingSet["date"]
#     print(data)
#     refer_date = datetime.datetime.timestamp(datetime.datetime.strptime(data[0], "%m/%d/%Y"))
#     for i in range(len(data)):
#         TrainingSet["date"][i] = math.ceil((datetime.datetime.timestamp(datetime.datetime.strptime(data[i], "%m/%d/%Y")) - refer_date) / 86400)
#     TrainingSet.to_csv(r'.\data\Modified_data.csv', index=False, header=True)




















    