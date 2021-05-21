
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#from statsmodels.tsa.ar_model import AutoReg

from statsmodels.tsa import stattools 
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import seaborn as sns
#from sklearn.neighbors import NearestNeighbors
import linear_model
import utils
import country
import warnings
from AutoRegression import My_AutoReg
from AutoRegression import AutoReg1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    parser.add_argument('-d','--test_day', required=False)
    parser.add_argument('-l','--test_lag', required=False)
    parser.add_argument('-n','--test_num', required=False)
    parser.add_argument('-m','--test_method', required=False)
    parser.add_argument('-p','--test_par', required=False)
    
    io_args = parser.parse_args()
    question = io_args.question
    

    def read_dataset(filename):
        with open(os.path.join("..", "data", filename), "rb") as f:
            dataset = pd.read_csv(f,keep_default_na=False)
        return dataset

    if question == 'find_nan_ct':
        dataset = read_dataset("phase1_training_data.csv")
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        dataset=utils.get_all(dataset)
        dataset_CA = dataset[dataset[ct_key]=="CA"]
        
        print("Input Country Number:",n_ct)
        n_ct1=0
        i=0
        ct_nan_ind=n_ct*[None]
        for nct in range(n_ct):
            if type(ct_inv_mapper[nct]) == str:
                n_ct1 +=1
            else:
                ct_nan_ind[i] = nct
                i+=1      
        print("Output Country Number:",n_ct1)
        if n_ct1 == n_ct:
            print("No 'nan' named country found!")
                   
    if question == 'lagplot':
        dataset = read_dataset("phase1_training_data.csv")
        dataset=utils.get_all(dataset)
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)

        dataset_CA=dataset[dataset[ct_key]=='CA']
        #cases_CA = dataset_CA["cases"]
        #deaths_CA = dataset_CA["deaths"]
        
        testname ="deaths_100k"
        test_CA = dataset_CA[testname]
        #autocorrelation_plot(cases_CA)
        
        #pacf analysis
        nlags = 300
        #conf_CA = np.zeros(2)
        pacf_CA, conf_CA = stattools.pacf(test_CA, nlags=nlags,alpha=0.05)
        #pacf1 = pacf_CA[pacf_CA > conf_CA[1] or pacf_CA < conf_CA[0]]
        conf0 = conf_CA[:,0]
        conf1 = conf_CA[:,1]
        ind_lags = np.argsort(pacf_CA)
        ind_lags = ind_lags[::-1]
        
        #pacf_lag_plot for test dataset with testname

        nlags_max =100
        #pacf output
        plot_pacf(test_CA,lags=nlags_max)
        #plotname = testname+"_pacf_lag_plot_CA.pdf"
        plotname = testname+"_pacf_lag_plot_CA.pdf"
        fname = os.path.join("..", "figs", plotname)
        #plt.xlim(left = 80)
        plt.savefig(fname)
        plt.clf()
        
        plt.plot(conf0)
        plt.plot(conf1)
        plotname = "conf.pdf"
        fname = os.path.join("..", "figs", plotname)
        plt.xlim(left = 80)
        plt.savefig(fname)
        plt.clf()
        
        testname=testname
        refname="date"
        sns.scatterplot(dataset_CA[refname],dataset_CA[testname])
        plotname = testname+"_CA.pdf"
        fname = os.path.join("..", "figs", plotname)
        plt.xlim(left = 50)
        plt.savefig(fname)
        plt.clf()
    
    if question == 'ar_ca':
        dataset = read_dataset("phase1_training_data.csv")
        #Input arg parameters
        test_lag = io_args.test_lag
        test_lag = int(test_lag)
        test_day= io_args.test_day
        test_day = int(test_day)

        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
        dataset_CA=dataset[dataset[ct_key]=='CA']
        cases_CA = dataset_CA["cases"]
        deaths_CA = dataset_CA["deaths"]
        
        #Choose experiment set
        exp_set = deaths_CA
        exp_set = exp_set.reset_index(drop = True)

        X = exp_set
        train_set, test_set = X[:len(X)-test_day], X[len(X)-test_day:]
        
        # train autoregression
        lags=test_lag
        model1 = AutoReg1(lags=lags)
        model1.fit(train_set)

        model3 = My_AutoReg(lag=lags)
        model3.fit(train_set)
        #print('Coefficients: %s' % model_fit.params)
       
        # make predictions
        predictions_train= model1.predict(start=lags, end=len(train_set)-1)
        # predictions_train= model3.predict(train_set,start=lags, end=len(train_set)-1)
        # predictions_test= model.predict(start=len(train_set), end=len(train_set)+len(test_set))
        # predictions_all= model.predict(start=lags, end=len(train_set)+len(test_set))
        #Train Error
        rmse_train = sqrt(mean_squared_error(train_set[lags:], predictions_train))
        print('Train RMSE: %.3f' % rmse_train)        
        # rmse_test = sqrt(mean_squared_error(test_set, predictions_test))
        # print('Test RMSE: %.3f' % rmse_test)
        # plot results
        plt.plot(train_set[lags:])
        plt.plot(predictions_train, color='red')
        fname = os.path.join("..", "figs", "prediction_train_plot_CA.pdf")
        plt.savefig(fname)
        plt.clf()
        
        # plt.plot(test_set)
        # plt.plot(predictions_test, color='red')
        # fname = os.path.join("..", "figs", "prediction_test_plot_CA.pdf")
        # plt.savefig(fname)
        # plt.clf()
        
        
        # plt.plot(exp_set[lags:])
        # plt.plot(predictions_all, color='red')
        # fname = os.path.join("..", "figs", "prediction_all_plot_CA.pdf")
        # plt.xlim(left = len(exp_set[lags:]) -2*len(test_set))
        # plt.savefig(fname)
        # plt.clf()            
        # plt.close('all')
         
    # if question == 'knn':
    #     dataset = read_dataset("phase1_training_data.csv")
    #     test_num= io_args.test_num
    #     test_num= int(test_num)
    #     test_method= io_args.test_method
    #     #test_method=int(test_method)
        

    #     n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
    #     dataset_CA=dataset[dataset[ct_key]=='CA']
    #     cases_CA = dataset_CA["cases"]
    #     cases14_CA =dataset_CA["cases_14_100k"]
    #     day_n = len(cases_CA)
        
    #     exp_name = "cases_14_100k"
    #     exp_dataset = np.zeros((n_ct,day_n))
    #     exp_ct_name ='CA'
        
    #     #Choose the dataset regarding the name interested
    #     cases14_exp = dataset[dataset[ct_key]==exp_ct_name][exp_name]
        
    #     for nct in range(1,n_ct):
    #         #fillter out nan set
    #         dataset_ct = dataset[dataset[ct_key]==ct_inv_mapper[nct]]
    #         dataset1 = dataset_ct[exp_name]
    #         exp_dataset[nct] = dataset1
        
    #     exp_dataset = exp_dataset[1:]
    #     target= np.zeros((1,day_n))
    #     target[0]= cases14_exp
    #     #!!!!!!remember the true country index is ind_ct = indices+1
    #     #Using KNN model to find K nearest countries 
    #     n_neigh = test_num

    #     metric_method =test_method
    #     model = NearestNeighbors(metric=metric_method)
    #     model.fit(exp_dataset)
    #     distances, indices=model.kneighbors(target,n_neighbors=n_neigh)
    #     #!!!!!!remember the true country index is ind_ct = indices+1
    #     indices = indices +1
    #     indices = indices[0]
    #     distances = distances[0]
    #     ct_neigh =n_neigh*[None]
    #     for i in range(n_neigh):
    #         ct_neigh[i] = ct_inv_mapper[indices[i]] 
    #     #ct_neigh = [ct_inv_mapper[i] for i in range(len(indices))]
    #     ct_neigh_dict=dict(zip(ct_neigh,list(distances)))
    #     print(ct_neigh_dict)
        
    #     #Distance Plot of N_nearest countreis
    #     plt.plot(distances[1:])
    #     plt.xlabel("Neighbor_Country")
    #     plt.ylabel("Distance")
    #     plotname="Distance_plot_n"+str(n_neigh)+"_"+exp_ct_name+"_"+metric_method+".pdf"
    #     fname = os.path.join("..", "figs", plotname)
    #     plt.savefig(fname)
    #     plt.clf()
    #     plt.close('all')
            
    if question == "linear":
        test_par = io_args.test_par
        
        dataset = read_dataset("phase1_training_data.csv")
        dataset = utils.get_all(dataset)
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
        dataset_CA=dataset[dataset[ct_key]=='CA']
        cases_CA = dataset_CA["cases"]
        cases14_CA =dataset_CA["cases_14_100k"]
        day_n = len(cases_CA)
        
        ref_name = "deaths_100k"
        exp_name = "deaths_100k"

        #exp_dataset = np.zeros((n_ct,day_n))
        exp_ct_name ='CA'
        ref_ct_name ='UK'
                 
        exp_dataset= dataset[dataset[ct_key]==exp_ct_name][exp_name]
        ref_dataset= dataset[dataset[ct_key]==ref_ct_name][ref_name]
        exp_dataset = exp_dataset.reset_index(drop = True)
        ref_dataset = ref_dataset.reset_index(drop = True)
        
        y=exp_dataset
        
        test_ratio=11/day_n
        
        lag = 0
        
        test_day = int(day_n*test_ratio)
        train_day = day_n-test_day
        train_ref_dataset=ref_dataset[:train_day]  
        train_exp_dataset=exp_dataset[:train_day]
        
        
        if exp_name == ref_name:
            maxcorr,lag=country.cross_corr(train_ref_dataset,train_exp_dataset,60)
            print("Best Lag =",lag)
            print("Cross correlation =",maxcorr)
        else:
            maxcorr,_=pearsonr(train_ref_dataset,train_exp_dataset)
        
        print("Correlation =",maxcorr)
        
        
        train_ref_dataset=utils.shift_fill0(train_ref_dataset,lag)


        X=utils.s2v(train_ref_dataset)
        y=utils.s2v(train_exp_dataset)
        Xtest=utils.s2v(ref_dataset[train_day:])
        ytest=utils.s2v(exp_dataset[train_day:])   

        poly_par =1
        if test_par !=None:
            poly_par = int(test_par)
        model = linear_model.LeastSquaresPoly(p=poly_par)
        model.fit(X,y)
        titlename = exp_name+"_"+exp_ct_name+"_vs_"+ref_name+"_"+ref_ct_name+"_lag"+str(lag)+"_p"+str(poly_par)
        filename = "LRplot_" + titlename + ".pdf" 
        utils.test_and_plot(model,X,y,Xtest,ytest,title=titlename,
                            filename=filename)
        
       
    if question == "corr":
                    
        dataset = read_dataset("phase1_training_data.csv")
        dataset = utils.get_all(dataset)
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
        day_n = len(dataset[dataset[ct_key]=='CA']["cases"])
        
        exp_name = "cases_14_100k"
        exp_ct_name ='CA'

        n_neigh = 100
        cutoff = 0.5
        n_neigh, n_neigh_indices, n_neigh_distances, n_neigh_lags, n_neigh_dict,n_neigh_dict_lags = country.select_country(
            dataset,exp_name,exp_ct_name,n_neigh=n_neigh,method="cross",cutoff=cutoff)
        print("N_neighbor = %d with cutoff = %.3f"%(n_neigh,cutoff))
        print(n_neigh_dict)
        print(n_neigh_dict_lags)
        
        plt.plot(n_neigh_distances)
        plt.xlabel("Neighbor_Country")
        plt.ylabel("Correlation")
        plotname="Ncorr_plot_n"+str(n_neigh)+"_"+exp_ct_name+".pdf"
        fname = os.path.join("..", "figs", plotname)
        plt.savefig(fname)
        plt.clf()
        plt.close('all')
        
#------------AMLR: Main prediciton model based on  AutoReg and normal Multi-variable Linear Regressions-------    
    
    if question == "AMLR":
        warnings.filterwarnings("ignore")
        test_par = io_args.test_par
        test_day = io_args.test_day
        test_num = io_args.test_num

        
        dataset = read_dataset("phase1_training_data.csv")
        dataset = utils.get_all(dataset)
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
        dataset_CA=dataset[dataset[ct_key]=='CA']
        cases_CA = dataset_CA["cases"]
        cases14_CA =dataset_CA["cases_14_100k"]
        day_n = len(cases_CA)
        
        exp_fea_name = "deaths_100k"
        ref_fea_name = "deaths_100k"
        #exp_dataset = np.zeros((n_ct,day_n))
        exp_ct_name ='CA'
        #ref_ct_name ='UK'
        
        #Find nearest neighbors of CA,return indices, get a combined all neighbor country dataset
        n_neigh = 100
        cutoff = 0.9
        if test_num != None:
            test_num = int(test_num)
            n_neigh = test_num
        if test_par != None:               
            test_par = float(test_par)
            cutoff = test_par
        n_neigh, n_neigh_indices, n_neigh_distances,n_neigh_lags, n_neigh_dict,n_neigh_dict_lags = country.select_country(
            dataset,exp_fea_name,exp_ct_name,n_neigh=n_neigh,
            method="cross",cutoff=cutoff,R=30,ref_fea_name=ref_fea_name)
        print("y is '%s' from country '%s'"%(exp_fea_name,exp_ct_name))
        print("N_neighbor = %d with cutoff = %.3f"%(n_neigh,cutoff))
        #print(n_neigh_dict)
        
        #-------------------------------------------------------------------------------------------------
        #   Combine the lags list and modify each country, return a dataset of reference
        #   the reference is used to be a X input matrix
        #   all subset of dataset are shifted based on the best lag according to the cross correlation
        #   we are fitting all the correlated countries with the same time-shift
        #   so that the total linear correlation coefficient is significantly improved
        #   we just considered the case that all countries don't have the same date of outbreak of pandemic
        #   this is an important factor to fit our data
        #--------------------------------------------------------------------------------------------------
        ref_dataset=country.crosscombine_ct(dataset,ref_fea_name,n_neigh_indices,n_neigh_lags)
        exp_dataset= dataset[dataset[ct_key]==exp_ct_name][exp_fea_name]
        exp_dataset=exp_dataset.reset_index(drop = True)
        
        test_ratio=11/day_n
        if test_day != None:               
            test_day = float(test_day)
            test_ratio = test_day
            
        test_day = int(day_n*test_ratio)
        train_day = day_n-test_day     
        
        X=ref_dataset[:train_day]
        y=utils.s2v(exp_dataset[:train_day])
        
        Xtest=ref_dataset[train_day:]
        ytest=utils.s2v(exp_dataset[train_day:])
        
        self_AR_X=exp_dataset[:train_day]
        self_AR_y=exp_dataset[train_day:]
        
        #k is the lag chosen as parameter in AutoReg model
        #optimal lags is 2 as simplist optimal lag
        #optimal lags is 12 as second best lag
        k=2
        #Implement auto regression on the self feature of the selected country
        self_auto_model = AutoReg1(lags=k)
        self_auto_model.fit(self_AR_X)
        self_train_predictions = self_auto_model.predict(start=k,end=train_day-1)
        self_train_rmse = sqrt(mean_squared_error(self_AR_X[k:], self_train_predictions))
        self_test_predictions = self_auto_model.predict(start=train_day, end=train_day+test_day-1)
        self_test_rmse = sqrt(mean_squared_error(self_AR_y, self_test_predictions))
        #---------------------------------------------------------------------------------------------------
        #Implement the parallel Linear Regression on the neighbors of selected country:
        #
        #   Feature(selected country,past)= LR (Feature(other countries,past) )
        #
        #So we can train a model to fit the past feature of selected country
        #Later, we will use this model to parallelly predict the Future Feature of the spec country:
        #
        #   Feature(selected country,future)= a*LR (Feature(other countries,future) ) + b*AR (Feature(spec country,past)) + c
        #
        #Note that we have already shifted the training set so that all outbreaks happend at the same time
        #Based on this we are able to say, the correlation decides whether we can predict the spec country
        #----------------------------------------------------------------------------------------------------
        parallel_model = linear_model.LeastSquaresPoly(p=1)
        parallel_model.fit(X,y)
        parallel_predictions=parallel_model.predict(X)
        train_parallel_rmse = sqrt(mean_squared_error(parallel_predictions,y))
        print("MultiLinear train Process rmse is %.3f"%train_parallel_rmse)

        #Implement auto_regression for all neighbors for feature "cases_100k" except selected country
        test_rmse_ct = np.zeros(n_neigh)
        train_rmse_ct = np.zeros(n_neigh)
        test_predictions = np.zeros((n_neigh,test_day))
        for nct in range(n_neigh):
            dataset_ct = dataset[dataset[ct_key]==ct_inv_mapper[n_neigh_indices[nct]]]
            X = dataset_ct[ref_fea_name]
            X = X.reset_index(drop = True)
            train, test = X[:(len(X)-test_day)], X[(len(X)-test_day):]

            auto_model=AutoReg1(lags=k)
            auto_model.fit(train)
            train_predictions = auto_model.predict(start=k,end=len(train)-1)
            train_rmse_ct[nct] = sqrt(mean_squared_error(train[k:], train_predictions))
            test_predictions[nct] = auto_model.predict(start=len(train), end=len(train)+len(test)-1)
            test_rmse_ct[nct] = sqrt(mean_squared_error(test, test_predictions[nct]))
        
        #Compute all errors happend in the AutoReg process
        #Add up and get the mean value of rmse of each phase of the regression process of each country:
        test_auto_rmse = np.mean(test_rmse_ct)
        train_auto_rmse = np.mean(train_rmse_ct)
        
        #This is the matrix of the auto predicted value of all other countries
        #This can be considered as an X of a LR model, then we can predict the future feature of spec country
        auto_predictions = test_predictions.T
        
        #Prediction phase and calculate the final result of the selected feature of the spec country
        MLR_predictions = parallel_model.predict(auto_predictions)
        test_exp_dataset = exp_dataset[train_day:]
        final_rmse = sqrt(mean_squared_error(test_exp_dataset, MLR_predictions))
        baseline = sqrt(mean_squared_error(test_exp_dataset, np.zeros(test_day)))
        
#------------------Output Phase-------------------------------------------------------------------------------------------------
        print("AutoReg train Process  mean_rmse is %.3f"%train_auto_rmse)
        print("AutoReg test Process(validation process)  mean_rmse is %.3f"%test_auto_rmse)
        print("Final test rmse is %.3f"%final_rmse)
        print("Final test percentage error is %.3f%%"%(final_rmse/baseline*100))
        #Predict the final deaths feature value based on the population information of the spec country
        #Theres almost no errors added here regarding the defination of deaths_100k which is sctrictly proportional to deaths
        
#------------------deaths feature-----------------------------------------------------------------------------------------------           
        if exp_fea_name == ref_fea_name and exp_fea_name =="deaths_100k":
            deaths_test_dataset=dataset[dataset[ct_key]==exp_ct_name]["deaths"][train_day:]
            num_people_100k = utils.get_num_people_100k(dataset,exp_ct_name)
            
            MLR_deaths_predictions = num_people_100k * MLR_predictions
            
            self_deaths_predictions = num_people_100k * self_test_predictions
            
            MLR_deaths_rmse = sqrt(mean_squared_error(MLR_deaths_predictions.T[0],deaths_test_dataset))
            
            print("Without Self_AR, MultiLR deaths test rmse is %.3f"%MLR_deaths_rmse)
            
            
            # predict the deaths by combining two models together : stacked accroding to the correlation coefficients
            
            AMLR_deaths_predictions = utils.estimator(MLR_deaths_predictions.T[0],self_deaths_predictions,n_neigh_distances)
            
            AMLR_deaths_rmse = sqrt(mean_squared_error(AMLR_deaths_predictions,deaths_test_dataset))
            
            print("With Self_AR, Final AutoMultiLR deaths test rmse is %.3f"%AMLR_deaths_rmse)
            
            utils.predictions_plot(AMLR_deaths_predictions,deaths_test_dataset,
                                   exp_ct_name,exp_fea_name,"test",n_neigh=n_neigh
                                   )
#-------------------AMLR model training and prediction-----------------Completed--------------------------------------------------------------

            

    if question == "find_n": #I found that N_neighbors = 7 is best
        warnings.filterwarnings("ignore")
        test_par = io_args.test_par
        test_day = io_args.test_day
        test_num = io_args.test_num

        
        dataset = read_dataset("phase1_training_data.csv")
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
        dataset_CA=dataset[dataset[ct_key]=='CA']
        cases_CA = dataset_CA["cases"]
        cases14_CA =dataset_CA["cases_14_100k"]
        day_n = len(cases_CA)
        
        exp_fea_name = "cases_100k"
        ref_fea_name = "cases_100k"
        #exp_dataset = np.zeros((n_ct,day_n))
        exp_ct_name ='CA'
        #ref_ct_name ='UK'
        
        #Find nearest neighbors of CA,return indices, get a combined all neighbor country dataset
        # n_neigh = 100
    #N_Neighbor Loop to find N optimal
        
        n_neigh_max = 60
        n_neigh_min = 1
        cutoff = 0.9
        if test_num != None:
            test_num = int(test_num)
            n_neigh_max=test_num
        if test_par != None:               
            test_par = float(test_par)
            cutoff = test_par
        
        test_ratio=11/day_n
        if test_day != None:               
            test_day = float(test_day)
            test_ratio = test_day
            
        test_day = int(day_n*test_ratio)
        train_day = day_n-test_day
        neigh_final_rmse=np.zeros(n_neigh_max+1)
        train_parallel_rmse = np.zeros(n_neigh_max+1)
        report_rmse = np.zeros(n_neigh_max+1)
        for n in range(n_neigh_min,n_neigh_max+1):
            n_neigh = n
    
            n_neigh, n_neigh_indices, n_neigh_distances,n_neigh_lags, n_neigh_dict,n_neigh_dict_lags = country.select_country(
                dataset,exp_fea_name,exp_ct_name,n_neigh=n,
                method="cross",cutoff=cutoff,R=30,ref_fea_name=ref_fea_name)
            
            #print(n_neigh_dict)
            
            #ref_dataset=country.combine_ct(dataset,ref_fea_name,n_neigh_indices)
            ref_dataset=country.crosscombine_ct(dataset,ref_fea_name,n_neigh_indices,n_neigh_lags)
            exp_dataset= dataset[dataset[ct_key]==exp_ct_name][exp_fea_name]
            
            X=ref_dataset[:train_day]
            y=utils.s2v(exp_dataset[:train_day])
            Xtest=ref_dataset[train_day:]
            ytest=utils.s2v(exp_dataset[train_day:])
            
            # print("y is '%s' from country '%s'"%(exp_fea_name,exp_ct_name))
            # print("N_neighbor = %d with cutoff = %.3f"%(n_neigh,cutoff))
            parallel_model = linear_model.LeastSquaresPoly(p=1)
            parallel_model.fit(X,y)
            parallel_predictions=parallel_model.predict(X)
            train_parallel_rmse[n] = sqrt(mean_squared_error(parallel_predictions,y))
            print("MultiLinear train Process rmse is %.3f"%train_parallel_rmse[n])
            #optimal lags is 2 as simplist optimal lag
            #optimal lags is 12 as second best lag
            k=2
            #Implement auto_regression for all neighbors for feature "cases_100k"
            test_rmse_ct = np.zeros(n_neigh)
            train_rmse_ct = np.zeros(n_neigh)
            test_predictions = np.zeros((n_neigh,test_day))
            for nct in range(n_neigh):
                dataset_ct = dataset[dataset[ct_key]==ct_inv_mapper[n_neigh_indices[nct]]]
                X = dataset_ct[ref_fea_name]
                X = X.reset_index(drop = True)
                train, test = X[:(len(X)-test_day)], X[(len(X)-test_day):]
    
                auto_model=AutoReg1(lags=k)
                auto_model.fit(train)
                train_predictions = auto_model.predict(start=k,end=len(train)-1)
                train_rmse_ct[nct] = sqrt(mean_squared_error(train[k:], train_predictions))
                test_predictions[nct] = auto_model.predict(start=len(train), end=len(train)+len(test)-1)
                test_rmse_ct[nct] = sqrt(mean_squared_error(test, test_predictions[nct]))
    
            test_auto_rmse = np.mean(test_rmse_ct)
            train_auto_rmse = np.mean(train_rmse_ct)
            auto_predictions = test_predictions.T
            
            final_predictions = parallel_model.predict(auto_predictions)
            test_exp_dataset = exp_dataset[train_day:]
            final_rmse = sqrt(mean_squared_error(test_exp_dataset, final_predictions))
            baseline = sqrt(mean_squared_error(test_exp_dataset, np.zeros(test_day)))
            
            
            # print("AutoReg train Process  mean_rmse is %.3f"%train_auto_rmse)
            # print("AutoReg test Process(validation process)  mean_rmse is %.3f"%test_auto_rmse)
            print("N_neighbor= %d, Final test rmse is %.3f"%(n_neigh,final_rmse))
            # print("Final test percentage error is %.3f%%"%(final_rmse/baseline*100))
            neigh_final_rmse[n]=final_rmse
            # edit to report training phase error or test phase error
            #report_rmse[n]=train_parallel_rmse[n]
            report_rmse[n]=neigh_final_rmse[n]
        plt.plot(np.arange(1,n_neigh_max+1),report_rmse[1:n_neigh_max+1],marker="*")
        #plt.plot(np.arange(1,n_neigh_max+1),neigh_final_rmse[1:n_neigh_max+1],marker="*")
        #plt.plot(n_neigh_distances)
        plt.xlabel("N_neighbors")
        plt.ylabel("Final rmse")
    
        plotname="Find_N_plot_d"+str(test_day)+"_"+exp_ct_name+".pdf"
        fname = os.path.join("..", "figs", plotname)
        plt.savefig(fname)
        plt.clf()
        plt.close('all')            
    
            
    if question == "AMLR_future":
        warnings.filterwarnings("ignore")
        test_par = io_args.test_par
        test_day = io_args.test_day
        test_num = io_args.test_num

        
        dataset = read_dataset("phase1_training_data.csv")
        dataset = utils.get_all(dataset)
        n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
        
        dataset_CA=dataset[dataset[ct_key]=='CA']
        cases_CA = dataset_CA["cases"]
        cases14_CA =dataset_CA["cases_14_100k"]
        day_n = len(cases_CA)
        
        exp_fea_name = "deaths_100k"
        ref_fea_name = "deaths_100k"

        exp_ct_name ='CA'
        #ref_ct_name ='UK'
        
        #Find nearest neighbors of CA,return indices, get a combined all neighbor country dataset
        n_neigh = 100
        cutoff = 0.9
        if test_num != None:
            test_num = int(test_num)
            n_neigh = test_num
        if test_par != None:               
            test_par = float(test_par)
            cutoff = test_par
        if test_day != None:               
            test_day = int(test_day)
        
        train_day = day_n 
        
        n_neigh, n_neigh_indices, n_neigh_distances,n_neigh_lags, n_neigh_dict,n_neigh_dict_lags = country.select_country(
            dataset,exp_fea_name,exp_ct_name,n_neigh=n_neigh,
            method="cross",cutoff=cutoff,R=30,ref_fea_name=ref_fea_name)
        print("y is '%s' from country '%s'"%(exp_fea_name,exp_ct_name))
        print("N_neighbor = %d with cutoff = %.3f"%(n_neigh,cutoff))
        #print(n_neigh_dict)
        
        #-------------------------------------------------------------------------------------------------
        #   Combine the lags list and modify each country, return a dataset of reference
        #   the reference is used to be a X input matrix
        #   all subset of dataset are shifted based on the best lag according to the cross correlation
        #   we are fitting all the correlated countries with the same time-shift
        #   so that the total linear correlation coefficient is significantly improved
        #   we just considered the case that all countries don't have the same date of outbreak of pandemic
        #   this is an important factor to fit our data
        #--------------------------------------------------------------------------------------------------
        ref_dataset=country.crosscombine_ct(dataset,ref_fea_name,n_neigh_indices,n_neigh_lags)
        exp_dataset= dataset[dataset[ct_key]==exp_ct_name][exp_fea_name]
        exp_dataset=exp_dataset.reset_index(drop = True)
        

 
        
        X=ref_dataset[:train_day]
        y=utils.s2v(exp_dataset[:train_day])
        
        AR_X=exp_dataset[:train_day]
        AR_y=exp_dataset[train_day:]
        
        #k is the lag chosen as parameter in AutoReg model
        #optimal lags is 2 as simplist optimal lag
        k=2
        #Implement auto regression on the self feature of the selected country

        AR_model = AutoReg1(lags=k)
        AR_model.fit(AR_X)
        AR_train_predictions = AR_model.predict(start=k,end=train_day-1)
        AR_test_predictions = AR_model.predict(start=train_day, end=train_day+test_day-1)
        AR_train_rmse = sqrt(mean_squared_error(AR_X[k:], AR_train_predictions))

        #---------------------------------------------------------------------------------------------------
        #Implement the parallel Linear Regression on the neighbors of selected country:
        #
        #   Feature(selected country,past)= LR (Feature(other countries,past) )
        #
        #So we can train a model to fit the past feature of selected country
        #Later, we will use this model to parallelly predict the Future Feature of the spec country:
        #
        #   Feature(selected country,future)= a*LR (Feature(other countries,future) ) + b*AR (Feature(spec country,past)) + c
        #
        #Note that we have already shifted the training set so that all outbreaks happend at the same time
        #Based on this we are able to say, the correlation decides whether we can predict the spec country
        #----------------------------------------------------------------------------------------------------
        parallel_model = linear_model.LeastSquaresPoly(p=1)
        parallel_model.fit(X,y)
        parallel_predictions=parallel_model.predict(X)
        train_parallel_rmse = sqrt(mean_squared_error(parallel_predictions,y))
        print("MultiLinear train Process rmse is %.3f"%train_parallel_rmse)

        #Implement auto_regression for all neighbors for feature "cases_100k" except selected country

        train_rmse_ct = np.zeros(n_neigh)
        test_predictions = np.zeros((n_neigh,test_day))
        for nct in range(n_neigh):
            dataset_ct = dataset[dataset[ct_key]==ct_inv_mapper[n_neigh_indices[nct]]]
            X = dataset_ct[ref_fea_name]
            X = X.reset_index(drop = True)
            
            auto_model=AutoReg1(lags=k)
            auto_model.fit(X)
            train_predictions = auto_model.predict(start=k,end=train_day-1)

            train_rmse_ct[nct] = sqrt(mean_squared_error(X[k:], train_predictions))
            test_predictions[nct] = auto_model.predict(start=train_day, end=train_day+test_day-1)

        
        #Compute all errors happend in the AutoReg process
        #Add up and get the mean value of rmse of each phase of the regression process of each country:

        train_auto_rmse = np.mean(train_rmse_ct)
        
        #This is the matrix of the auto predicted value of all other countries
        #This can be considered as an X of a LR model, then we can predict the future feature of spec country
        auto_predictions = test_predictions.T
        
        #Prediction phase and calculate the final result of the selected feature of the spec country
        MLR_predictions = parallel_model.predict(auto_predictions)

        
#------------------Output Phase-------------------------------------------------------------------------------------------------
        print("AutoReg train Process  mean_rmse is %.3f"%train_auto_rmse)
        
#------------------deaths feature-----------------------------------------------------------------------------------------------           
        if exp_fea_name == ref_fea_name and exp_fea_name =="deaths_100k":
            num_people_100k = utils.get_num_people_100k(dataset,exp_ct_name)
            
            MLR_deaths_predictions = num_people_100k * MLR_predictions
            
            AR_deaths_predictions = num_people_100k * AR_test_predictions

            # predict the deaths by combining two models together : stacked accroding to the correlation coefficients
            
            AMLR_deaths_predictions = utils.estimator(MLR_deaths_predictions.T[0],AR_deaths_predictions,n_neigh_distances)
            with open(os.path.join("..", "data", "sample_phase1_submission_origin.csv"), "rb") as f:
                sample_submission_dataset=pd.read_csv(f)
                sample_submission_dataset["deaths"]=AMLR_deaths_predictions
            sample_submission_dataset.to_csv(r'..\data\sample_phase1_submission.csv', index=False, header=True)
                #TrainingSet.to_csv(r'.\data\Modified_data.csv', index=False, header=True)
            
            
            
            # y_test = np.asarray(y_test)
            # y_pred = AMLR_deaths_predictions 
            # utils.predictions_plot(y_pred,y_test,"CA","deaths","Test")



#-------------------AMLR model training and prediction-----------------Completed--------------------------------------------------------------
        
        
