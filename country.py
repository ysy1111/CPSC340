import numpy as np
import utils
from scipy.stats import pearsonr
import pandas as pd

def select_country(dataset,exp_fea_name,exp_ct_name,ref_fea_name=None,n_neigh=10,method="corr",cutoff=0.5,itself=False,R=30):
    n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
    nan_num = 0
    if method == "corr" or "cross":
        day_n = len(dataset[dataset[ct_key]=='CA']["cases"])
        if ref_fea_name == None:
            ref_fea_name = exp_fea_name

        exp_dataset = np.zeros((n_ct,day_n))

        #Choose the dataset regarding the name interested
        target= np.zeros((1,day_n))
        target[0] = dataset[dataset[ct_key]==exp_ct_name][exp_fea_name]
        corr=np.zeros(n_ct)
        lags=np.zeros(n_ct)
        indices=np.zeros(n_ct) 
        if method == "corr":
            corr_pair = np.zeros((2,n_ct))
        elif method == "cross":
            corr_pair = np.zeros((3,n_ct))
            
           
        
        #Construct Experiment Dataset (Matrix) =feature(country,day)
        for nct in range(nan_num,n_ct):  #fillter out nan set
            exp_dataset[nct] = dataset[dataset[ct_key]==ct_inv_mapper[nct]][ref_fea_name]
        
        #Compute Corr between N_CT countries    
        if method == "corr":
            for nct in range(nan_num,n_ct):  #fillter out nan set    
                corr[nct]=pearsonr(target[0],exp_dataset[nct])[0]
                indices[nct]=nct
        elif method == "cross":
            for nct in range(nan_num,n_ct):  #fillter out nan set    
                series1=pd.Series(exp_dataset[nct])
                series2=pd.Series(target[0])
                corr[nct],lags[nct]=cross_corr(series1,series2,R)
                indices[nct]=nct
                
        corr_pair[0] = indices
        corr_pair[1] = corr
        if method == "cross":
            corr_pair[2] = lags
        
        #remove the nan set
        #sort the corr_pair
        corr_pair1 = corr_pair.T[1:]
        corr_pair= corr_pair1[np.argsort(corr_pair1[:,1],axis =0)] 
        corr_pair= corr_pair[::-1]
        
        #remove itself from the correlation list
        if itself == False:
            corr_pair = corr_pair[1:]
        #choose the neigh distance vector  
        corr_n_neigh =corr_pair[:,1][:n_neigh]
        n_neigh_distances = corr_n_neigh
        
        if cutoff != None:
            n_neigh_distances =n_neigh_distances[n_neigh_distances >= cutoff]
            
        #update n_neigh after cutoff
        n_neigh = len(n_neigh_distances)
        n_neigh_indices = corr_pair[:,0][:n_neigh]
        if method == "cross":
            n_neigh_lags = corr_pair[:,2][:n_neigh]
        
        #create dict for n neigh countries with descent order of distances
        ct_neigh = [ct_inv_mapper[n_neigh_indices[i]] for i in range(n_neigh)]
        n_neigh_dict=dict(zip(ct_neigh,list(corr_n_neigh)))
        n_neigh_dict_lags=dict(zip(ct_neigh,list(n_neigh_lags)))
        if method == "corr":
            return n_neigh, n_neigh_indices, n_neigh_distances, n_neigh_dict 
        elif method =="cross":
            return n_neigh, n_neigh_indices, n_neigh_distances, n_neigh_lags, n_neigh_dict, n_neigh_dict_lags 


def combine_ct(dataset,fea_name,ind_ct):
    n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
    day_n = len(dataset[dataset[ct_key]=='CA']["cases"])
    nmax = len(ind_ct)
    exp_dataset = np.zeros((nmax,day_n))
    exp_name = fea_name
    for n in range(nmax):
        exp_dataset[n] = dataset[dataset[ct_key]==ct_inv_mapper[ind_ct[n]]][exp_name]
    return exp_dataset.T #dim=(cases14,country) fea=country

def cross_corr(ref_s,exp_s,R):
    
    ref_s1 = ref_s.reset_index(drop = True)
    exp_s1 = exp_s.reset_index(drop = True)
    maxcorr =0.0
    lag_best = 0
    for lag in range(R):
        ref_s1 = ref_s1.shift(lag)
        ref_s1[:lag]=0.0
        corr= pearsonr(ref_s1, exp_s1)[0]
        if corr>maxcorr:
            maxcorr = corr
            lag_best = lag
    return maxcorr,lag_best


def crosscombine_ct(dataset,fea_name,ind_ct,lags):
    n_ct, ct_key, ct_mapper, ct_inv_mapper, ct_ind =utils.make_mapper(dataset)
    day_n = len(dataset[dataset[ct_key]=='CA']["cases"])
    nmax = len(ind_ct)
    exp_dataset = np.zeros((nmax,day_n))
    exp_name = fea_name
    for n in range(nmax):
        exp_dataset[n] = dataset[dataset[ct_key]==ct_inv_mapper[ind_ct[n]]][exp_name]
        exp_dataset[n] = utils.shift_fill0_array(exp_dataset[n],lags[n])
    return exp_dataset.T #dim=(cases14,country) fea=country
      
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    