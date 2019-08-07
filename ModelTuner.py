
# coding: utf-8



from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, BaggingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, cross_val_predict, ShuffleSplit
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matminer.data_retrieval import retrieve_MDF
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matminer.datasets import load_dataset
import numpy as np
import pandas as pd
import math
import os

#build the class for tunning model

class ModelTuner:
    '''Determines the best model parameters given a certain dataset'''

    def __init__(self, model, param_list):
        ''' Args:
            model (BaseEstimator): Model to be trained
            param_list ([(str, list)]): List of parameters to be tuned and the ranges of acceptable values
        '''
        self.model = model
        self.param_list = param_list
        self.model.get_params()

    def tune(self, X, y):
        '''Given a dataset, return a tuned model
        Args:
        X (ndarray): List of features for each entry
        y (ndarray): List of labels for each entry
        Returns:
        - (BaseEstimator): Model with the tuned hyperparameters
        - (dict): Value of the parameters that were tuned for the best estimator'''
        # Clone model template
        model = clone(self.model)
        # Loop over all parameters to tune
        best_params = {}
        for param_name, param_values in self.param_list:
            # Create the grid search for tuning that parameter
            gs = GridSearchCV(model, {param_name: param_values}, cv=ShuffleSplit(n_splits=1, test_size=0.1))
            # Run the tuning
            gs.fit(X, y)
            # Get the best estimator
            model = gs.best_estimator_
            # Store the tuned values
            best_params.update(gs.best_params_)

        return model, best_params

#build the function for making plots for regression strategies

def rresult(name,y,model,recordfile):
    '''make a plot for the measured values and predicted values
        Args:
        name (str): Property to be predicted
        y (ndarray): List of labels for each entry
        model ([[list,float]]): preditcion value and mean absolute error for each model
        recordfile: file name to store the results of the model
    ''' 
    i = math.ceil(len(recordfile)/2)
    index = 0
    #make plots
    fig, ax = plt.subplots(i, 2, sharey=True, sharex=True,figsize=(15,i*5))
    ax[0,0].set_ylabel("Predicted " + name, fontsize=i*10,y=0)
    ax[i-1,1].set_xlabel("Calculated " + name, fontsize=i*10,x=0)
    for x_axis in range(i):
        for y_axis in range(2):
            ax[x_axis,y_axis].set_title(recordfile[index].split('.')[0])
            ax[x_axis,y_axis].scatter(y, model[index][0], marker='.', color='r')
            ax[x_axis,y_axis].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
            ax[x_axis,y_axis].text(0.70, 0.03, 'MAE: {:.5f} '.format(model[index][1]), transform=ax[x_axis,y_axis].transAxes, fontsize=15,bbox={'facecolor': 'w', 'edgecolor': 'k'})
            index += 1

#build the function for displaying the accuracy of the classification strategies 

def cresult(name,x,y,model):
    
    '''calculate the accuracy of a classification model
    
        Args:
        name (str): Property to be predicted
        x (ndarray): List of features for each entry
        y (ndarray): List of labels for each entry
        model (BaseEstimator): model has been tuned which has the best params
        
        Returns:
        - (float): value of the accuracy
    
    '''
    # calculate the MSE with 10-fold cross validation
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, x, y, cv=crossvalidation)
    # get the MAE
    mse_scores = [abs(s) for s in scores]
    e =  np.mean(mse_scores)
    print(name,e)
    return e

#build the function for transforming the raw data to composition based features (Ward et al 2016) and tuning the model

def test(df,prediction,attribute,regression,preprocess,model_tuner,recordfile,dataname):
    '''do some tests with different datasets and different machine learning methods
    
        Args:
        df (dataframe): Dataset to be trained
        prediction (str): Property to be predicted
        attribute (str): Features to be trained
        regression (bool): Whether or not to figure a regression problem
        preprocess (bool): Whether or not to do preprocess on the values of the predicted property
        model_tuner ([ModelTuner]): Model to be trained
        recordfile (list): List of str contains the name of record files
        dataname (str): The name of the dataset
    
    '''
    #list storing the best model, prediction value and performances
    model_performance = []
    #get the data we need
    data = df[[prediction, attribute]]
    #rename the columns
    data = data.rename(columns={prediction:prediction,  attribute:'composition'})
    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'composition')
    #remove the entry of which the value of property is None
    data = data[data[prediction]!='None']
    for k in [prediction]:
        data[k] = pd.to_numeric(data[k])

    original_count = len(data)
    #Remove entries with NaN or infinite property
    data = data[~ data[prediction].isnull()]
    print('Removed %d/%d entries with NaN or infinite property'%(original_count - len(data), original_count))

    #Get only the groundstate and each composition
    original_count = len(data)
    data['composition'] = data['composition_obj'].apply(lambda x: x.reduced_formula)
    data.sort_values(prediction, ascending=True, inplace=True)
    data.drop_duplicates('composition', keep='first', inplace=True)
    print('Removed %d/%d duplicate entries '%(original_count - len(data), original_count))
    
    #convert the raw materials data into the required input for an ML model: a finite list of quantitative attributes. 
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    feature_labels = feature_calculators.feature_labels()

    #Compute the features
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj')

    print('Generated %d features'%len(feature_labels))
    print('Training set size:', 'x'.join([str(x) for x in data[feature_labels].shape]))

    original_count = len(data)
    #Remove entries with NaN or infinite features
    data = data[~ data[feature_labels].isnull().any(axis=1)]
    print('Removed %d/%d entries with NaN or infinite features'%(original_count - len(data), original_count))
    #if the value of property need preprocessing use log() to transform the original values 
    if preprocess:
        data.loc[:,prediction] =  np.log(df.loc[:,prediction]-2*min(df.loc[:,prediction])) 
    #Remove entries with NaN or infinite property
    data = data[~ data[prediction].isnull()]
    print('Removed %d/%d entries with NaN or infinite property'%(original_count - len(data), original_count))
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    #train different models
    for i in range(len(model_tuner)): 
        model, best_params = model_tuner[i].tune(data[feature_labels],data[prediction])
        if regression == True:
            # calculate the MSE with 10-fold cross validation
            cv_prediction = cross_val_predict(model, data[feature_labels], data[prediction], cv=crossvalidation)
            # get the MAE
            score = getattr(metrics,'mean_absolute_error')(data[prediction], cv_prediction)
            #score = rresult(prediction,data[feature_labels], data[prediction],model)
            best_params['MAE'] = score
            model_performance.append([cv_prediction,score])
        else:
            score = cresult(prediction,data[feature_labels], data[prediction],model)
            best_params['Accuracy'] = score
        #save the best params for each model to the disk
        if os.path.exists(recordfile[i]):
            (pd.DataFrame(best_params,index = [dataname+'/'+prediction] )).to_csv(recordfile[i],mode='a',header=False)
        else:
            (pd.DataFrame(best_params,index = [dataname+'/'+prediction] )).to_csv(recordfile[i],mode='a',header=True)
    #if a regression problem we need to do a linear regression
    if regression == True:
        #score = rresult(prediction,data[feature_labels], data[prediction],LinearRegression())
        # calculate the MSE with 10-fold cross validation
        cv_prediction = cross_val_predict(LinearRegression(), data[feature_labels], data[prediction], cv=crossvalidation)
        # get the MAE
        score = getattr(metrics,'mean_absolute_error')( data[prediction], cv_prediction)
        model_performance.append([cv_prediction,score])
        if os.path.exists(recordfile[3]):
            (pd.DataFrame([[score]],columns = ['MAE'],index = [dataname+'/'+prediction] )).to_csv(recordfile[3],mode='a',header=False)
        else:
            (pd.DataFrame([[score]],columns = ['MAE'],index = [dataname+'/'+prediction] )).to_csv(recordfile[3],mode='a',header=True)
        rresult(prediction, data[prediction],model_performance,recordfile)
        

#build the function for comparing different models

def compare(dataset,regression,file):
    result = pd.DataFrame()
    for f in file:
        record = pd.read_csv(f)
        temp = pd.DataFrame()
        index = []
        for i in range(len(record)):
            #find the record of the specific dataset
            if str(record.iat[i,0].split('/')[0]) == dataset:
                #find the performance of the specific dataset
                temp = pd.concat([temp,pd.DataFrame([record.iat[i,-1]],columns = [str(f.split('.')[:-1])])])
                index.append(record.iat[i,0])
        result = pd.concat([result,temp],axis=1)
    temp = pd.DataFrame()
    for i in range(len(result)):
        performance = result.iloc[i,:]
        #find the model of the best performance
        if regression == True:
            best_model = performance[performance == performance.min()].index.format()[0]
        else:
            best_model = performance[performance == performance.max()].index.format()[0]
        temp = pd.concat([temp,pd.DataFrame([[best_model]],columns=['Best_Model'])])
    result = pd.concat([result,temp],axis=1)
    result.index = index
    #save the result to the disk
    (result).to_csv(dataset+'.csv',header=True)
    return(result)

