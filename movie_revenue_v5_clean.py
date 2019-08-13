# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:51:43 2019

@author: nmadapati
"""

import pandas as  pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression , ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import datetime
import json
from sklearn.decomposition import PCA
from dateutil.parser import parse

##########################################################
## Checking for Null
##########################################################
def is_nan(x):
    return (x is np.nan or x != x)


##########################################################
## Extract month from release date
##########################################################

    
def releasedate(data):
    if is_nan(data) :
        pass
    else :
        dt = parse(data)
        return  dt.strftime("%m") 

        
##########################################################
## PCA of the one-hot encoded features 
##########################################################
        
    
def PCA_module(data, column_list, no_column ):
    pca = PCA(n_components =no_column)
    pca.fit(data)
    pc12 = pca.transform(data)
    dframe_pc12 = pd.DataFrame(pc12, columns = column_list)
    return dframe_pc12

##########################################################
## Converting Column List into seperate Columns 
##########################################################
    
def prepare_col_from_data_list(data, index, output_col):
    for item in data :
        output_col.loc[index, item] = 1
        
 

##########################################################
## Parsing Json Object into Strings 
##########################################################
        
def json_parser(data):
    if pd.isna(data) :
        return "No data"
    else :
        data = data.replace("\'", "\"")
        try:
            j1 =  json.loads(data)
            return j1
        except Exception as e:
            return "No data"
        
      

##########################################################
## Generating list from a  Json string for specific key
##########################################################      
def extract_list_data(data, data_type):
    return (list(d[data_type] for d in data if data_type in d))


##########################################################
## Set language column values - more like one-hot encoder
########################################################## 
            
def set_col_from_lang_list(data, index, df_data):
    lg_1 = ['en','fr', 'es', 'de', 'it', 'ru', 'ja', 'zh', 'pt','hi','ar', 'cn', 'ko' ]
   
   
    for item in data :
        if item in lg_1 :
            df_data.loc[index, 'lg_1'] = 1
        else :
            df_data.loc[index, 'lg_other'] = 1
            
##########################################################
## Compare differnt  Regression Model
##########################################################             
def different_regression_model() :
    names = ["Linear", "ElasticNet", 
             "Lasso", "Ridge"]

    classifiers = [
                        LinearRegression(normalize=True),
                        ElasticNet(normalize=True),
                        Lasso(normalize=True), 
                        Ridge(normalize=True)
                        
                  ]
    anova_filter = SelectKBest(f_regression, k='all')
        
    for name, clf in zip(names, classifiers):
        estimator = make_pipeline(anova_filter, clf)
        estimator.fit(X_train, y_train)
        print("Model name :", clf)
        predit = estimator.predict(X_test)

        print("R^2 sqaure ", metrics.r2_score(y_test,predit))
        
        print('MAE:', metrics.mean_absolute_error(y_test,predit))
        print('MSE:', metrics.mean_squared_error(y_test,predit))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predit)))
        
##########################################################
## Final model for Linear Regression 
########################################################## 
        
def final_model() :
    
     anova_filter = SelectKBest(f_regression, k='all')
     lm = Lasso(normalize=True)
        
     estimator = make_pipeline(anova_filter, lm)
     estimator.fit(X_train, y_train)
       
     predit = estimator.predict(X_test)
     df_predit = pd.DataFrame(predit)
     print(df_predit.head(), y_test.head())
     print("R^2 sqaure ", metrics.r2_score(y_test,predit))
    
     print('MAE:', metrics.mean_absolute_error(y_test,predit))
     print('MSE:', metrics.mean_squared_error(y_test,predit))
     print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predit)))
     
     return estimator
     
#############################################################################
## covert list into one-hot encoding 
#############################################################################
def one_hot_encoding(data) :
    
    pca_list = ['production_companies', 'Keywords', 'cast', 
                    'crew', 'spoken_languages', ]
    df_pca_data = data[pca_list].copy()
    
    data['release_month'] = data['release_date'].apply(releasedate)
    
    for row in data.itertuples():
        prepare_col_from_data_list(data['genres'][row.Index],row.Index, data )
        set_col_from_lang_list(data['spoken_languages'][row.Index],row.Index, data)
        prepare_col_from_data_list(data['production_companies'][row.Index],row.Index,
                                   df_pca_data)
        prepare_col_from_data_list(data['Keywords'][row.Index],row.Index, df_pca_data)
        prepare_col_from_data_list(data['cast'][row.Index],row.Index, df_pca_data)
        prepare_col_from_data_list(data['crew'][row.Index],row.Index, df_pca_data)
        
    
    df_pca_data.replace(np.nan, 0, inplace=True)
    df_pca_data.drop(pca_list, axis=1, inplace= True)
    return df_pca_data, data
    
    

##########################################################
## Extract data from the json string  
##########################################################  
    
def extract_data(data) :
       
    extract_id_list = ['genres','Keywords','crew', 'cast']
    
    for item in extract_id_list :
        data[item]= data[item].apply(extract_list_data,args=['name'])
        
    data['production_companies']= data['production_companies'].apply(extract_list_data,args=['id'])
    data['spoken_languages']= data['spoken_languages'].apply(extract_list_data,args=['iso_639_1'])
    data['production_countries']= data['production_countries'].apply(extract_list_data,args=['iso_3166_1'])
    
    return data


##########################################################
## Feature data is parsed using json_parser 
##########################################################  
    
def parse_data(data) : 
    json_parser_list =[ 'genres','spoken_languages', 'Keywords',  
                       'production_companies',  'production_countries',
                       'crew', 'cast']
    
    for item in json_parser_list :
         data[item] = data[item].apply(json_parser)
         
    return data

##########################################################
##  Data normalization 
#########################################################
def data_normalization(data, revenue_val_bn) :
    amount_mean = data['budget'].mean()

    data['budget'] = data['budget'].apply(lambda x: amount_mean if x == 0 else x)
    min = data['budget'].min()
    max= data['budget'].max()
    data['budget_mod'] = ((data['budget'] - min)/ (max - min))
    
    if revenue_val_bn :
        min_rev = data['revenue'].min()
        max_rev = data['revenue'].max()
        data['revenue_mod'] = ((data['revenue'] - min_rev)/ (max_rev - min_rev))
    return data 
    
##########################################################
## process raw input data  and created required feilds  
########################################################## 
    
def data_preparation(data, depend_val_bn):
    n_data = data_normalization(data, depend_val_bn)
    
    df_parse = parse_data(n_data)
    
    
    df_extract = extract_data(df_parse)
    
    
    df_pca, df_modified = one_hot_encoding(df_extract)
    
    prod_col_list= ['PC_PROD1', 'PC_PROD2', 'PC_PROD3', 'PC_PROD5', 'PC_PROD6']
    
    prod_pca = PCA_module(df_pca, prod_col_list, 5)
  
    data.replace(np.nan, 0, inplace=True)
    
    drop_items = ['belongs_to_collection', 'homepage', 'genres','imdb_id',
                  'production_companies', 'production_countries',
                  'original_title', 'overview','poster_path',
                  'spoken_languages', 'Keywords', 'status',
                  'tagline','title','cast','crew',
                  'release_date', 'original_language','id','budget_mod']
                   
        
    data.drop(drop_items, axis=1, inplace= True)
    
    if depend_val_bn :
        drop_items = ['revenue_mod']
        data.drop(drop_items, axis=1, inplace= True)
    
    
    data = data.reset_index()
    df_final_local = pd.concat([df_modified, prod_pca], axis=1)
    return df_final_local


##########################################################
## Final Submission test data  
##########################################################     
def submission_data() : 
    df_sub = pd.read_csv("test_movie.csv")
    df_sub_final = data_preparation(df_sub, False)
    df_sub_final['TV Movie'] = 0
    predit_revenue = model.predict(df_sub_final)
    df_revenue = pd.DataFrame(predit_revenue)
    print(df_revenue.head())


###################################
# Global data 
(X_train, X_test), (Y_train, Y_test) = (None, None), (None, None)

       
if __name__ == "__main__":
    df = pd.read_csv("train_movie.csv")
    
    
    
    max_revenue = df['revenue'].max()
    min_revenue = df['revenue'].min()
    
    df_final = data_preparation(df, True)
    
    y = df_final['revenue']
    
    drop_items = ['revenue']
    df_final.drop(drop_items, axis=1, inplace= True)
    
    
    X_train, X_test, y_train, y_test = train_test_split(df_final,y,test_size=0.3)
    start_model = datetime.datetime.now()
    
    
    
    model = final_model()





    














