from sklearn.externals import joblib
import pandas as pd
from tqdm import tqdm
import numpy as np
from lightgbm import LGBMClassifier, plot_importance, plot_metric, plot_tree, LGBMRanker
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import itertools
import gc
import collections
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from utils import *

previous_application_df = pd.read_csv('../data/previous_application.csv')
all_application_df = pd.read_csv('application.csv')
aggrated_df = all_application_df[['SK_ID_CURR']]

# for idx, df in enumerate(filted_dfs):
#     for f in fs:
#         # mean
#         mean_df = df.groupby('SK_ID_CURR')[f].mean().reset_index().rename(columns={f: 'MEAN_'+f})
#         aggrated_df = pd.merge(aggrated_df, mean_df, how='left', on='SK_ID_CURR')
#         # range
#         max_df = df.groupby('SK_ID_CURR')[f].max().reset_index().rename(columns={f: 'MAX_'+f})
#         min_df = df.groupby('SK_ID_CURR')[f].min().reset_index().rename(columns={f: 'MIN_'+f})
#         aggrated_df = pd.merge(aggrated_df, max_df, how='left', on='SK_ID_CURR')
#         aggrated_df = pd.merge(aggrated_df, min_df, how='left', on='SK_ID_CURR')
#         aggrated_df['AMT_IN_PREV_RANGE_'+f+'_DF_'+str(idx)] = ((aggrated_df[f] <= aggrated_df['MAX_'+f]) & (aggrated_df[f] >= aggrated_df['MIN_'+f])).astype(int)
#         aggrated_df['AMT_LESS_PREV_MAX_'+f+'_DF_'+str(idx)] = (aggrated_df[f] <= aggrated_df['MAX_'+f]).astype(int)
#         aggrated_df['AMT_LARGE_PREV_MIN_'+f+'_DF_'+str(idx)] = (aggrated_df[f] >= aggrated_df['MIN_'+f]).astype(int)

#         aggrated_df.drop(columns=['MEAN_'+f, 'MIN_'+f, 'MAX_'+f], inplace=True)
#         print(aggrated_df[[ 'AMT_IN_PREV_RANGE_'+f+'_DF_'+str(idx), 'AMT_LESS_PREV_MAX_'+f+'_DF_'+str(idx), 'AMT_LARGE_PREV_MIN_'+f+'_DF_'+str(idx)]].head(5))

#     # loan duration comparsion
#     mean_df = df.groupby('SK_ID_CURR')['FE_1'].mean().reset_index().rename(columns={'FE_1': 'MEAN_FE_1'})
#     max_df = df.groupby('SK_ID_CURR')['FE_1'].max().reset_index().rename(columns={'FE_1': 'MAX_FE_1'})
#     min_df = df.groupby('SK_ID_CURR')['FE_1'].min().reset_index().rename(columns={'FE_1': 'MIN_FE_1'})
#     aggrated_df = pd.merge(aggrated_df, mean_df, how='left', on='SK_ID_CURR')
#     aggrated_df = pd.merge(aggrated_df, max_df, how='left', on='SK_ID_CURR')
#     aggrated_df = pd.merge(aggrated_df, min_df, how='left', on='SK_ID_CURR')
#     aggrated_df['AMT_IN_PREV_RANGE_FE_1'+'_DF_'+str(idx)] = ((all_application_df['[FE_APP]LOAN_DURATION'] <= aggrated_df['MAX_FE_1']) & (all_application_df['[FE_APP]LOAN_DURATION'] >= aggrated_df['MIN_FE_1'])).astype(int)
#     aggrated_df['AMT_LESS_PREV_MAX_FE_1'+'_DF_'+str(idx)] = (all_application_df['[FE_APP]LOAN_DURATION'] <= aggrated_df['MAX_FE_1']).astype(int)
#     aggrated_df['AMT_LARGE_PREV_MIN_FE_1'+'_DF_'+str(idx)] = (all_application_df['[FE_APP]LOAN_DURATION'] >= aggrated_df['MIN_FE_1']).astype(int)
#     aggrated_df.drop(columns=[ 'MIN_FE_1', 'MAX_FE_1'], inplace=True)



previous_application_df['NAME_CLIENT_TYPE'].replace('XNA', 'New')

previous_application_df['NAME_CONTRACT_TYPE'].replace('XNA', 'Cash loans')


# print(previous_application_df['NAME_CONTRACT_STATUS'].value_counts())
# print(previous_application_df['NAME_CASH_LOAN_PURPOSE'].value_counts())
# print(previous_application_df['NAME_CONTRACT_TYPE'].value_counts())
# print(previous_application_df['NAME_CLIENT_TYPE'].value_counts())





previous_application_df = pd.merge(previous_application_df, all_application_df[['SK_ID_CURR', 'DAYS_BIRTH', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']]\
    .rename(columns={
        'AMT_CREDIT': 'APP_AMT_CREDIT',
        'AMT_ANNUITY': 'APP_AMT_ANNUITY',
        'AMT_GOODS_PRICE': 'APP_AMT_GOODS_PRICE',
        'AMT_INCOME_TOTAL': 'APP_AMT_INCOME_TOTAL',
        
}), on='SK_ID_CURR', how='left')


previous_application_df['FE_CNT_PAYMENT_DIFF'] = previous_application_df['CNT_PAYMENT'] - (np.round(previous_application_df['AMT_CREDIT'] / previous_application_df['AMT_ANNUITY']))
previous_application_df['FE_AGE_END'] = (previous_application_df['DAYS_TERMINATION'] + previous_application_df['DAYS_BIRTH'].abs())/365
previous_application_df['FE_1'] = previous_application_df['AMT_CREDIT'] / previous_application_df['AMT_ANNUITY']
previous_application_df['FE_2'] = previous_application_df['DAYS_TERMINATION'] - previous_application_df['DAYS_DECISION']
previous_application_df['FE_3'] = previous_application_df['FE_1'] - previous_application_df['FE_2']

previous_application_df['FE_4'] = previous_application_df['APP_AMT_CREDIT'] / (1+previous_application_df['AMT_CREDIT'])
previous_application_df['FE_5'] = previous_application_df['APP_AMT_ANNUITY'] / (1+previous_application_df['AMT_ANNUITY'])
previous_application_df['FE_6'] = previous_application_df['APP_AMT_GOODS_PRICE'] / (1+previous_application_df['AMT_GOODS_PRICE'])
previous_application_df['FE_7'] = previous_application_df['AMT_ANNUITY'] / (1+previous_application_df['APP_AMT_INCOME_TOTAL'])

previous_application_df['FE_8'] = previous_application_df['RATE_DOWN_PAYMENT'] * previous_application_df['AMT_CREDIT']
previous_application_df['FE_9'] = previous_application_df['FE_8'] - previous_application_df['AMT_DOWN_PAYMENT']

previous_application_df['FE_10'] = previous_application_df['RATE_INTEREST_PRIMARY'] - previous_application_df['RATE_INTEREST_PRIVILEGED']

approved_previous_application_df = previous_application_df[(previous_application_df['NAME_CONTRACT_STATUS']=='Approved')]
refused_previous_application_df = previous_application_df[(previous_application_df['NAME_CONTRACT_STATUS']=='Refused')]
canceled_previous_application_df = previous_application_df[(previous_application_df['NAME_CONTRACT_STATUS']=='Canceled')]
active_previous_application_df = previous_application_df[(previous_application_df['NAME_CONTRACT_STATUS']=='Approved') & (previous_application_df['DAYS_TERMINATION']> 0)]



cash_p_df = previous_application_df[(previous_application_df['NAME_CONTRACT_TYPE']=='Cash loans')]
consumer_p_df = previous_application_df[(previous_application_df['NAME_CONTRACT_TYPE']=='Consumer loans')]
revolving_p_df = previous_application_df[(previous_application_df['NAME_CONTRACT_TYPE']=='Revolving loans')]

repeat_p_df = previous_application_df[(previous_application_df['NAME_CLIENT_TYPE']=='Repeater')]
new_p_df = previous_application_df[(previous_application_df['NAME_CLIENT_TYPE']=='New')]
refreshed_p_df = previous_application_df[(previous_application_df['NAME_CLIENT_TYPE']=='Refreshed')]

division_config = {
    'prefix': 'PREV',
    'count_all_dfs': False,
    'dfs': [previous_application_df,
            approved_previous_application_df,
            active_previous_application_df,
            refused_previous_application_df,
            canceled_previous_application_df,
            cash_p_df,
            consumer_p_df,
            revolving_p_df,
           ],
    'column': 'DAYS_DECISION',
    'groupby_column': 'SK_ID_CURR',
    'groups':[3, 6, 18, 30, 42, 54, 66],
    'step': -30,
    'grouped_dfs': {},
    'grouped_gys': {},
}



numerical_features_config = [
    ('FE_CNT_PAYMENT_DIFF', ['mean', 'max', 'min']),
    ('FE_AGE_END', ['mean', 'max']),
    ('FE_1', ['mean', 'max', 'min']),
    ('FE_2', ['mean', 'max', 'min']),
    ('FE_3', ['mean', 'max', 'min']),
    ('FE_4', ['mean', 'max', 'min']),
    ('FE_5', ['mean', 'max', 'min']),
    ('FE_6', ['mean', 'max', 'min']),
    ('FE_7', ['mean', 'max', 'min']),
    ('FE_8', ['mean', 'max', 'min']),
    ('FE_9', ['mean', 'max', 'min']),
    ('FE_10', ['mean', 'max']),
    ('AMT_ANNUITY',  ['trend', 'mean', 'max']),
    ('AMT_GOODS_PRICE',  ['trend', 'mean', 'max', 'min', 'sum']),
    ('AMT_APPLICATION',  ['trend', 'mean', 'max', 'min', 'sum']),
    ('AMT_CREDIT',  ['trend', 'mean', 'max']),
    ('AMT_DOWN_PAYMENT',  ['mean', 'max', 'min', 'sum']),
    ('RATE_INTEREST_PRIMARY',  ['mean', 'max', 'min']),
    ('RATE_DOWN_PAYMENT',  ['mean', 'max', 'min']),
    ('RATE_INTEREST_PRIVILEGED',  ['mean', 'max', 'min']),
    ('CNT_PAYMENT',  ['mean', 'max', 'sum']),
    ('SELLERPLACE_AREA',  ['max', 'mean']),   
    ('DAYS_DECISION',  ['mean', 'max', 'min']),
    ('DAYS_FIRST_DRAWING',  ['mean', 'max']),
    ('DAYS_FIRST_DUE',  ['mean', 'max', 'min']),
    ('DAYS_LAST_DUE_1ST_VERSION', ['mean', 'max', 'min']),
    ('DAYS_LAST_DUE',  ['mean', 'max', 'min']),
    ('DAYS_TERMINATION',  ['mean', 'max', 'min']),                
 ]


categorical_features_config = [
    'NFLAG_LAST_APPL_IN_DAY',
    'FLAG_LAST_APPL_PER_CONTRACT',
    'NAME_CONTRACT_TYPE',
    'NAME_CASH_LOAN_PURPOSE',
    'NAME_CONTRACT_STATUS',
    'NAME_PAYMENT_TYPE',
    'CODE_REJECT_REASON',
    'NAME_CLIENT_TYPE',
    'NAME_GOODS_CATEGORY',
    'NAME_PORTFOLIO',
    'NAME_PRODUCT_TYPE',
    'CHANNEL_TYPE',
    'NAME_TYPE_SUITE',
    'NAME_SELLER_INDUSTRY',
    'WEEKDAY_APPR_PROCESS_START',
    'HOUR_APPR_PROCESS_START',
    'NFLAG_INSURED_ON_APPROVAL',
    'NAME_YIELD_GROUP',
    'PRODUCT_COMBINATION',
]


# time-based statistic
aggrated_df = divided_statistic(aggrated_df, division_config, numerical_features_config, categorical_features_config)

print(aggrated_df.shape)
training_pdf = pd.merge(all_application_df, aggrated_df, how='left', on='SK_ID_CURR')

params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'n_estimators':20000,
    'learning_rate':0.05,
    'num_leaves': 123, #34,
    'feature_fraction': 1,
    'subsample':0.8715623,
    'max_depth': 3,
    'reg_alpha': 10,
    'reg_lambda':0.0735294,
    'min_child_weight': 20,
    'verbose': -1,
}

categorical_features= ['EMERGENCYSTATE_MODE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 
'REGION_RATING_CLIENT', 'WALLSMATERIAL_MODE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS',
 'FONDKAPREMONT_MODE', 'FLAG_OWN_CAR', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE',
  'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE',
   'HOUSETYPE_MODE', 'WEEKDAY_APPR_PROCESS_START']

path = 'result/allF_'
a, b, c, feature_importances = validate_model_lgb(training_pdf, categorical_features, params, path, 5, 850, 1)

for k, v in sorted(feature_importances, key=lambda x: x[1], reverse=True):
    if v == 0 and k in aggrated_df.columns:
        aggrated_df.drop(columns=[k], inplace=True)

print('with top features', aggrated_df.shape)
training_pdf = pd.merge(all_application_df, aggrated_df, how='left', on='SK_ID_CURR')
a, b, c, feature_importances = validate_model_lgb(training_pdf, categorical_features, params, path, 5, 850, 1)

print(aggrated_df.shape)
aggrated_df.to_csv('prev_application.csv', index=False)

# Fold 0.8267319934644835 0.7825724070759278 0.04415958638855566 1684
# Fold 0.8293963229698258 0.77975088224359 0.04964544072623578 1833
# Fold 0.8320662410108759 0.7822466553598273 0.049819585651048603 1952
# Fold 0.8210969076225565 0.7840392591250342 0.037057648497522244 1474
# Fold 0.8265716512855064 0.7845069509596341 0.042064700325872284 1729
#          AUC mean: 0.7826232309528027 std: 0.001670228499319262 min: 0.77975088224359 max: 0.7845069509596341
#          OOF AUC: 0.7826029712284044

# Fold 0.8333082332469282 0.7825740030223949 0.05073423022453327 1945
# Fold 0.8350596401840196 0.7796080959450113 0.05545154423900822 2055
# Fold 0.8230970974821443 0.7830776914959896 0.04001940598615472 1493
# Fold 0.8221349227033552 0.7841267883622309 0.03800813434112427 1483
# Fold 0.83582507632713 0.784235165018885 0.05158991130824497 2126
#          AUC mean: 0.7827243487689024 std: 0.0016797552882985255 min: 0.7796080959450113 max: 0.784235165018885
#          OOF AUC: 0.7826856533974593