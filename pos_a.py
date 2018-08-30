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


pos_df = pd.read_csv('../data/POS_CASH_balance.csv')
all_application_df = pd.read_csv('application.csv')
aggrated_df = all_application_df[['SK_ID_CURR']]
print(pos_df.shape)

# Engineering
pos_df['[FE_POS]TOTAL_DPD'] = pos_df['SK_DPD_DEF'] + pos_df['SK_DPD']
pos_df['[FE_POS]SAME_DPD'] = (pos_df['SK_DPD_DEF'] == pos_df['SK_DPD']).astype(int)

pos_df['[FE_POS]LATE'] = (pos_df['SK_DPD'] > 0).astype(int)
pos_df['[FE_POS]LATE_T'] = (pos_df['SK_DPD_DEF'] > 0).astype(int)

pos_df.sort_values('MONTHS_BALANCE', ascending=False, inplace=True)


k_configs = [0.1, 0.2, 0.3, 0.4]

# k = 0.4
# gy_cols = ['SK_ID_CURR']
# print(k)
# groupby = pos_df.groupby(gy_cols)
# cuted_first_k_df = groupby.apply(get_first_k(k, gy_cols)).reset_index()
# cuted_last_k_df = groupby.apply(get_last_k(k, gy_cols)).reset_index()
# print(cuted_first_k_df.head(1))
# cuted_first_k_df.to_csv('pos_first_'+str(k)+'.csv', index=False)
# cuted_last_k_df.to_csv('pos_last_'+str(k)+'.csv', index=False)
# exit()



numerical_features_config = [
        ('[FE_POS]TOTAL_DPD', ['mean', 'max', ]),
        ('[FE_POS]SAME_DPD', ['mean' ]), 
        ('[FE_POS]LATE', ['mean' ]),
        ('[FE_POS]LATE_T', ['mean']),
        ('SK_DPD', ['trend', 'mean', 'max', ]),
        ('SK_DPD_DEF', ['trend', 'mean', 'max', ]), 
        ('CNT_INSTALMENT_FUTURE', ['trend', 'mean', 'max', ]),   
        ('CNT_INSTALMENT', [ 'mean', 'max', ]),   
    ]


temp_list = []
for k in k_configs:
    print(k)
    cuted_first_k_df = (pd.read_csv('pos_first_'+str(k)+'.csv'))
    cuted_last_k_df = (pd.read_csv('pos_last_'+str(k)+'.csv'))
    for n_c in numerical_features_config:
        # print('', n_c[0])
        for agg in n_c[1]:
            # print('', '', agg)
            new_name = n_c[0]+'_'+agg+'_LAST_'+str(k)
            if agg == 'mean':
                t = cuted_first_k_df.groupby(['SK_ID_CURR'])[n_c[0]].mean()
            if agg == 'max':
                t = cuted_first_k_df.groupby(['SK_ID_CURR'])[n_c[0]].max()
            if agg == 'min':
                t = cuted_first_k_df.groupby(['SK_ID_CURR'])[n_c[0]].min()
            if agg == 'sum':
                t = cuted_first_k_df.groupby(['SK_ID_CURR'])[n_c[0]].sum()
            if agg == 'trend':
                continue
                # t = cuted_first_k_df.groupby(['SK_ID_CURR'])[n_c[0]].agg(get_trend)
            # print('', '', '', 'agg done')
            t = t.reset_index().rename(columns={n_c[0]:new_name})
            temp_list.append(t[['SK_ID_CURR', new_name]])

            new_name = n_c[0]+'_'+agg+'_FIRST_'+str(k)
            if agg == 'mean':
                t = cuted_last_k_df.groupby(['SK_ID_CURR'])[n_c[0]].mean()
            if agg == 'max':
                t = cuted_last_k_df.groupby(['SK_ID_CURR'])[n_c[0]].max()
            if agg == 'min':
                t = cuted_last_k_df.groupby(['SK_ID_CURR'])[n_c[0]].min()
            if agg == 'sum':
                t = cuted_last_k_df.groupby(['SK_ID_CURR'])[n_c[0]].sum()
            if agg == 'trend':
                t = cuted_last_k_df.groupby(['SK_ID_CURR'])[n_c[0]].agg(get_trend)
            # print('', '', '', 'agg done')
            t = t.reset_index().rename(columns={n_c[0]:new_name})
            temp_list.append(t[['SK_ID_CURR', new_name]])

for t in tqdm(temp_list):
    aggrated_df = pd.merge(aggrated_df, t, how='left', on=[ 'SK_ID_CURR'])  


features = ['SK_DPD', 'SK_DPD_DEF', 'CNT_INSTALMENT_FUTURE']

# Trending Features (group by previous application)
trends_df = pos_df[['SK_ID_PREV', 'SK_ID_CURR']]
trends_df = trends_df.drop_duplicates()

print('Getting Trends')
time_range = [-3, -6, -18]
ts = []
for time in time_range:
    df = pos_df[pos_df['MONTHS_BALANCE'] >= time]
    print(time, df.shape)
    groupby = df.groupby(['SK_ID_PREV', 'SK_ID_CURR'])
    for feature in tqdm(features):
        ts.append(groupby[feature].agg(get_trend).reset_index().rename(columns={feature:feature+'_TREND_'+str(time)}))
        ts.append(groupby[feature].agg('skew').reset_index().rename(columns={feature:feature+'_SKEW_'+str(time)}))

for t in tqdm(ts):
    trends_df = pd.merge(trends_df, t, how='left', on=['SK_ID_PREV', 'SK_ID_CURR'])  

ts = []
for time in time_range:
    for feature in tqdm(features):
        ts += (agg_num(trends_df.groupby('SK_ID_CURR'), 'POS', feature+'_TREND_'+str(time), ['mean', 'max', 'min']))
        ts += (agg_num(trends_df.groupby('SK_ID_CURR'), 'POS', feature+'_SKEW_'+str(time), ['mean', 'max', 'min']))

for t in tqdm(ts):
    aggrated_df = pd.merge(aggrated_df, t, how='left', on='SK_ID_CURR')  



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

categorical_features= ['EMERGENCYSTATE_MODE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 'REGION_RATING_CLIENT', 'WALLSMATERIAL_MODE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'FONDKAPREMONT_MODE', 'FLAG_OWN_CAR', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE', 'WEEKDAY_APPR_PROCESS_START']

path = 'result/allF_'
a, b, c, feature_importances = validate_model_lgb(training_pdf, categorical_features, params, path, 5, 850, 10)
json.dump(sorted(feature_importances, key=lambda x: x[1], reverse=True), open('sorted_features.json', 'w'))

for k, v in sorted(feature_importances, key=lambda x: x[1], reverse=True):
    if v == 0 and k in aggrated_df.columns:
        aggrated_df.drop(columns=[k], inplace=True)

print(aggrated_df.shape)
aggrated_df.to_csv('pos_a.csv', index=False)


# A only
# Fold 0.810853003543331 0.7726484129577031 0.03820459058562786 1538
# Fold 0.8109052292819253 0.7734079301880769 0.03749729909384847 1546
# Fold 0.8169927076257599 0.7769010449911502 0.040091662634609726 1912
# Fold 0.822863863109937 0.776841758806151 0.04602210430378595 2300
# Fold 0.8158522495909742 0.7774895392853891 0.038362710305585135 1854
#          AUC mean: 0.7754577372456941 std: 0.0020110138927896728 min: 0.7726484129577031 max: 0.7774895392853891
#          OOF AUC: 0.7754323434199931

# Fold 0.8180796397518181 0.776913850631594 0.04116578912022406 1642
# Fold 0.8307162089131792 0.7751810580745802 0.05553515083859906 2405
# Fold 0.8199063549512093 0.7801966721754513 0.03970968277575804 1787
# Fold 0.8168899908101386 0.7799066714610033 0.036983319349135235 1663
# Fold 0.8159936301030704 0.7821652392757048 0.03382839082736555 1609
#          AUC mean: 0.7788726983236668 std: 0.002495120243162857 min: 0.7751810580745802 max: 0.7821652392757048
#          OOF AUC: 0.7788123911236504
# Save submission to result/allF_0.77887CV_SUB.csv
# it took 1663.1993441581726 seconds.