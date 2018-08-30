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
from joblib import Parallel, delayed
import multiprocessing



def enginnering(df):
    # Engineering
    df = pd.merge(df, all_application_df[['SK_ID_CURR', 'DAYS_BIRTH', '[FE_APP]LOAN_DURATION']], on='SK_ID_CURR', how='left')
    df['FE_DIFF_DAY'] = df['DAYS_ENTRY_PAYMENT'].fillna(365243) - df['DAYS_INSTALMENT'].fillna(365243)
    df['FE_DIFF_AMT'] = df['AMT_PAYMENT'].fillna(0) / (1+df['AMT_INSTALMENT'].fillna(0))
    print(df['FE_DIFF_DAY'].describe())
    print(df['FE_DIFF_AMT'].describe())

    # identify early payment
    df['FE_FLAG_EARLY_PAY_1'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x <= -30 else 0)
    df['FE_FLAG_EARLY_PAY_2'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x < -15 and x >= -30 else 0)
    df['FE_FLAG_EARLY_PAY_3'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x < -30 and x >= -45 else 0)
    df['FE_FLAG_EARLY_PAY_4'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x < -45 and x >= -60 else 0)
    df['FE_FLAG_EARLY_PAY_5'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x < -60 else 0)

    df['FE_FLAG_FULL_PAY'] = df['FE_DIFF_AMT'].apply(lambda x: 1 if x >= 0.98 else 0)

    df['FE_FLAG_LATE_PAY_1'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x > 0 and x <= 15 else 0)
    df['FE_FLAG_LATE_PAY_2'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x > 15 and x <= 30 else 0)
    df['FE_FLAG_LATE_PAY_3'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x > 30 and x <= 45 else 0)
    df['FE_FLAG_LATE_PAY_4'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x > 45 and x <= 60 else 0)
    df['FE_FLAG_LATE_PAY_5'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x > 60 and x <= 90 else 0)
    df['FE_FLAG_LATE_PAY_6'] = df['FE_DIFF_DAY'].apply(lambda x: 1 if x > 90 else 0)


    # TODO compare the app loan duration with his pervious loan duration
    
    df.loc[df['FE_DIFF_DAY'] >= 30, 'FE_DPD_30_IN_NUM'] = df.loc[df['FE_DIFF_DAY'] >= 30, 'NUM_INSTALMENT_NUMBER']
    df.loc[df['FE_DIFF_DAY'] >= 90, 'FE_DPD_90_IN_NUM'] = df.loc[df['FE_DIFF_DAY'] >= 90, 'NUM_INSTALMENT_NUMBER']
    df.loc[df['FE_DIFF_DAY'] < 30, 'FE_DPD_30_IN_NUM'] = 365243
    df.loc[df['FE_DIFF_DAY'] < 90, 'FE_DPD_90_IN_NUM'] = 365243
 

    df['FE_APP_PROJECTED_DIFF_DAY'] = df['FE_DIFF_DAY'] * df['[FE_APP]LOAN_DURATION']
    df['FE_APP_PROJECTED_DIFF_AMT'] = df['FE_DIFF_AMT'] * df['[FE_APP]LOAN_DURATION']
    
    return df

# DAYS_ENTRY_PAYMENT
time_feature = 'DAYS_INSTALMENT'
all_application_df = pd.read_csv('application.csv')

print(all_application_df[['[FE_APP]LOAN_DURATION']].head(10))

installments_df = pd.read_csv('../data/installments_payments.csv')
aggrated_df = all_application_df[['SK_ID_CURR', '[FE_APP]LOAN_DURATION']]

installments_df = enginnering(installments_df)
installments_df.sort_values(time_feature, ascending=False, inplace=True)


for i in tqdm([1, 30, 60, 90, 120]):
    bad_installments_df = installments_df[installments_df['FE_DIFF_DAY'] >= i]
    first_bad_installment_df = bad_installments_df.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].min().reset_index().rename(columns={'NUM_INSTALMENT_NUMBER': 'FIRST_'+str(i)+'_DPD'})
    aggrated_df = pd.merge(aggrated_df, first_bad_installment_df, how='left', on='SK_ID_CURR')
    aggrated_df['AMT_APP_LOAN_DURATION_SAFTY_BY_HISTORY_'+str(i)] = aggrated_df['FIRST_'+str(i)+'_DPD'] -  aggrated_df['[FE_APP]LOAN_DURATION']
    aggrated_df['FLAG_APP_LOAN_DURATION_SAFTY_BY_HISTORY_'+str(i)] = (aggrated_df['FIRST_'+str(i)+'_DPD'] > aggrated_df['[FE_APP]LOAN_DURATION']).astype(int) 

# print(aggrated_df.head(100))

aggrated_df.drop(columns=['[FE_APP]LOAN_DURATION'], inplace=True)

# Trending Features (group by previous application)
trends_df = installments_df[['SK_ID_PREV', 'SK_ID_CURR']]
trends_df = trends_df.drop_duplicates()

features = ['FE_DIFF_DAY', 'FE_DIFF_AMT', 'FE_APP_PROJECTED_DIFF_DAY', 'FE_APP_PROJECTED_DIFF_AMT']

time_range = [-3 * 30, -6 * 30, -18 * 30]
ts = []
for time in time_range:
    df = installments_df[installments_df[time_feature] >= time]
    print(time, df.shape)
    groupby = df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    for feature in tqdm(features):
        ts.append(groupby[feature].agg(get_trend).reset_index().rename(columns={feature:feature+'_TREND_'+str(time)}))
        ts.append(groupby[feature].agg('skew').reset_index().rename(columns={feature:feature+'_SKEW_'+str(time)}))

for t in tqdm(ts):
    trends_df = pd.merge(trends_df, t, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])  

ts = []
for time in time_range:
    for feature in tqdm(features):
        ts += (agg_num(trends_df.groupby('SK_ID_CURR'), 'IN', feature+'_TREND_'+str(time), ['mean', 'max', 'min']))
        ts += (agg_num(trends_df.groupby('SK_ID_CURR'), 'IN', feature+'_SKEW_'+str(time), ['mean', 'max', 'min']))

for t in tqdm(ts):
    aggrated_df = pd.merge(aggrated_df, t, how='left', on='SK_ID_CURR')  

print(aggrated_df.head(10))

numerical_features_config = [
        ('FE_DIFF_DAY', ['trend', 'mean', 'max', ]),
        ('FE_DIFF_AMT', ['trend', 'mean', 'max', ]), 
        ('FE_FLAG_FULL_PAY', ['mean', 'sum']), 
        ('FE_FLAG_EARLY_PAY_1', ['mean', 'sum']), 
        ('FE_FLAG_EARLY_PAY_2', ['mean', 'sum']), 
        ('FE_FLAG_EARLY_PAY_3', ['mean', 'sum']), 
        ('FE_FLAG_EARLY_PAY_4', ['mean', 'sum']), 
        ('FE_FLAG_EARLY_PAY_5', ['mean', 'sum']), 
        ('FE_FLAG_LATE_PAY_1', ['mean', 'sum']),
        ('FE_FLAG_LATE_PAY_2', ['mean', 'sum']),
        ('FE_FLAG_LATE_PAY_3', ['mean', 'sum']),
        ('FE_FLAG_LATE_PAY_4', ['mean', 'sum']),
        ('FE_FLAG_LATE_PAY_5', ['mean', 'sum']),
        ('FE_FLAG_LATE_PAY_6', ['mean', 'sum']),
        ('FE_DPD_30_IN_NUM', ['min', 'mean']),
        ('FE_DPD_90_IN_NUM', ['min', 'mean']),
        ('DAYS_ENTRY_PAYMENT', ['mean', 'max', ]),
        ('DAYS_INSTALMENT', ['mean', 'max', ]), 
        ('AMT_PAYMENT', ['mean', 'max', ]),
        ('AMT_INSTALMENT', ['mean', 'max', ]), 
        ('NUM_INSTALMENT_VERSION', ['mean', 'max', ]),
        ('NUM_INSTALMENT_NUMBER', ['mean', 'max', ]),
        ('FE_APP_PROJECTED_DIFF_DAY', ['trend', 'mean', 'max', ]),
        ('FE_APP_PROJECTED_DIFF_AMT', ['trend', 'mean', 'max', ]),
    ]


temp_list = []

k_configs = [0.1, 0.2, 0.3, 0.4]

# k = 0.4
# gy_cols = ['SK_ID_CURR']
# print(k)
# groupby = installments_df.groupby(gy_cols)
# cuted_first_k_df = groupby.apply(get_first_k(k, gy_cols)).reset_index()
# cuted_last_k_df = groupby.apply(get_last_k(k, gy_cols)).reset_index()
# print(cuted_first_k_df.head(1))
# cuted_first_k_df.to_csv('installment_first_'+str(k)+'.csv', index=False)
# cuted_last_k_df.to_csv('installment_last_'+str(k)+'.csv', index=False)

# exit()



for k in k_configs:
    print(k)
    cuted_first_k_df = (pd.read_csv('installment_first_'+str(k)+'.csv'))
    cuted_last_k_df = (pd.read_csv('installment_last_'+str(k)+'.csv'))
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
    'verbose': False,
}

categorical_features= ['EMERGENCYSTATE_MODE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 'REGION_RATING_CLIENT', 'WALLSMATERIAL_MODE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'FONDKAPREMONT_MODE', 'FLAG_OWN_CAR', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE', 'WEEKDAY_APPR_PROCESS_START']

path = 'result/allF_'
a, b, c, feature_importances = validate_model_lgb(training_pdf, categorical_features, params, path, 5, 850, 10)
json.dump(sorted(feature_importances, key=lambda x: x[1], reverse=True), open('sorted_features.json', 'w'))

for k, v in sorted(feature_importances, key=lambda x: x[1], reverse=True):
    if v == 0 and k in aggrated_df.columns:
        aggrated_df.drop(columns=[k], inplace=True)

print(aggrated_df.shape)
aggrated_df.to_csv('installment_a.csv', index=False)

# A only
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Fold 0.8419299450469931 0.7817665859181876 0.06016335912880555 2309
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Fold 0.8442300212493918 0.7799219757183494 0.06430804553104241 2463
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Fold 0.8395351782995318 0.7852256846263873 0.05430949367314453 2232
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Fold 0.8275237853616433 0.7833399906496544 0.044183794711988855 1656
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Fold 0.8297229080897601 0.7848041584973158 0.04491874959244435 1732
#          AUC mean: 0.7830116790819789 std: 0.001965660975462976 min: 0.7799219757183494 max: 0.7852256846263873
#          OOF AUC: 0.7829777128835863
# Save submission to result/allF_0.78301CV_SUB.csv
# it took 3030.5082590579987 seconds.
# (356255, 372)








# Fold 0.8380519151242134 0.7823773453689071 0.05567456975530627 2109
# Fold 0.8323458537395318 0.778983820468094 0.05336203327143785 1820
# Fold 0.8332309761002644 0.7846223990666683 0.04860857703359611 1897
# Fold 0.8367008502094554 0.7843043761715206 0.05239647403793479 2111
# Fold 0.8293232927553662 0.7846592097955227 0.044664082959843476 1710
# 	 AUC mean: 0.7829894301741426 std: 0.0021727121894049525 min: 0.778983820468094 max: 0.7846592097955227
# 	 OOF AUC: 0.7829772593148823
# Save submission to result/allF_0.78299CV_??LB_SUB.csv
# it took 2090.537146806717 seconds.

# Fold 0.8366268987534494 0.7824287897058032 0.054198109047646215 2026
# Fold 0.8389653027417686 0.7788343617128093 0.06013094102895933 2151
# Fold 0.8365231166805929 0.7844233795971841 0.05209973708340876 2039
# Fold 0.8383608647324937 0.7848379304618507 0.05352293427064303 2184
# Fold 0.8352366821565924 0.7850554782000633 0.05018120395652914 2002
# 	 AUC mean: 0.7831159879355422 std: 0.002334118507066982 min: 0.7788343617128093 max: 0.7850554782000633
# 	 OOF AUC: 0.783094105338902

# LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Training until validation scores don't improve for 1000 rounds.
# Early stopping, best iteration is:
# [2124]	training's auc: 0.841526	valid_1's auc: 0.781693
# Fold 0.8415264753855369 0.7816933433749715 0.05983313201056539 2124
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Training until validation scores don't improve for 1000 rounds.
# Early stopping, best iteration is:
# [2049]	training's auc: 0.839577	valid_1's auc: 0.779779
# Fold 0.8395769346286088 0.7797788509876367 0.059798083640972166 2049
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Training until validation scores don't improve for 1000 rounds.
# Early stopping, best iteration is:
# [2043]	training's auc: 0.839234	valid_1's auc: 0.785001
# Fold 0.8392339990784979 0.785001453744138 0.05423254533435995 2043
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Training until validation scores don't improve for 1000 rounds.
# Early stopping, best iteration is:
# [2209]	training's auc: 0.841836	valid_1's auc: 0.783685
# Fold 0.8418358035007061 0.7836847176214006 0.05815108587930551 2209
# [LightGBM] [Warning] feature_fraction is set=1, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.
# Training until validation scores don't improve for 1000 rounds.
# Early stopping, best iteration is:
# [1909]	training's auc: 0.835942	valid_1's auc: 0.786419
# Fold 0.8359424298225221 0.7864192172025553 0.04952321261996684 1909
# 	 AUC mean: 0.7833155165861404 std: 0.0023558905112148914 min: 0.7797788509876367 max: 0.7864192172025553
# 	 OOF AUC: 0.7832904896154461
