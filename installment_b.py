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

division_config = {
    'prefix': 'I',
    'count_all_dfs': False,
    'dfs': [installments_df],
    'column': time_feature,
    'groupby_column': 'SK_ID_CURR',
    'groups':[3, 6, 18, 30, 42, 54, 66],
    'step': -30,
    'grouped_dfs': {},
    'grouped_gys': {},
}

categorical_features_config = [] 
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
aggrated_df.to_csv('installment_b.csv', index=False)




# b only
# Fold 0.8388530478868467 0.7809760117490159 0.057877036137830795 2114
# Fold 0.8399195397635448 0.7804604069938532 0.059459132769691614 2206
# Fold 0.8398124090999244 0.7857290543328033 0.05408335476712112 2219
# Fold 0.8389303925082465 0.7839907208321241 0.05493967167612246 2209
# Fold 0.8326409605797909 0.7848687669729282 0.04777219360686269 1854
#          AUC mean: 0.7832049921761449 std: 0.0021098581674270102 min: 0.7804604069938532 max: 0.7857290543328033
#          OOF AUC: 0.7831897168522199








# a+b
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
