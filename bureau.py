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

bureau_balance_df = pd.read_csv('../data/bureau_balance.csv')
bureau_df = pd.read_csv('../data/bureau.csv')



bureau_balance_df['STATUS'] = bureau_balance_df['STATUS'].replace('C', 0)
bureau_balance_df['STATUS'] = bureau_balance_df['STATUS'].replace('X', 0)
bureau_balance_df['STATUS'] = bureau_balance_df['STATUS'].astype(int)



all_application_df = pd.read_csv('application.csv')
aggrated_df = all_application_df[['SK_ID_CURR']]

bureau_df = pd.merge(bureau_df, all_application_df[['SK_ID_CURR', 'DAYS_BIRTH', '[FE_APP]LOAN_DURATION']], on='SK_ID_CURR', how='left')

# TODO
for t in [12]:
    temp_df = bureau_balance_df[bureau_balance_df['MONTHS_BALANCE']>=-t]
    bb_df = bureau_balance_df.groupby('SK_ID_BUREAU')['STATUS'].sum().reset_index().rename(columns={'STATUS':'STATUS_'+str(t)})
    bureau_df = pd.merge(bureau_df, bb_df, on='SK_ID_BUREAU', how='left')


bureau_df['[FE_B]DEBT_RATIO'] = bureau_df['AMT_CREDIT_SUM_DEBT'] / bureau_df['AMT_CREDIT_SUM']
bureau_df['DAYS_CREDIT_ENDDATE'] =  bureau_df['DAYS_CREDIT_ENDDATE'].fillna(365243)
bureau_df['DAYS_CREDIT_ENDDATE'] = bureau_df['DAYS_CREDIT_ENDDATE'].apply(lambda x: x if x >= 0 else 0)
bureau_df['[FE_B]AGE_END'] = (bureau_df['DAYS_CREDIT_ENDDATE'] + bureau_df['DAYS_BIRTH'].abs())/365
bureau_df['[FE_B]AGE_START'] = (bureau_df['DAYS_CREDIT'] + bureau_df['DAYS_BIRTH'].abs())/365


bureau_df['[FE_B]DAY_UPDATE_CREDIT_RATIO'] = bureau_df['DAYS_CREDIT_UPDATE'] / bureau_df['DAYS_CREDIT']
bureau_df['[FE_B]SUM_DEBT_MAX_OVERDUE_RATIO'] = bureau_df['AMT_CREDIT_SUM_DEBT'] / bureau_df['AMT_CREDIT_MAX_OVERDUE']

bureau_df['[FE_B]AVG_PROLONG_SINCE_START'] = bureau_df['CNT_CREDIT_PROLONG'] / (bureau_df['DAYS_CREDIT'])
bureau_df['[FE_B]AVG_CREDITSUM_SINCE_START'] = bureau_df['AMT_CREDIT_SUM'] / (bureau_df['DAYS_CREDIT'])
bureau_df['[FE_B]AVG_DEBT_SINCE_START'] = bureau_df['AMT_CREDIT_SUM_DEBT'] / bureau_df['DAYS_CREDIT']
bureau_df['[FE_B]AVG_OVERDUE_SINCE_START'] = bureau_df['AMT_CREDIT_MAX_OVERDUE'] / bureau_df['DAYS_CREDIT']
bureau_df['[FE_B]AVG_OVERDUE_DAY_SINCE_START'] = bureau_df['CREDIT_DAY_OVERDUE'] / bureau_df['DAYS_CREDIT']


bureau_df['[FE_B]AVG_DEBT_SINCE_UPDATE'] = bureau_df['AMT_CREDIT_SUM_DEBT'] / bureau_df['DAYS_CREDIT_UPDATE']
bureau_df['[FE_B]AVG_OVERDUE_SINCE_UPDATE'] = bureau_df['AMT_CREDIT_MAX_OVERDUE'] / bureau_df['DAYS_CREDIT_UPDATE']
bureau_df['[FE_B]AVG_OVERDUE_DAY_SINCE_UPDATE'] = bureau_df['CREDIT_DAY_OVERDUE'] / bureau_df['DAYS_CREDIT_UPDATE']


# projection
bureau_df['[FE_B]PROJECTED_DEBT'] = bureau_df['[FE_B]AVG_DEBT_SINCE_START'] * bureau_df['DAYS_CREDIT_ENDDATE']
bureau_df['[FE_B]PROJECTED_OVERDUE_DAY'] = bureau_df['[FE_B]AVG_OVERDUE_DAY_SINCE_UPDATE'] * bureau_df['DAYS_CREDIT_ENDDATE']
bureau_df['[FE_B]APP_PROJECTED_DEBT'] = bureau_df['[FE_B]AVG_DEBT_SINCE_START'] * bureau_df['[FE_APP]LOAN_DURATION']


active_bureau_df = bureau_df[(bureau_df['CREDIT_ACTIVE']=='Active')]
closed_bureau_df = bureau_df[(bureau_df['CREDIT_ACTIVE']=='Closed')]
active_bureau_credit_card_df = bureau_df[(bureau_df['CREDIT_TYPE'] == 'Credit card') &(bureau_df['CREDIT_ACTIVE']=='Active')]
closed_bureau_credit_card_df = bureau_df[(bureau_df['CREDIT_TYPE'] == 'Credit card') &(bureau_df['CREDIT_ACTIVE']=='Closed')]
sold_bureau_df = bureau_df[(bureau_df['CREDIT_ACTIVE']=='Sold')]


division_config = {
    'prefix': 'B',
    'count_all_dfs': False,
    'dfs': [bureau_df, active_bureau_df, closed_bureau_df, closed_bureau_credit_card_df, active_bureau_credit_card_df, sold_bureau_df],
    'column': 'DAYS_CREDIT',
    'groupby_column': 'SK_ID_CURR',
    'groups':[3, 6, 18, 30, 42, 54, 66],
    'step': -30,
    'grouped_dfs': {},
    'grouped_gys': {},
}

numerical_features_config = [
    ('STATUS_12', ['mean', 'median', 'max', 'std']),
    ('[FE_B]DEBT_RATIO', ['mean', 'max', 'std']),
    ('[FE_B]AGE_END', ['mean', 'max', ]),
    ('[FE_B]AGE_START', ['mean', 'max', ]),
    ('[FE_B]AVG_CREDITSUM_SINCE_START', ['mean', 'max', 'sum']),
    ('[FE_B]DAY_UPDATE_CREDIT_RATIO', ['mean', 'max']),
    ('[FE_B]AVG_DEBT_SINCE_START', ['mean', 'max']),
    ('[FE_B]AVG_OVERDUE_SINCE_START', ['mean', 'max']),
    ('[FE_B]AVG_OVERDUE_DAY_SINCE_START', ['mean']),
    ('[FE_B]AVG_DEBT_SINCE_UPDATE', ['mean', 'max']),
    ('[FE_B]AVG_OVERDUE_SINCE_UPDATE', ['mean', 'max']),
    ('[FE_B]AVG_OVERDUE_DAY_SINCE_UPDATE', ['mean']),
    ('[FE_B]SUM_DEBT_MAX_OVERDUE_RATIO', ['mean', 'max']), 
    ('[FE_B]PROJECTED_DEBT', ['mean', 'max']),
    ('[FE_B]APP_PROJECTED_DEBT', ['mean', 'max']),
    ('[FE_B]AVG_PROLONG_SINCE_START', ['mean']),
    ('AMT_ANNUITY', ['trend', 'mean', 'max']),
    ('AMT_CREDIT_SUM_LIMIT', ['trend', 'mean', 'max', ]),
    ('AMT_CREDIT_SUM_DEBT', ['trend', 'mean', 'max']),
    ('AMT_CREDIT_SUM', ['trend', 'mean', 'max']),
    ('AMT_CREDIT_SUM_OVERDUE', ['trend', 'mean', 'max' ]),
    ('AMT_CREDIT_MAX_OVERDUE', ['mean', 'max', ]),
    ('CREDIT_DAY_OVERDUE', ['mean', 'std']),
    ('DAYS_CREDIT', ['mean', 'max', ]),
    ('DAYS_CREDIT_ENDDATE', ['mean', 'max', ]),
    ('DAYS_ENDDATE_FACT', ['mean', 'max', ]),
    ('DAYS_CREDIT_UPDATE', ['mean', 'max', ]),
    ('CNT_CREDIT_PROLONG', ['mean', 'max', ]),
]

# numerical_features_config += groupby_features

categorical_features_config = [
        'CREDIT_ACTIVE',
        'CREDIT_TYPE',
        'CREDIT_CURRENCY',
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

categorical_features= ['EMERGENCYSTATE_MODE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 'REGION_RATING_CLIENT', 'WALLSMATERIAL_MODE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'FONDKAPREMONT_MODE', 'FLAG_OWN_CAR', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE', 'WEEKDAY_APPR_PROCESS_START']

path = 'result/allF_'
a, b, c, feature_importances = validate_model_lgb(training_pdf, categorical_features, params, path, 5, 850, 1)

for k, v in sorted(feature_importances, key=lambda x: x[1], reverse=True):
    if v == 0 and k in aggrated_df.columns:
        aggrated_df.drop(columns=[k], inplace=True)

print(aggrated_df.shape)
aggrated_df.to_csv('bureau.csv', index=False)

# Fold 0.8288113741554093 0.7793792174354872 0.04943215671992207 1762
# Fold 0.8351983499885515 0.779326089353814 0.05587226063473749 2062
# Fold 0.8369559158069537 0.7830107496198739 0.053945166187079785 2195
# Fold 0.8340832490919361 0.7848963687852928 0.04918688030664331 2102
# Fold 0.8352804670656292 0.7825713186496893 0.05270914841593988 2108
# AUC mean: 0.7818367487688314 std: 0.0021735692538231884 min: 0.779326089353814 max: 0.7848963687852928
# OOF AUC: 0.7818218273120306
# Save submission to result/allF_0.78184CV_??LB_SUB.csv
# it took 1948.828064918518 seconds.

# Fold 0.8521327053493527 0.7783850817194058 0.07374762362994691 2844
# Fold 0.8390456693309039 0.7798194414690619 0.05922622786184195 2159
# Fold 0.8490916450439547 0.7839096218054746 0.0651820232384801 2689
# Fold 0.8368043523177469 0.7845794253105307 0.05222492700721615 2102
# Fold 0.8233117386532457 0.7826142211569567 0.040697517496289026 1457
# AUC mean: 0.781861558292286 std: 0.002383442768444462 min: 0.7783850817194058 max: 0.7845794253105307
