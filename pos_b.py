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


# Active                   9151119
# Completed                 744883
# Signed                     87260
# Demand                      7065
# Returned to the store       5461
# Approved                    4917
# Amortized debt               636
# Canceled                      15
# XNA                            2
# Name: NAME_CONTRACT_STATUS, dtype: int64
a_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Active')]
c_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Completed')]
s_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Signed')]
d_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Demand')]
r_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Returned to the store')]
ap_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Approved')]
am_pos_df = pos_df[(pos_df['NAME_CONTRACT_STATUS']=='Amortized debt')]


division_config = {
    'prefix': 'POS',
    'count_all_dfs': False,
    'dfs': [
            pos_df,
            a_pos_df,
            c_pos_df,
            s_pos_df,
            d_pos_df,
            r_pos_df,
            ap_pos_df,
            am_pos_df
            ],
    'column': 'MONTHS_BALANCE',
    'groupby_column': 'SK_ID_CURR',
    'groups':[3, 6, 18, 30, 42, 54, 66],
    'step': -1,
    'grouped_dfs': {},
    'grouped_gys': {},
}

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


categorical_features_config = [
        'NAME_CONTRACT_STATUS'
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
a, b, c, feature_importances = validate_model_lgb(training_pdf, categorical_features, params, path, 5, 850, 10)
json.dump(sorted(feature_importances, key=lambda x: x[1], reverse=True), open('sorted_features.json', 'w'))

for k, v in sorted(feature_importances, key=lambda x: x[1], reverse=True):
    if v == 0 and k in aggrated_df.columns:
        aggrated_df.drop(columns=[k], inplace=True)

print(aggrated_df.shape)
aggrated_df.to_csv('pos_b.csv', index=False)

# B
# Fold 0.8193012942571042 0.7773376349790427 0.04196365927806145 1591
# Fold 0.8180095747994429 0.7748014155939305 0.043208159205512464 1522
# Fold 0.8230668201859287 0.7803675340913822 0.04269928609454654 1796
# Fold 0.8130788971277602 0.7790177705548049 0.034061126572955236 1325
# Fold 0.8259514824858337 0.7815494139148083 0.044402068571025466 1963
#          AUC mean: 0.7786147538267938 std: 0.002367254612443308 min: 0.7748014155939305 max: 0.7815494139148083
#          OOF AUC: 0.7785880481768053

# Total

# Fold 0.8251855175859044 0.7757390452257387 0.04944647236016575 2058
# Fold 0.8208413043211176 0.7724288887735845 0.04841241554753306 1793
# Fold 0.8236696510949788 0.7773877388994661 0.0462819121955127 1987
# Fold 0.8169255456204344 0.7783823909414471 0.03854315467898728 1643
# Fold 0.8092536659508013 0.7803142826856999 0.02893938326510137 1296
# AUC mean: 0.7768504693051873 std: 0.002661544045602956 min: 0.7724288887735845 max: 0.7803142826856999
# OOF AUC: 0.7767927775047649
# Save submission to result/allF_0.77685CV_??LB_SUB.csv
# it took 1141.281842470169 seconds.

# Fold 0.8247641194821469 0.7770845955292766 0.04767952395287034 1962
# Fold 0.8220773046646602 0.7737625678776855 0.04831473678697473 1792
# Fold 0.8325197908336569 0.7784916510840934 0.05402813974956355 2401
# Fold 0.8131307188738317 0.7785607268638753 0.03456999200995636 1406
# Fold 0.8130624705506899 0.7810130809185354 0.032049389632154535 1422
# AUC mean: 0.7777825244546932 std: 0.002374536177791974 min: 0.7737625678776855 max: 0.7810130809185354

# Fold 0.8204519076270508 0.7772608407424613 0.04319106688458951 1680
# Fold 0.8223299030582807 0.7744648608676106 0.047865042190670115 1783
# Fold 0.8267952599778959 0.7799460293369718 0.04684923064092417 2042
# Fold 0.8274249117376238 0.7796009425584305 0.04782396917919329 2111
# Fold 0.8110618008766128 0.7826905750088424 0.02837122586777041 1294
# AUC mean: 0.7787926497028633 std: 0.002765994884501261 min: 0.7744648608676106 max: 0.7826905750088424
# OOF AUC: 0.7787341118342109

# time is not useful

# Fold 0.8209787803319132 0.7774742558338524 0.04350452449806075 1703
# Fold 0.829916508003927 0.775261654796694 0.05465485320723307 2160
# Fold 0.8334208145360162 0.779803752467816 0.05361706206820027 2406
# Fold 0.8238305452504538 0.7794915327931564 0.04433901245729743 1910
# Fold 0.8130320084535343 0.7824064915130751 0.03062551694045923 1364
# AUC mean: 0.7788875374809188 std: 0.0023971519757222323 min: 0.775261654796694 max: 0.7824064915130751
# OOF AUC: 0.7788283002276273
# Save submission to result/allF_0.77889CV_??LB_SUB.csv
# it took 1358.596748828888 seconds.
# (356255, 73)
