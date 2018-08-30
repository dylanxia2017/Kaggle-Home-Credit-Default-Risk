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

credit_card_balance_df = pd.read_csv('../data/credit_card_balance.csv')
all_application_df = pd.read_csv('application.csv')
aggrated_df = all_application_df[['SK_ID_CURR']]

# Engineering
credit_card_balance_df['FE_TOTAL_DPD'] = credit_card_balance_df['SK_DPD_DEF'] + credit_card_balance_df['SK_DPD']
credit_card_balance_df['[FE_CB]SAME_DPD'] = (credit_card_balance_df['SK_DPD_DEF'] == credit_card_balance_df['SK_DPD']).astype(int)

credit_card_balance_df['[FE_CB]LATE'] = (credit_card_balance_df['SK_DPD'] > 0).astype(int)
credit_card_balance_df['[FE_CB]LATE_T'] = (credit_card_balance_df['SK_DPD_DEF'] > 0).astype(int)

credit_card_balance_df['FE_TOTAL_AMT_DRAW'] = credit_card_balance_df[[
'AMT_DRAWINGS_ATM_CURRENT',
'AMT_DRAWINGS_CURRENT',
'AMT_DRAWINGS_OTHER_CURRENT',
'AMT_DRAWINGS_POS_CURRENT',
]].fillna(0).apply(sum, axis=1)


credit_card_balance_df['FE_TOTAL_CNT_DRAW'] = credit_card_balance_df[[
'CNT_DRAWINGS_ATM_CURRENT',
'CNT_DRAWINGS_CURRENT',
'CNT_DRAWINGS_OTHER_CURRENT',
'CNT_DRAWINGS_POS_CURRENT',
]].fillna(0).apply(sum, axis=1)

credit_card_balance_df['FE_USED_RATIO'] = credit_card_balance_df['FE_TOTAL_AMT_DRAW'] / credit_card_balance_df['AMT_CREDIT_LIMIT_ACTUAL']

credit_card_balance_df['FE_PAID_RATIO']= credit_card_balance_df['AMT_PAYMENT_TOTAL_CURRENT']\
    / (np.maximum(credit_card_balance_df['AMT_TOTAL_RECEIVABLE'], credit_card_balance_df['AMT_INST_MIN_REGULARITY']) - 0)
credit_card_balance_df['FE_FLAG_FULL_PAY'] = credit_card_balance_df['FE_PAID_RATIO'].apply(lambda x: 1 if x >= 0.95 else 0)


k_configs = [0.1, 0.2, 0.3, 0.4]

# k = 0.4
# gy_cols = ['SK_ID_CURR']
# print(k)
# groupby = credit_card_balance_df.groupby(gy_cols)
# cuted_first_k_df = groupby.apply(get_first_k(k, gy_cols)).reset_index()
# cuted_last_k_df = groupby.apply(get_last_k(k, gy_cols)).reset_index()
# print(cuted_first_k_df.head(1))
# cuted_first_k_df.to_csv('credit_bal_first_'+str(k)+'.csv', index=False)
# cuted_last_k_df.to_csv('credit_bal_last_'+str(k)+'.csv', index=False)
# exit()


numerical_features_config = [
    ('FE_TOTAL_AMT_DRAW', ['mean', 'max' ]),
    ('FE_TOTAL_CNT_DRAW', ['mean', 'max']),
    ('FE_USED_RATIO', ['trend', 'mean', 'max', ]),
    ('FE_TOTAL_DPD', ['mean', 'max', ]),
    ('[FE_CB]SAME_DPD', ['mean', 'max', ]),
    ('[FE_CB]LATE', ['mean', 'max', ]),
    ('[FE_CB]LATE_T', ['mean', 'max', ]),
    ('FE_PAID_RATIO', ['mean', 'max', ]),
    ('FE_FLAG_FULL_PAY', ['mean', 'max', 'sum']),
    ('SK_DPD', ['trend', 'mean', 'max', ]),
    ('SK_DPD_DEF', ['trend','mean', 'max', ]), 
    ('AMT_BALANCE', ['mean', 'max']),   
    ('AMT_CREDIT_LIMIT_ACTUAL', ['mean', 'max', ]),
    ('AMT_DRAWINGS_ATM_CURRENT', ['trend', 'mean', 'max', ]),
    ('AMT_DRAWINGS_CURRENT', ['trend', 'mean', 'max', ]),
    ('AMT_DRAWINGS_OTHER_CURRENT', ['mean', 'max', ]),
    ('AMT_DRAWINGS_POS_CURRENT', ['mean', 'max', ]),
    ('AMT_INST_MIN_REGULARITY', ['mean', 'max', ]),
    ('AMT_PAYMENT_CURRENT', ['trend', 'mean', 'max',]),
    ('AMT_PAYMENT_TOTAL_CURRENT', ['mean', 'max', ]),
    ('AMT_RECEIVABLE_PRINCIPAL', ['mean', 'max', ]),
    ('AMT_RECIVABLE', ['mean', 'max', ]),
    ('AMT_TOTAL_RECEIVABLE', ['mean', 'max', ]),
    ('CNT_DRAWINGS_ATM_CURRENT', ['mean', 'max', ]),
    ('CNT_DRAWINGS_CURRENT', [ 'mean', 'max', ]),
    ('CNT_DRAWINGS_OTHER_CURRENT', ['mean', 'max', ]),
    ('CNT_DRAWINGS_POS_CURRENT', ['mean', 'max', ]),
    ('CNT_INSTALMENT_MATURE_CUM', ['mean', 'max', ]),
]


temp_list = []
for k in k_configs:
    print(k)
    cuted_first_k_df = (pd.read_csv('credit_bal_first_'+str(k)+'.csv'))
    cuted_last_k_df = (pd.read_csv('credit_bal_last_'+str(k)+'.csv'))
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


# Trending Features (group by previous application)
trends_df = credit_card_balance_df[['SK_ID_PREV', 'SK_ID_CURR']]
trends_df = trends_df.drop_duplicates()

credit_card_balance_df.sort_values('MONTHS_BALANCE', ascending=False, inplace=True)

# features = []
features = ['FE_USED_RATIO', 'SK_DPD', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_PAYMENT_CURRENT']

print('Getting Trends')
time_range = [-3, -6, -18]
ts = []
for time in time_range:
    df = credit_card_balance_df[credit_card_balance_df['MONTHS_BALANCE'] >= time]
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
        ts += (agg_num(trends_df.groupby('SK_ID_CURR'), 'CB', feature+'_TREND_'+str(time), ['mean', 'max', 'min']))
        ts += (agg_num(trends_df.groupby('SK_ID_CURR'), 'CB', feature+'_SKEW_'+str(time), ['mean', 'max', 'min']))

for t in tqdm(ts):
    aggrated_df = pd.merge(aggrated_df, t, how='left', on='SK_ID_CURR')  


print(aggrated_df.head(10))

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
aggrated_df.to_csv('credit_balance_a.csv', index=False)

# A only:
# Fold 0.8189543169343193 0.7730767322155367 0.04587758471878256 2086
# Fold 0.8236743619603927 0.7715934352074618 0.052080926752930856 2391
# Fold 0.813845406006931 0.7749161013380519 0.03892930466887912 1799
# Fold 0.8169069786922333 0.7765162654669496 0.04039071322528365 2008
# Fold 0.8087987937993646 0.775840861800686 0.03295793199867858 1518
#          AUC mean: 0.7743886792057373 std: 0.0018137378867200235 min: 0.7715934352074618 max: 0.7765162654669496
#          OOF AUC: 0.7743377142754992


# Fold 0.821517004317735 0.7740357321726814 0.0474812721450536 2110
# Fold 0.8158944777190782 0.7728806030490134 0.04301387467006479 1777
# Fold 0.8225976484494486 0.7755551110813529 0.047042537368095716 2174
# Fold 0.8166777905700106 0.776507174111096 0.04017061645891451 1857
# Fold 0.8191496875013875 0.7761128579256023 0.04303682957578525 1967
#          AUC mean: 0.7750182956679492 std: 0.0013592760645063345 min: 0.7728806030490134 max: 0.776507174111096
#          OOF AUC: 0.7749989789716756






