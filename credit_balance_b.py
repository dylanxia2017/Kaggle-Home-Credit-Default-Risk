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



# Trending Features (group by previous application)
trends_df = credit_card_balance_df[['SK_ID_PREV', 'SK_ID_CURR']]
trends_df = trends_df.drop_duplicates()

credit_card_balance_df.sort_values('MONTHS_BALANCE', ascending=False, inplace=True)

a_cb_df = credit_card_balance_df[(credit_card_balance_df['NAME_CONTRACT_STATUS']=='Active')]
c_cb_df = credit_card_balance_df[(credit_card_balance_df['NAME_CONTRACT_STATUS']=='Completed')]
s_cb_df = credit_card_balance_df[(credit_card_balance_df['NAME_CONTRACT_STATUS']=='Signed')]
d_cb_df = credit_card_balance_df[(credit_card_balance_df['NAME_CONTRACT_STATUS']=='Demand')]
sent_cb_df = credit_card_balance_df[(credit_card_balance_df['NAME_CONTRACT_STATUS']=='Sent proposal')]


division_config = {
    'prefix': 'C',
    'count_all_dfs': False,
    'dfs': [
            credit_card_balance_df, 
            a_cb_df, 
            c_cb_df, 
            s_cb_df
            ],
    'column': 'MONTHS_BALANCE',
    'groupby_column': 'SK_ID_CURR',
    'groups':[3, 6, 18, 30, 42, 54, 66],
    'step': -1,
    'grouped_dfs': {},
    'grouped_gys': {},
}

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
aggrated_df.to_csv('credit_balance_b.csv', index=False)
# b
# Fold 0.8221402032687521 0.7742218665541525 0.04791833671459966 2135
# Fold 0.8253269207674252 0.7729366260357514 0.05239029473167378 2348
# Fold 0.8264083564990218 0.7766566328663806 0.04975172363264113 2408
# Fold 0.8089299445774856 0.7765691535033933 0.03236079107409229 1431
# Fold 0.8085247248762643 0.7770993234723829 0.03142540140388139 1403
#          AUC mean: 0.7754967204864122 std: 0.0016274631658295374 min: 0.7729366260357514 max: 0.7770993234723829
#          OOF AUC: 0.7754494596180609


# all

# Fold 0.8231529242469009 0.7736009008832816 0.0495520233636193 2075
# Fold 0.8134674923145185 0.7718008335440963 0.041666658770422194 1557
# Fold 0.8288058064829479 0.7755590012696727 0.05324680521327518 2431
# Fold 0.825352312861918 0.7764502783256964 0.04890203453622166 2236
# Fold 0.8093995357898347 0.7762517647231917 0.03314777106664302 1382
# AUC mean: 0.7747325557491878 std: 0.0017786071392090035 min: 0.7718008335440963 max: 0.7764502783256964
# OOF AUC: 0.7746998385128692
# Save submission to result/allF_0.77473CV_??LB_SUB.csv
# it took 1173.178748369217 seconds.

# Fold 0.8238173818601552 0.7734937124162178 0.0503236694439374 2119
# Fold 0.8132592652946056 0.7717505710285244 0.04150869426608117 1555
# Fold 0.8186493892749737 0.7762802713962095 0.042369117878764184 1829
# Fold 0.8150762361545298 0.7767002977365606 0.03837593841796927 1662
# Fold 0.8133354226526864 0.7764285509826901 0.03690687166999629 1555
# AUC mean: 0.7749306807120405 std: 0.0019684781054589108 min: 0.7717505710285244 max: 0.7767002977365606
# OOF AUC: 0.7748978221783966

# Fold 0.8209704468859177 0.7734058178019779 0.04756462908393977 1955
# Fold 0.8183254385703707 0.7723865063830705 0.04593893218730016 1816
# Fold 0.8286052996893024 0.7757330586974378 0.052872240991864605 2385
# Fold 0.8233404553834489 0.7764275713107233 0.04691288407272565 2116
# Fold 0.8110785476268783 0.7764081488686722 0.0346703987582061 1445
# AUC mean: 0.7748722206123763 std: 0.0016642323181663304 min: 0.7723865063830705 max: 0.7764275713107233
# OOF AUC: 0.7748453324663231

# Fold 0.820702790707815 0.7738966888991272 0.04680610180868783 1902
# Fold 0.831868946859118 0.7723419295273506 0.05952701733176746 2552
# Fold 0.8177079008462533 0.7754038497296488 0.04230405111660451 1776
# Fold 0.8233786034185047 0.7767893410122515 0.04658926240625316 2095
# Fold 0.8117060641564653 0.7765938056125263 0.03511225854393896 1467
# AUC mean: 0.7750051229561808 std: 0.0016844913174167862 min: 0.7723419295273506 max: 0.7767893410122515
# OOF AUC: 0.7749545315186395
