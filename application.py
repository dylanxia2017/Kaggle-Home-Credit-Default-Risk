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


train_application_df = pd.read_csv('../data/application_train.csv')
test_application_df = pd.read_csv('../data/application_test.csv')
previous_application_df = pd.read_csv('../data/previous_application.csv')
all_application_df = pd.concat([train_application_df, test_application_df], axis=0)



categorical_features = [
'REGION_RATING_CLIENT',
'REGION_RATING_CLIENT_W_CITY'
]

for column in train_application_df.columns:
    if column == 'TARGET' or column == 'SK_ID_CURR':
        continue
    if train_application_df[column].dtype == 'object':
        categorical_features.append(column)
categorical_features = list(set(categorical_features))
print(categorical_features)

for column in categorical_features:
    all_application_df[column] = all_application_df[column].fillna('XNA')
    all_application_df[column], _ = pd.factorize(all_application_df[column])


# clearning
all_application_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# engineering
all_application_df['[FE_APP]LONG_EMPLOYMENT'] = (all_application_df['DAYS_EMPLOYED'] > -2000).astype(int)


all_application_df['[FE_APP]CREDIT_COVERAGE'] = all_application_df['AMT_CREDIT'] / all_application_df['AMT_GOODS_PRICE']
all_application_df['[FE_APP]LOAN_DURATION'] = all_application_df['AMT_CREDIT'] / all_application_df['AMT_ANNUITY']


all_application_df['[FE_APP]AGE'] = (all_application_df['DAYS_BIRTH'] / 365).astype(int)
all_application_df['[FE_APP]AGE_FINISH'] = (all_application_df['[FE_APP]LOAN_DURATION']/12 + all_application_df['DAYS_BIRTH'].abs())/365
all_application_df['[FE_APP]AGE_EMPLOY'] = (all_application_df['DAYS_EMPLOYED'] - all_application_df['DAYS_BIRTH'])/365
all_application_df['[FE_APP]AGE_REGISTRATION'] = (all_application_df['DAYS_REGISTRATION'] - all_application_df['DAYS_BIRTH'])/365
all_application_df['[FE_APP]AGE_ID'] = (all_application_df['DAYS_ID_PUBLISH'] - all_application_df['DAYS_BIRTH'])/365


score_columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
all_application_df['[FE_APP]MIN_SCORE'] = all_application_df[score_columns].min(axis=1)
all_application_df['[FE_APP]MAX_SCORE'] = all_application_df[score_columns].max(axis=1)
all_application_df['[FE_APP]MEAN_SCORE'] = all_application_df[score_columns].mean(axis=1)
all_application_df['[FE_APP]SUM_SCORE'] = all_application_df[score_columns].sum(axis=1)
all_application_df['[FE_APP]MEDIAN_SCORE'] = all_application_df[score_columns].median(axis=1)
all_application_df['[FE_APP]STD_SCORE'] = all_application_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
all_application_df['[FE_APP]STD_SCORE'] = all_application_df['[FE_APP]STD_SCORE'].fillna(all_application_df['[FE_APP]STD_SCORE'].mean())

all_application_df['[FE_APP]TOTAL_DOCUMENT_MISS'] = all_application_df[[
  'FLAG_DOCUMENT_2',
  'FLAG_DOCUMENT_3',
  'FLAG_DOCUMENT_4',
  'FLAG_DOCUMENT_5',
  'FLAG_DOCUMENT_6',
  'FLAG_DOCUMENT_7',
  'FLAG_DOCUMENT_8',
  'FLAG_DOCUMENT_9',
  'FLAG_DOCUMENT_10',
  'FLAG_DOCUMENT_11',
  'FLAG_DOCUMENT_12',
  'FLAG_DOCUMENT_13',
  'FLAG_DOCUMENT_14',
  'FLAG_DOCUMENT_15',
  'FLAG_DOCUMENT_16',
  'FLAG_DOCUMENT_17',
  'FLAG_DOCUMENT_18',
  'FLAG_DOCUMENT_19',
  'FLAG_DOCUMENT_20',
  'FLAG_DOCUMENT_21',
]].fillna(0).apply(sum, axis=1)

all_application_df.drop(columns=[
  'FLAG_DOCUMENT_10',
  'FLAG_DOCUMENT_11', 
  'FLAG_DOCUMENT_12', 
  'FLAG_DOCUMENT_13',
  'FLAG_DOCUMENT_14',
  'FLAG_DOCUMENT_15', 
  'FLAG_DOCUMENT_16',
  'FLAG_DOCUMENT_17',
  'FLAG_DOCUMENT_18',
  'FLAG_DOCUMENT_19',
  'FLAG_DOCUMENT_2',
  'FLAG_DOCUMENT_20',
  'FLAG_DOCUMENT_21',
  'FLAG_DOCUMENT_4',
  'FLAG_DOCUMENT_7',
  'FLAG_DOCUMENT_9',
], inplace=True)


all_application_df['[FE_APP]TOTAL_ENQUIRIES'] = all_application_df[[
  'AMT_REQ_CREDIT_BUREAU_HOUR',
  'AMT_REQ_CREDIT_BUREAU_DAY',
  'AMT_REQ_CREDIT_BUREAU_WEEK',
  'AMT_REQ_CREDIT_BUREAU_MON',
  'AMT_REQ_CREDIT_BUREAU_QRT',
  'AMT_REQ_CREDIT_BUREAU_YEAR',
]].fillna(0).apply(sum, axis=1)


all_application_df['[FE_APP]ANNUITY_TO_INCOME_RATIO'] = all_application_df['AMT_ANNUITY'] / (1 + all_application_df['AMT_INCOME_TOTAL'])
all_application_df['[FE_APP]CNT_ADULT'] = all_application_df['CNT_FAM_MEMBERS'] - all_application_df['CNT_CHILDREN']
all_application_df['[FE_APP]INC_PER_CHILD'] = all_application_df['AMT_INCOME_TOTAL'] / (1 + all_application_df['CNT_CHILDREN'])
all_application_df['[FE_APP]INC_PER_ADULT'] = all_application_df['AMT_INCOME_TOTAL'] / (1 + all_application_df['[FE_APP]CNT_ADULT'])
all_application_df['[FE_APP]RATIO_CHILD_ADULT'] = all_application_df['CNT_CHILDREN'] / (1 + all_application_df['[FE_APP]CNT_ADULT'])




all_application_df['[FE_APP]AMT_INCOME_TOTAL_T'] =  np.log1p(all_application_df['AMT_INCOME_TOTAL'])
all_application_df['[FE_APP]AMT_CREDIT_T'] =  np.log1p(all_application_df['AMT_CREDIT'])
all_application_df['[FE_APP]AMT_ANNUITY_T'] =  np.log1p(all_application_df['AMT_ANNUITY'])
all_application_df['[FE_APP]AMT_GOODS_PRICE_T'] =  np.log1p(all_application_df['AMT_GOODS_PRICE'])
all_application_df['[FE_APP]REGION_POPULATION_RELATIVE_T'] =  np.sqrt(all_application_df['REGION_POPULATION_RELATIVE'])
all_application_df['[FE_APP]DAYS_BIRTH_T'] =  np.sqrt(all_application_df['DAYS_BIRTH'].abs())
all_application_df['[FE_APP]DAYS_EMPLOYED_T'] =  np.sqrt(all_application_df['DAYS_EMPLOYED'].abs())
all_application_df['[FE_APP]DAYS_REGISTRATION_T'] =  np.sqrt(all_application_df['DAYS_REGISTRATION'].abs())
all_application_df['[FE_APP]OWN_CAR_AGE_T'] =  np.sqrt(all_application_df['OWN_CAR_AGE'].abs())


all_application_df['[FE_APP]APARTMENTS_AVG_T'] =  np.log1p(all_application_df['APARTMENTS_AVG']*50)
all_application_df['[FE_APP]YEARS_BEGINEXPLUATATION_AVG_T'] =  all_application_df['YEARS_BEGINEXPLUATATION_AVG']**30
all_application_df['[FE_APP]YEARS_BUILD_AVG_T'] =  all_application_df['YEARS_BUILD_AVG']**3

all_application_df['[FE_APP]COMMONAREA_AVG_T'] =  all_application_df['COMMONAREA_AVG']**(-1/200)
all_application_df['[FE_APP]ELEVATORS_AVG_T'] =  all_application_df['ELEVATORS_AVG']**(1/40)
all_application_df['[FE_APP]ENTRANCES_AVG_T'] =  all_application_df['ENTRANCES_AVG']**(1/3)
all_application_df['[FE_APP]FLOORSMAX_AVG_T'] =  all_application_df['FLOORSMAX_AVG']**(1/2.5)
all_application_df['[FE_APP]FLOORSMIN_AVG_T'] =  all_application_df['FLOORSMIN_AVG']**(1/2.2)
all_application_df['[FE_APP]LANDAREA_AVG_T'] =  all_application_df['LANDAREA_AVG']**(1/5)
all_application_df['[FE_APP]LIVINGAPARTMENTS_AVG_T'] =  all_application_df['LIVINGAPARTMENTS_AVG']**(1/3)
all_application_df['[FE_APP]LIVINGAREA_AVG_T'] =  all_application_df['LIVINGAREA_AVG']**(1/3.5)
all_application_df['[FE_APP]NONLIVINGAPARTMENTS_AVG_T'] =  all_application_df['NONLIVINGAPARTMENTS_AVG']**(1/7)
all_application_df['[FE_APP]NONLIVINGAREA_AVG_T'] =  all_application_df['NONLIVINGAREA_AVG']**(1/5)
all_application_df['[FE_APP]TOTALAREA_MODE_T'] =  all_application_df['TOTALAREA_MODE']**(1/3)
all_application_df['[FE_APP]OBS_30_CNT_SOCIAL_CIRCLE_T'] =  all_application_df['OBS_30_CNT_SOCIAL_CIRCLE']**(1/7)
all_application_df['[FE_APP]DEF_30_CNT_SOCIAL_CIRCLE_T'] =  all_application_df['DEF_30_CNT_SOCIAL_CIRCLE']**(1/7)
all_application_df['[FE_APP]OBS_60_CNT_SOCIAL_CIRCLE_T'] =  all_application_df['OBS_60_CNT_SOCIAL_CIRCLE']**(1/7)
all_application_df['[FE_APP]DEF_60_CNT_SOCIAL_CIRCLE_T'] =  all_application_df['DEF_60_CNT_SOCIAL_CIRCLE']**(1/7)
all_application_df['[FE_APP]DAYS_LAST_PHONE_CHANGE_T'] =  all_application_df['DAYS_LAST_PHONE_CHANGE'].abs()**(1/2)



def apply_aggreation_configs(df, configs):
    for groupby_cols, specs in configs:
        group_object = df.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            
            agg_name = '[FE_APP]GROUP_{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            aggreation_result_df = group_object[select].agg(agg).reset_index().rename(index=str,
                                      columns={select: agg_name})
            df = pd.merge(df, aggreation_result_df, how='left', on=groupby_cols)
            
            r_name = '[FE_APP]DIFF_{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            df[r_name] = ((df[select] - df[agg_name]))
            
            r_name = '[FE_APP]DIV_{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            df[r_name] = (df[select] / df[agg_name])
    return df
            

aggreation_configs = [
    (['ORGANIZATION_TYPE'], 
     [
         ('AMT_CREDIT', 'mean'),
         ('AMT_ANNUITY', 'median'),
         ('AMT_ANNUITY', 'mean'),
         ('AMT_GOODS_PRICE', 'median'),
         ('AMT_GOODS_PRICE', 'mean'),
         ('EXT_SOURCE_1', 'median'),
         ('EXT_SOURCE_2', 'median'),
         ('EXT_SOURCE_3', 'median'),
         ('[FE_APP]LOAN_DURATION', 'median'),
         ('[FE_APP]CREDIT_COVERAGE', 'median'),
     ]
    ),
    (['OCCUPATION_TYPE'], 
     [
         ('AMT_CREDIT', 'median'),
         ('AMT_ANNUITY', 'median'),
         ('AMT_GOODS_PRICE', 'median'),
         ('EXT_SOURCE_1', 'median'),
         ('EXT_SOURCE_2', 'median'),
         ('EXT_SOURCE_3', 'median'),
         ('[FE_APP]LOAN_DURATION', 'median'),
         ('[FE_APP]CREDIT_COVERAGE', 'median'),
     ]
    ),
    (['CODE_GENDER'], 
     [
         ('AMT_CREDIT', 'median'),
         ('[FE_APP]LOAN_DURATION', 'median'),
     ]
    ),
    (['NAME_EDUCATION_TYPE'], 
     [
         ('AMT_CREDIT', 'median'),
         ('AMT_ANNUITY', 'median'),
         ('EXT_SOURCE_1', 'median'),
         ('[FE_APP]LOAN_DURATION', 'median'),
     ]
    ),
    (['NAME_CONTRACT_TYPE'], 
     [
         ('AMT_CREDIT', 'median'),
         ('[FE_APP]LOAN_DURATION', 'median'),
     ]
    ),
]

all_application_df = apply_aggreation_configs(all_application_df, aggreation_configs)


all_application_df.to_csv('application.csv', index=False)

params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'n_estimators':20000,
    'learning_rate':0.1,
    'num_leaves': 123, #34,
    'feature_fraction': 1,
    'subsample':0.8715623,
    'max_depth': 3,
    'reg_alpha': 10,
    'reg_lambda':0.0735294,
    'min_child_weight': 20,
    'verbose': False,
}

path = 'result/allF_'

a, b, c, feature_importances = validate_model_lgb(all_application_df, categorical_features, params, path, 5, 850)



# Fold 0.8091702276852056 0.7688799202397254 0.04029030744548023 848
# Fold 0.8085306006555572 0.767769422838373 0.040761177817184135 827
# Fold 0.8072060154246798 0.7712471763850036 0.03595883903967623 808
# Fold 0.8113993348956272 0.7737283363579368 0.0376709985376904 944
# Fold 0.8068787320689039 0.7735435452878571 0.033335186781046855 791
# 	 AUC mean: 0.7710336802217792 std: 0.0024041727591493766 min: 0.767769422838373 max: 0.7737283363579368
# 	 OOF AUC: 0.7709867226133618
# Save submission to result/allF_0.77103CV_??LB_SUB.csv
# it took 827.5191123485565 seconds.

# Fold 0.8023348755774744 0.7694244550368265 0.032910420540647856 667
# Fold 0.8169299216176387 0.7681854628044293 0.048744458813209324 1084
# Fold 0.8023126319186483 0.7714880367535873 0.030824595165060975 680
# Fold 0.8136805428616236 0.7736827121438231 0.0399978307178005 1015
# Fold 0.8021507743491147 0.7730843498810438 0.02906642446807095 666
# 	 AUC mean: 0.7711730033239419 std: 0.0020992355538067852 min: 0.7681854628044293 max: 0.7736827121438231
# 	 OOF AUC: 0.7711193423689834

# Fold 0.8020613476018458 0.7691314563649176 0.03292989123692813 656
# Fold 0.8068594317258871 0.7680951227992983 0.038764308926588886 793
# Fold 0.8069921978423471 0.7717321995073105 0.03525999833503657 805
# Fold 0.8057080826814725 0.7738576423702497 0.03185044031122286 783
# Fold 0.8051200625658501 0.7734492082211007 0.031670854344749366 740
# 	 AUC mean: 0.7712531258525753 std: 0.0022939163160940724 min: 0.7680951227992983 max: 0.7738576423702497
# 	 OOF AUC: 0.771205256517861