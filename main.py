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


categorical_features= ['EMERGENCYSTATE_MODE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 'REGION_RATING_CLIENT',
 'WALLSMATERIAL_MODE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'FONDKAPREMONT_MODE', 'FLAG_OWN_CAR',
  'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY',
   'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE', 'WEEKDAY_APPR_PROCESS_START']

def run_sort_features(all_application_df):
    aggrated_df = pd.read_csv('ag.csv')
    p4 = pd.read_csv('chad_interaction.csv')
    aggrated_df = pd.merge(aggrated_df, p4, how='left', on='SK_ID_CURR')
    # aggrated_df = all_application_df[['SK_ID_CURR']]
    # bureau_df = pd.read_csv('bureau.csv')
    # prev_df = pd.read_csv('prev_application.csv')
    # pos_a_df = pd.read_csv('pos_a.csv')
    # pos_b_df = pd.read_csv('pos_b.csv')
    # c_a_df = pd.read_csv('credit_balance_a.csv')
    # c_b_df = pd.read_csv('credit_balance_b.csv')
    # i_a_df = pd.read_csv('installment_a.csv')
    # i_b_df = pd.read_csv('installment_b.csv')
    # aggrated_df = pd.merge(aggrated_df, all_application_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, bureau_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, prev_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, pos_a_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, c_a_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, i_a_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, pos_b_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, c_b_df, how='left', on='SK_ID_CURR')
    # aggrated_df = pd.merge(aggrated_df, i_b_df, how='left', on='SK_ID_CURR')

    fs = [
    'B_COUNT_3',
    'C_COUNT_3',
    'POS_COUNT_3',
    'I_COUNT_3',
    'PREV_COUNT_3'
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')

    fs = [
    'B_COUNT_6',
    'C_COUNT_6',
    'POS_COUNT_6',
    'I_COUNT_6',
    'PREV_COUNT_6'
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')

    fs = [
    'B_COUNT_18',
    'C_COUNT_18',
    'POS_COUNT_18',
    'I_COUNT_18',
    'PREV_COUNT_18'
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')


    fs = [
    'B_COUNT_30',
    'C_COUNT_30',
    'POS_COUNT_30',
    'I_COUNT_30',
    'PREV_COUNT_30'
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')

    fs = [
    'C_COUNT_42',
    'POS_COUNT_42',
    'I_COUNT_42',
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')

    fs = [
    'C_COUNT_54',
    'POS_COUNT_54',
    'I_COUNT_54',
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')


    fs = [
    'C_COUNT_66',
    'POS_COUNT_66',
    'I_COUNT_66',
    ]

    combinations = list(itertools.combinations(fs, 2))
    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')




    fs = [
    'B_COUNT_global',
    'C_COUNT_global',
    'POS_COUNT_global',
    'I_COUNT_global',
    'PREV_COUNT_global'
    ]

    combinations = list(itertools.combinations(fs, 2))
    

    for index, combination in tqdm(enumerate(combinations)):
        cross_feature(aggrated_df, aggrated_df, combination[0], combination[1], '/')

    print(aggrated_df.shape)
    params = {
        'boosting_type':'gbdt',
        'objective':'binary',
        'n_estimators':50000,
        'learning_rate':0.05,
        'num_leaves': 123, #34,
        'feature_fraction': 0.8,
        'subsample':0.8715623,
        'max_depth': 3,
        'reg_alpha': 10,
        'reg_lambda':0.0735294,
        'min_child_weight': 20,
        'verbose': -1,
    }


    path = 'result/allF_'
    a, b, c, feature_importances = validate_model_lgb(aggrated_df, categorical_features, params, path, 5, 850)

    aggrated_df.to_csv('ag.csv', index=False)
    json.dump(sorted(feature_importances, key=lambda x: x[1], reverse=True), open('sorted_features.json', 'w'))


# df = pd.read_csv('application.csv')
# run_sort_features(df)

aggrated_df = pd.read_csv('top_1000_features.csv')
sorted_features = json.load(open('sorted_features.json'))   
selected_categorical_features = []

p1= pd.read_csv('version 2.3_external_1_predict_revised.csv')
p2= pd.read_csv('version 2.3_external_1_predict.csv')
p3= pd.read_csv('version 2.3_external_3_predict.csv')

# 650
# Fold 0.8756352128464706 0.7983485587694997 0.07728665407697088 881
# Fold 0.8796525147398009 0.7971135408282121 0.08253897391158882 946
# Fold 0.8612961462809788 0.8017900886800846 0.059506057600894224 674
# Fold 0.8677292832056462 0.8008892179636713 0.06684006524197494 770
# Fold 0.8691943008975309 0.8011732978969952 0.0680210030005357 800
#          AUC mean: 0.7998629408276926 std: 0.0018075802958210819 min: 0.7971135408282121 max: 0.8017900886800846
#          OOF AUC: 0.7998369530944482

# 700
# Fold 0.8554714116045279 0.7981866343259514 0.057284777278576504 581
# Fold 0.8770045227736811 0.7964929204183426 0.0805116023553385 890
# Fold 0.8536780928909897 0.802362338231889 0.051315754659100765 567
# Fold 0.8679910855766977 0.8000730977785118 0.06791798779818592 763
# Fold 0.8724218139172359 0.8009641254634895 0.07145768845374645 832
#          AUC mean: 0.7996158232436368 std: 0.0020671461716900287 min: 0.7964929204183426 max: 0.802362338231889
#          OOF AUC: 0.7995778707462998
# 750
# Fold 0.8764231115767462 0.7975439523835123 0.07887915919323396 865
# Fold 0.873603809414443 0.7974842629502972 0.07611954646414587 827
# Fold 0.8666459101813015 0.8027698639579415 0.06387604622336007 722
# Fold 0.8805184687899956 0.7999661710363688 0.08055229775362682 939
# Fold 0.877261274067467 0.8015815325493072 0.07567974151815982 888
#          AUC mean: 0.7998691565754854 std: 0.0021189638381912345 min: 0.7974842629502972 max: 0.8027698639579415
#          OOF AUC: 0.7998447682128907
# 800
# Fold 0.8631218323086232 0.7994877118712447 0.06363412043737848 665
# Fold 0.8700877680226482 0.7963158669755803 0.07377190104706788 762
# Fold 0.8614755597171837 0.8019597928018727 0.05951576691531102 654
# Fold 0.8771980770442422 0.7997261264673505 0.07747195057689171 874
# Fold 0.8615176638282712 0.802331263037096 0.05918640079117521 651
#          AUC mean: 0.7999641522306289 std: 0.002153110858359452 min: 0.7963158669755803 max: 0.802331263037096
#          OOF AUC: 0.7999355150130494
# 900
# Fold 0.8701324047592749 0.798540713574027 0.07159169118524789 748
# Fold 0.8649186864507695 0.7968312706161946 0.0680874158345749 686
# Fold 0.8676098955640014 0.8018750244584013 0.06573487110560006 714
# Fold 0.8790642241023203 0.7993226369898022 0.07974158711251811 890
# Fold 0.8630915677927087 0.8006772596993358 0.062414308093372894 670
#          AUC mean: 0.7994493810675521 std: 0.0017371053396096898 min: 0.7968312706161946 max: 0.8018750244584013
#          OOF AUC: 0.7994312182903126
# 950
# Fold 0.8725719449272594 0.7987260784813087 0.0738458664459507 777
# Fold 0.8994332740736833 0.7962737339561126 0.10315954011757078 1207
# Fold 0.8611378564828507 0.8018109254122117 0.05932693107063902 619
# Fold 0.8816119183639218 0.8000724850382271 0.08153943332569469 918
# Fold 0.8556140960403895 0.8001731561295555 0.055440939910833986 562
#          AUC mean: 0.7994112758034831 std: 0.0018487640056185884 min: 0.7962737339561126 max: 0.8018109254122117
#          OOF AUC: 0.7993163893146765

top_features = [f[0] for f in sorted_features if f[0] in aggrated_df.columns][:1000]
df = aggrated_df[top_features + ['TARGET', 'SK_ID_CURR']]
df.to_csv('top_1000_features.csv', index=False)
print('done')

for i in [800, 750, 650]:
    print(i)
    n = 10
    # i =600
    top_features = [f[0] for f in sorted_features if f[0] in aggrated_df.columns][:i]

    df = aggrated_df[top_features + ['TARGET', 'SK_ID_CURR']]
    selected_categorical_features = [f for f in categorical_features if f in df.columns]

    df = pd.merge(df, p2, how='left', on='SK_ID_CURR')
    df = pd.merge(df, p3, how='left', on='SK_ID_CURR')

    params = {      
        'boosting_type':'gbdt',
        'objective':'binary',
        'n_estimators':50000,
        'learning_rate':0.005,
        'num_leaves': 123,
        'feature_fraction': 0.1,
        'subsample':0.8715623,
        'max_depth': 4,
        'reg_alpha': 10,
        'reg_lambda':0.0735294,
        'min_child_weight': 20,
        'verbose': False,
    }

    path = 'result/' + str(i) + 'F_'
    a, b, c, im = validate_model_lgb(df, selected_categorical_features, params, path, 10, 850, n)

# Fold 0.8750096505926048 0.8063995308178389 0.06861011977476594 18702
# Fold 0.8618107444882508 0.7967707739381481 0.06503997055010269 14406
# Fold 0.8845134126391342 0.8015758706838181 0.08293754195531611 22066
# Fold 0.9010217951728194 0.798315737933794 0.10270605723902548 28504
# Fold 0.8719347719271667 0.806662125 1187476 0.06527264680841915 17818
# Fold 0.8653723092416786 0.8033164120376962 0.06205589720398241 15618
# Fold 0.8874028232599968 0.8072724721717891 0.08013035108820765 23311
# Fold 0.8557031461676002 0.8003475882445299 0.05535555792307034 12723
# Fold 0.8718745909100094 0.8023370503367846 0.06953754057322481 17722
# Fold 0.8731575402556337 0.8076610631497968 0.06549647710583695 18317
# 	 AUC mean: 0.8030658624432941 std: 0.0036776723534786278 min: 0.7967707739381481 max: 0.8076610631497968
# 	 OOF AUC: 0.8030315514745012

# Fold 0.8723631067131277 0.8061083292384893 0.06625477747463837 17823
# Fold 0.856304196033982 0.7965186651735604 0.0597855308604216 12775
# Fold 0.8806443845386335 0.7989165762609709 0.08172780827766257 20648
# Fold 0.8734599727411316 0.8067027565719914 0.06675721616914021 18183
# Fold 0.8600208854979976 0.8038283722322459 0.0561925132657517 14005
# Fold 0.8761551625242867 0.8073554667897812 0.0687996957345055 19288
# Fold 0.8758926020279085 0.8007617631722707 0.07513083885563787 19018
# Fold 0.8767833522092994 0.8023879901899021 0.07439536201939723 19400
# Fold 0.8641224020743957 0.8075237649950452 0.05659863707935053 15320
#          AUC mean: 0.8032097124696076 std: 0.003579560566440909 min: 0.7965186651735604 max: 0.8075237649950452
#          OOF AUC: 0.803184925240143

# Fold 0.884494444571184 0.8071281473722753 0.07736629719890864 21916
# Fold 0.8683064504702591 0.796443713534179 0.07186273693608014 16359
# Fold 0.8713934687836302 0.8020033842353409 0.06939008454828932 17486
# Fold 0.8823890257973948 0.798602323597698 0.08378670219969675 21171
# Fold 0.869623775558814 0.8066200120136322 0.06300376354518178 16974
# Fold 0.8656148781200641 0.8042269974008482 0.06138788071921586 15621
# Fold 0.8778967419543591 0.8080264789176282 0.06987026303673083 19778
# Fold 0.8786646720558753 0.8010845912845017 0.07758008077137357 19897
# Fold 0.873863319701748 0.8023341569987367 0.07152916270301135 18332
# Fold 0.8670257920200972 0.8073458745902435 0.05967991742985368 16208
#          AUC mean: 0.8033815679945084 std: 0.003765336846024333 min: 0.796443713534179 max: 0.8080264789176282
#          OOF AUC: 0.8033611112458897


# 22049077 ken



# 800
# Fold 0.8862246519215536 0.807782923526234 0.0784417283953196 21191
# Fold 0.8654487976839662 0.7968369923794797 0.06861180530448652 14610
# Fold 0.883904483613241 0.8025546157181295 0.08134986789511156 20488
# Fold 0.8966381184114016 0.7987702916319708 0.09786782677943084 25035
# Fold 0.8789100390397698 0.8069479984500211 0.07196204058974875 18771
# Fold 0.8750109864622805 0.8044181789951715 0.07059280746710894 17514
# Fold 0.8876542254432426 0.8082013904670013 0.07945283497624123 21881
# Fold 0.8690920192714111 0.8008122326797942 0.06827978659161693 15692
# Fold 0.88105961805063 0.8024832993107185 0.07857631873991155 19571
# Fold 0.8762929185839164 0.8080112425709626 0.06828167601295387 18073
#          AUC mean: 0.8036819165729485 std: 0.003861553600165504 min: 0.7968369923794797 max: 0.8082013904670013
#          OOF AUC: 0.8036643871169986
# Save submission to result/800F_0.80368CV_SUB.csv
# it took 32401.981915473938 seconds.
