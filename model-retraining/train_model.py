import pandas as pd
import numpy as np
from train_pipelines.utils import _drop_application_columns
from train_pipelines.utils import do_agg
from train_pipelines.utils import do_label_encoder
from train_pipelines.utils import general_preprocess
from train_pipelines.utils import reduce_memory
from train_pipelines import bureau_pipeline
from train_pipelines import previous_balance_pipeline
from train_pipelines import previous_pipeline
from sklearn.pipeline import Pipeline
import joblib
import os
import gc
from lightgbm import LGBMClassifier
import train_pipelines.config as config
from sklearn.metrics import roc_auc_score
from datetime import datetime
import hashlib
import sys

data_folder = "../home-credit-default-risk/"
prediction_folder = './binary_files/'
GENERATE_AUX = int(sys.argv[1])

train_df = pd.read_csv(os.path.join(data_folder,'application_train.csv'))
train_df = general_preprocess(train_df)

if GENERATE_AUX:
    group1 = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE']
    group2 = ['CREDIT_TO_ANNUITY_GROUP', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
    transformer = Pipeline([('do_agg1', do_agg(group1, 'EXT_SOURCES_MEAN', 'GROUP1_EXT_SOURCES_MEDIAN','median')),
                            ('do_agg2', do_agg(group1, 'EXT_SOURCES_MEAN', 'GROUP1_EXT_SOURCES_STD','std')),
                            ('do_agg3', do_agg(group1, 'AMT_INCOME_TOTAL', 'GROUP1_INCOME_MEDIAN','median')),
                            ('do_agg4', do_agg(group1, 'AMT_INCOME_TOTAL', 'GROUP1_INCOME_STD','std')),
                            ('do_agg5', do_agg(group1, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP1_CREDIT_TO_ANNUITY_MEDIAN','median')),
                            ('do_agg6', do_agg(group1, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP1_CREDIT_TO_ANNUITY_STD','std')),
                            ('do_agg7', do_agg(group1, 'AMT_CREDIT', 'GROUP1_CREDIT_MEDIAN','median')),
                            ('do_agg8', do_agg(group1, 'AMT_CREDIT', 'GROUP1_CREDIT_STD','std')),
                            ('do_agg9', do_agg(group1, 'AMT_ANNUITY', 'GROUP1_ANNUITY_MEDIAN','median')),
                            ('do_agg10', do_agg(group1, 'AMT_ANNUITY', 'GROUP1_ANNUITY_STD','std')),
                            ('do_agg11', do_agg(group2, 'EXT_SOURCES_MEAN','GROUP2_EXT_SOURCES_MEDIAN','median')),
                            ('do_agg12', do_agg(group2, 'EXT_SOURCES_MEAN','GROUP2_EXT_SOURCES_STD','std')),
                            ('do_agg13', do_agg(group2, 'AMT_INCOME_TOTAL','GROUP2_INCOME_MEDIAN','median')),
                            ('do_agg14', do_agg(group2, 'AMT_INCOME_TOTAL','GROUP2_INCOME_STD','std')),
                            ('do_agg15', do_agg(group2, 'CREDIT_TO_ANNUITY_RATIO','GROUP2_CREDIT_TO_ANNUITY_MEDIAN','median')),
                            ('do_agg16', do_agg(group2, 'CREDIT_TO_ANNUITY_RATIO','GROUP2_CREDIT_TO_ANNUITY_STD','std')),
                            ('do_agg17', do_agg(group2, 'AMT_CREDIT','GROUP2_CREDIT_MEDIAN','median')),
                            ('do_agg18', do_agg(group2, 'AMT_CREDIT','GROUP2_CREDIT_STD','std')),
                            ('do_agg19', do_agg(group2, 'AMT_ANNUITY','GROUP2_ANNUITY_MEDIAN','median')),
                            ('do_agg20', do_agg(group2, 'AMT_ANNUITY','GROUP2_ANNUITY_STD','std')),
                            ('do_label_encoder', do_label_encoder())])
    transformer.fit(train_df,None)
    joblib.dump(transformer, os.path.join(prediction_folder, "transformer.joblib"))
    
    bureau_df = bureau_pipeline.get_bureau(data_folder, num_rows=None)
    joblib.dump(bureau_df, os.path.join(prediction_folder, "bureau_and_balance.joblib"))
    del bureau_df
    gc.collect()
    print('bureau_df done!')

    prev_df = previous_pipeline.get_previous_applications(data_folder, num_rows=None)
    joblib.dump(prev_df, os.path.join(prediction_folder, "previous.joblib"))
    del prev_df
    gc.collect()
    print('prev_df done!')

    pos_df = previous_balance_pipeline.get_pos_cash(data_folder, num_rows=None)
    joblib.dump(pos_df, os.path.join(prediction_folder, "pos_cash.joblib"))
    del pos_df
    gc.collect()
    print('pos_df done!')

    ins_df = previous_balance_pipeline.get_installment_payments(data_folder, num_rows=None)
    joblib.dump(ins_df, os.path.join(prediction_folder, "payments.joblib"))
    del ins_df
    gc.collect()
    print('ins_df done!')

    cc_df = previous_balance_pipeline.get_credit_card(data_folder, num_rows=None)
    joblib.dump(cc_df, os.path.join(prediction_folder, "credit_card.joblib"))
    del cc_df
    gc.collect()
    print('cc_df done!')
    
    print ('Finsh generating aux data, please train the model!')
else:
    transformer = joblib.load(os.path.join(prediction_folder, "transformer.joblib"))
    transformed_data = transformer.transform(train_df)
    transformed_data = _drop_application_columns(transformed_data)
    del transformer
    del train_df
    gc.collect()
    
    bureau_df = joblib.load(os.path.join(prediction_folder, "bureau_and_balance.joblib"))
    transformed_data = pd.merge(transformed_data, bureau_df, on='SK_ID_CURR', how='left')
    del bureau_df
    gc.collect()
    
    prev_df = joblib.load(os.path.join(prediction_folder, "previous.joblib"))
    transformed_data = pd.merge(transformed_data, prev_df, on='SK_ID_CURR', how='left')
    del prev_df
    gc.collect()
    
    pos_df = joblib.load(os.path.join(prediction_folder, "pos_cash.joblib"))
    transformed_data = pd.merge(transformed_data, pos_df, on='SK_ID_CURR', how='left')
    del pos_df
    gc.collect()
    
    ins_df = joblib.load(os.path.join(prediction_folder, "payments.joblib"))
    transformed_data = pd.merge(transformed_data, ins_df, on='SK_ID_CURR', how='left')
    del ins_df
    gc.collect()
    
    cc_df = joblib.load(os.path.join(prediction_folder, "credit_card.joblib"))
    transformed_data = pd.merge(transformed_data, cc_df, on='SK_ID_CURR', how='left')
    del cc_df
    gc.collect()

    transformed_data = reduce_memory(transformed_data)
    transformed_data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in transformed_data.columns]
    
    del_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    
    predictors = [feat for feat in transformed_data.columns if feat not in del_features]
    with open(os.path.join(prediction_folder, "features.txt"), "w") as output:
        for feat in predictors:
            output.write(feat +"\n")

    train, valid, test = np.split(transformed_data.sample(frac=1),
                                  [int(0.80 * len(transformed_data)), int(0.90 * len(transformed_data))])
    del transformed_data
    gc.collect()
    
    train_y = train['TARGET'].values
    train_X = train[predictors].values
    valid_y = valid['TARGET'].values
    valid_X = valid[predictors].values
    
    created_date = str(datetime.today())
#     created_date_v = created_date.strftime("%m_%d_%Y_%H_%M_%S")
    
    params = {'random_state': config.RANDOM_SEED, 'nthread': config.NUM_THREADS}
    clf = LGBMClassifier(**{**params, **config.LIGHTGBM_PARAMS})
    clf = clf.fit(train_X, train_y,eval_set=[(train_X, train_y), (valid_X, valid_y)],
                      eval_metric='auc',verbose=200, early_stopping_rounds=100,
                      feature_name=predictors,
                      categorical_feature=config.CATEGORICAL_FEAT)
    joblib.dump(clf, os.path.join(prediction_folder, "lgb_model.joblib".format(created_date_v)))

    del train_X
    del train_y
    del valid_X
    del valid_y
    gc.collect()
    
    test_y = test["TARGET"].values
    test_X = test[predictors].values
    test_pred = clf.predict_proba(test_X)[:,1]
    auc = roc_auc_score(test_y, test_pred)
    
    
    
    version = hashlib.sha256(str.encode(created_date)).hexdigest()
    meta_data = {'version': version, 'created_at': created_date}
    joblib.dump(meta_data, os.path.join(prediction_folder, "meta_data.joblib"))
    print('Done training, AUC score on test set is {:.2f}'.format(auc))