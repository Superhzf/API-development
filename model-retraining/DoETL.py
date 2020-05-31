from HomeCreditDefaultPrediction.prediction.train_pipelines import config
from HomeCreditDefaultPrediction.prediction.train_pipelines import bureau_pipeline
from HomeCreditDefaultPrediction.prediction.train_pipelines import previous_pipeline
from HomeCreditDefaultPrediction.prediction.train_pipelines import previous_balance_pipeline
import joblib
import os
import gc


bureau_df = bureau_pipeline.get_bureau(config.DATA_FOLDER, num_rows=None)
joblib.dump(bureau_df, os.path.join(config.DATA_FOLDER, "bureau_and_balance.joblib"))
del bureau_df
gc.collect()
print('Done ETL bureau data!')

prev_df = previous_pipeline.get_previous_applications(config.DATA_FOLDER, num_rows=None)
joblib.dump(prev_df, os.path.join(config.DATA_FOLDER, "previous.joblib"))
del prev_df
gc.collect()
print('Done ETL previous application data!')

pos_df = previous_balance_pipeline.get_pos_cash(config.DATA_FOLDER, num_rows=None)
joblib.dump(pos_df, os.path.join(config.DATA_FOLDER, "pos_cash.joblib"))
del pos_df
gc.collect()
print('Done ETL point of sale data!')

ins_df = previous_balance_pipeline.get_installment_payments(config.DATA_FOLDER, num_rows=None)
joblib.dump(ins_df, os.path.join(config.DATA_FOLDER, "payments.joblib"))
del ins_df
gc.collect()
print('Done ETL installment payment data!')

cc_df = previous_balance_pipeline.get_credit_card(config.DATA_FOLDER, num_rows=None)
joblib.dump(cc_df, os.path.join(config.DATA_FOLDER, "credit_card.joblib"))
del cc_df
gc.collect()
print('Done ETL credit card data!')
