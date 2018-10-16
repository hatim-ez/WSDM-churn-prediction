from sklearn.model_selection import KFold
import lightgbm as lgb
import gc;
gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
from transactions_process import *
from user_logs_process import *

#True if we should replace all transactions for a same msno into an aggregated transaction instead of keeping just the latest
aggregate_transactions = False

#if date time is not imported (as in the machine we had access to), we have the file user_logs_V4.csv that contains the preprocessing
datetime_imported = False

train = pd.read_csv('input/train.csv')
train = pd.concat((train, pd.read_csv('input/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

test = pd.read_csv('input/sample_submission_v2.csv')
members = pd.read_csv('input/members_v3.csv')
transactions = pd.read_csv('input/transactions.csv')
transactions = pd.concat((transactions, pd.read_csv('input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

if aggregate_transactions:
    transactions = preprocess_transactions_aggregate(transactions)
transactions = preprocess_transactions(transactions)


train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis=0)

combined = pd.merge(combined, members, how='left', on='msno')
members = []
print('members merge...')

gender = {'male': 1, 'female': 2}
combined['gender'] = combined['gender'].map(gender)

combined = pd.merge(combined, transactions, how='left', on='msno')
transactions = []
print('transaction merge...')

train = combined[combined['is_train'] == 1]
test = combined[combined['is_train'] == 0]

train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train'], axis=1, inplace=True)

del combined

if datetime_imported:
    last_user_logs = preprocess_user_logs(pd.read_csv('input/user_logs_v2.csv'))
else:
    last_user_logs = pd.read_csv("input/user_logs_v4.csv")

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
last_user_logs = []
if aggregate_transactions:
    train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.number_cancellation == 0)).astype(np.int8)
    test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.number_cancellation == 0)).astype(np.int8)

    train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.number_cancellation != 0)).astype(np.int8)
    test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.number_cancellation != 0)).astype(np.int8)
else:
    train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
    test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

    train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel != 0)).astype(np.int8)
    test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel != 0)).astype(np.int8)
train = train.fillna(0)
test = test.fillna(0)

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)



k=2
train_dropped = train.drop(['msno'], axis=1)
test_dropped = test.drop(['msno'], axis=1)
for i in range(k):
    kf = KFold(n_splits=k)
    params = {'learning_rate': 0.02, 'max_depth': 7, 'lambda_l1': 16.7, 'objective': 'binary', 'metric': 'logloss',
              'max_bin': 1000, 'feature_fraction': .7, 'is_training_metric': False, 'seed': 99}
    y_pred=[0]*test.shape[0]
    X_pred=test_dropped

    for train1, test1 in kf.split(train_dropped):
        train_data = train_dropped.loc[train1]
        test_data = train_dropped.loc[test1]
        X_train = train_data
        X_test = test_data
        y_train = train_data['is_churn']
        y_test = test_data['is_churn']

        #lgb
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        print('Start training...')
        # train
        model = lgb.train(params, lgb_train, num_boost_round=1200, valid_sets=lgb_eval, verbose_eval=50)
                          #early_stopping_rounds=200)
        print('Start predicting...')
        # predict
        y_eval = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred += model.predict(X_pred, num_iteration=model.best_iteration)

        #xgb
        x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        params2 = {
            'nrounds': 600,
            'eta': 0.02,
            'max_depth': 7,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 100,
            'silent': True
        }
        model = xgb.train(params2, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50)
        if i != 0:
            y_pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
        else:
            y_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

y_pred /= 2*k
test['is_churn'] = y_pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('output/xgb_lgb_sub_preprocessed.csv.gz', index=False, compression='gzip')
