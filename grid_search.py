from sklearn.model_selection import KFold
import lightgbm as lgb
import gc;
gc.enable()
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



fold = 1
train_dropped = train.drop(['msno'], axis=1)
test_dropped = test.drop(['msno'], axis=1)
eta = [0.01, 0.02, 0.03, 0.05, 0.06, 0.1]
max_depth = [3,5,7,10]
for i in range(fold):
    for i in range(len(eta)):
        for j in range(len(max_depth)):
            params = {
        'nrounds': 400,
        'eta': eta[i],
        'max_depth': max_depth[j],
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 100,
        'silent': True
        }
            x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
            watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
            model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50)
            if i != 0:
                pred1 += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
            else:
                pred1 = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
            pred1 /= fold
            test['is_churn'] = pred1.clip(0.+1e-15, 1-1e-15)
            test[['msno','is_churn']].to_csv('xgbsub'+ str(i) + str(j)+ '.csv.gz', index=False, compression='gzip')