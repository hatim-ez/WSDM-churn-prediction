import pandas as pd


def preprocess_transactions_aggregate(transactions_df):
    transactions_df['transactions_per_user'] = transactions_df['is_cancel']
    transactions_df = transactions_df.sort_values(by=['msno', "transaction_date"], ascending=[True, False]).reset_index(
        drop=True)
    transactions_df = transactions_df.groupby('msno').agg(
        {'is_cancel': 'sum', 'payment_plan_days': 'sum', 'payment_method_id': 'first',
         'plan_list_price': 'first', 'actual_amount_paid': 'first',
         'is_auto_renew': 'first', 'transaction_date': 'first',
         'membership_expire_date': 'first', 'transactions_per_user': 'count'}) \
        .rename(columns={'is_cancel': 'number_cancellation', 'payment_plan_days': 'total_length'})
    transactions_df['cancellation_rate'] = transactions_df['number_cancellation'] / transactions_df['transactions_per_user']

    return transactions_df

def preprocess_transactions(transactions_df):
    transactions_df = transactions_df.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
    transactions_df = transactions_df.drop_duplicates(subset=['msno'], keep='first')
    transactions_df['amt_per_day'] = transactions_df['actual_amount_paid'] / transactions_df['payment_plan_days']
    transactions_df['discount'] = transactions_df['plan_list_price'] - transactions_df['actual_amount_paid']
    transactions_df['is_discount'] = transactions_df.discount.apply(lambda x: 1 if x > 0 else 0)
    transactions_df['membership_days'] = pd.to_datetime(transactions_df['membership_expire_date']).subtract(
        pd.to_datetime(transactions_df['transaction_date'])).dt.days.astype(int)
    return transactions_df

def keep_last_transaction(transactions_df):
    transactions_df = transactions_df.sort_values(by=['date'], ascending=[False])
    transactions_df = transactions_df.reset_index(drop=True)
    transactions_df = transactions_df.drop_duplicates(subset=['msno'], keep='first')
    return transactions_df

