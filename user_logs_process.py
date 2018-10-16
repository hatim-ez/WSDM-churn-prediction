import pandas as pd


def preprocess_user_logs(user_logs_df):
    user_logs_df = user_logs_df.sort_values(by=['msno', 'date'], ascending=[True, False])
    user_logs_df = user_logs_df.reset_index(drop=True)
    print("1er test...")
    user_logs_df_max_date = user_logs_df.groupby('msno').agg({'date': 'first'}).rename(columns={'date': 'max_date'})

    user_logs_df = user_logs_df.join(user_logs_df_max_date, on='msno')

    user_logs_df['diff_date'] = (pd.to_datetime(user_logs_df['max_date'], format='%Y%m%d') - pd.to_datetime(user_logs_df['date'], format='%Y%m%d')).dt.days

    #cols = ['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']
    user_logs_df["diff_date"] = user_logs_df['diff_date'] + 1
    user_logs_df['score_25'] = user_logs_df['num_25'] / user_logs_df['diff_date']
    user_logs_df['score_50'] = user_logs_df['num_50'] / user_logs_df['diff_date']
    user_logs_df['score_75'] = user_logs_df['num_75'] / user_logs_df['diff_date']
    user_logs_df['score_985'] = user_logs_df['num_985'] / user_logs_df['diff_date']
    user_logs_df['score_100'] = user_logs_df['num_100'] / user_logs_df['diff_date']

    user_logs_df.groupby('msno').agg('sum')

    user_logs_df['score_25'] = user_logs_df['score_25'] / (user_logs_df['num_25'] + 1)
    user_logs_df['score_50'] = user_logs_df['score_50'] / (user_logs_df['num_50'] + 1)
    user_logs_df['score_75'] = user_logs_df['score_75'] / (user_logs_df['num_75'] + 1)
    user_logs_df['score_985'] = user_logs_df['score_985'] / (user_logs_df['num_985'] + 1)
    user_logs_df['score_100'] = user_logs_df['score_100'] / (user_logs_df['num_100'] + 1)

    user_logs_df['num_25'] = user_logs_df['num_25'] / user_logs_df['num_unq']
    user_logs_df['num_50'] = user_logs_df['num_25'] / user_logs_df['num_unq']
    user_logs_df['num_75'] = user_logs_df['num_25'] / user_logs_df['num_unq']
    user_logs_df['num_985'] = user_logs_df['num_25'] / user_logs_df['num_unq']
    user_logs_df['num_100'] = user_logs_df['num_100'] / user_logs_df['num_unq']

    user_logs_df = user_logs_df.drop(['diff_date', 'date', 'max_date'], axis=1)

    print(user_logs_df.head())
    return user_logs_df
