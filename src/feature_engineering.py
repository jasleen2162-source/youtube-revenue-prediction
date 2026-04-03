def create_features(df):
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']

    df['watch_time_per_view'] = df['watch_time_minutes'] / df['views']

    return df