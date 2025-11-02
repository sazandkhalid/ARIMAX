import pandas as pd
from .config import ROLL_MEAN_K, USE_SENTIMENT_SURPRISE

def prepare_features(rets_df, sent_df):
    df = rets_df.join(sent_df, how="left")
    df['s'] = df['s_raw'].fillna(0)

    if USE_SENTIMENT_SURPRISE:
        df['s'] = df['s'] - df['s'].rolling(ROLL_MEAN_K).mean()

    df['s'] = df['s'].fillna(0)
    return df
