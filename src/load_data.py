import pandas as pd

from .config import PRICE_CSV, TEXT_CSV, INDEX_TZ
from dateutil.tz import gettz
from datetime import time

def load_prices():
    df = pd.read_csv(PRICE_CSV, parse_dates=['date'])
    df = df.sort_values('date')
    df['r'] = df['close'].pct_change()
    return df[['date', 'r']].dropna().set_index('date')

def load_text_events():
    df = pd.read_csv(TEXT_CSV)
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True).dt.tz_convert(INDEX_TZ)
    return df
