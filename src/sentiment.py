import pandas as pd
from dateutil.tz import gettz
from datetime import time
from .config import MARKET_CLOSE, INDEX_TZ

def simple_sentiment_score(text):
    text = str(text).lower()
    pos = any(w in text for w in ["beat","beats","strong","raise","optimistic","bullish"])
    neg = any(w in text for w in ["miss","weak","cut","pessimistic","bearish"])
    return (1 if pos else 0) + (-1 if neg else 0)

def build_daily_sentiment(df):
    tz = gettz(INDEX_TZ)

    def trading_day(ts):
        local = ts.astimezone(tz)
        d = local.date()
        if local.time() > time(*MARKET_CLOSE):
            return pd.Timestamp(d) + pd.Timedelta(days=1)
        return pd.Timestamp(d)

    df['s_raw'] = df['text'].map(simple_sentiment_score)
    df['trade_day'] = df['published_at'].map(trading_day)

    daily = df.groupby('trade_day')['s_raw'].mean().reset_index()
    daily = daily.rename(columns={'trade_day': 'date'})
    daily['date'] = pd.to_datetime(daily['date'])
    return daily.set_index('date').sort_index()
