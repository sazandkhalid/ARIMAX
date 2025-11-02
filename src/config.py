import numpy as np

SEED = 123
np.random.seed(SEED)

PRICE_CSV = "data/prices.csv"
TEXT_CSV = "data/text_events.csv"

INDEX_TZ = "America/New_York"
MARKET_CLOSE = (16, 0)  # 4pm ET

TRAIN_YEARS = 3
TEST_MONTHS = 3
EMBARGO_DAYS = 3

ROLL_MEAN_K = 20
USE_SENTIMENT_SURPRISE = True
