import numpy as np

def sign_strategy(df, tc_bps=5):
    pos = (df['mu_with'] > 0).astype(int)
    prev = pos.shift(1).fillna(0)
    turns = (pos != prev).astype(int)
    cost = turns * (tc_bps/10000.0)

    pnl = pos * df['r_real'] - cost
    sharpe = np.sqrt(252) * pnl.mean() / pnl.std()
    
    return {
        "sharpe": sharpe,
        "return_daily": pnl.mean(),
        "turnover": turns.mean()*252
    }
