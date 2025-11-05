import pandas as pd

def rolling_splits(
    dates: pd.Index,
    train_years: int = 3,
    test_months: int = 3,
    embargo_days: int = 3
):
    """
    Generator that yields (train_mask, test_mask) boolean masks
    for rolling-window time series forecasting.

    Parameters
    ----------
    dates : pd.Index
        Sorted datetime index for your dataset.
    train_years : int
        Size of training period in years.
    test_months : int
        Size of test period in months.
    embargo_days : int
        Gap between training and testing to prevent leakage.

    Yields
    ------
    (train_mask, test_mask) : tuple of boolean arrays
    """

    dates = pd.Index(dates)
    start = dates.min()
    end = dates.max()

    # start anchor: first end of training window
    anchor = start + pd.DateOffset(years=train_years)

    while anchor + pd.DateOffset(months=test_months) <= end:
        train_end = anchor
        test_start = anchor + pd.Timedelta(days=embargo_days)
        test_end = anchor + pd.DateOffset(months=test_months)

        train_mask = (dates >= start) & (dates <= train_end)
        test_mask = (dates > test_start) & (dates <= test_end)

        yield train_mask, test_mask
        anchor = anchor + pd.DateOffset(months=test_months)
