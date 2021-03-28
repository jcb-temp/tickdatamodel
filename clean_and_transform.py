import pandas as pd
import numpy as np

# ======================
# Functions Defined: ===
# ======================
# 
# format_columns_as_date
# zero_size_when_bid_gt_ask
# restrict_to_datetime_range


def format_columns_as_date(df, list_of_colnames, date_format):
    # Quick function that takes a list of columns and makes sure they are of datetime format
    # Does no checking of whether they _should_ be in datetime format

    # first, is this actually a list? if it's a string, make it a list
    list_of_colnames = [list_of_colnames] if isinstance(list_of_colnames, str) else list_of_colnames

    # now loop over the list and convert each column
    for colname in list_of_colnames:
        df[colname]= pd.to_datetime(df[colname], format = date_format)

    return df


def zero_size_when_bid_gt_ask(df, list_of_column_name_dictionaries):
    # Function to zero sizes when the ask < bid price
    # 
    # expect a list of dictionaries. Each dictionary contains three entries:
    # bidpricecol = name of the bid price column
    # askpricecol = name of the ask price column
    # bidsizecol = name of the bid size column
    # asksizecol = name of the ask size column
    # 
    # e.g.
    # {"bidpricecol":"L1_BidPrice", "askpricecol": "L1_AskPrice", "bidsizecol":"L1_BidSize", "asksizecol": "L1_AskSize"}

    for colnamedict in list_of_column_name_dictionaries:
        # get the bid and ask price and size column names
        bpc = colnamedict["bidpricecol"]
        apc = colnamedict["askpricecol"]
        bsc = colnamedict["bidsizecol"]
        asc = colnamedict["asksizecol"]

        # calculate a mask for this situation
        mask = df[apc] < df[bpc]

        # apply the mask to zero out the sizes
        df.loc[mask, bsc] = 0
        df.loc[mask, asc] = 0

    return df


def restrict_to_datetime_range(df, min_datetime, max_datetime, datetime_column = ""):
    # Takes a dataframe and restricts to rows within a certain datetime range.
    # note this is an inclusive restriction
    # The function prefers to use a datetime index, otherwise it uses a mask
    # datetime_column is only needed if there is no datetime index

    # first check if we have a pandas datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        # if so, this is easy and fast
        return df[min_datetime:max_datetime]
    else:
        # if not, use a mask
        mask = (df[datetime_column] >= min_datetime) & (df[datetime_column] <= max_datetime)
        return df.loc[mask]