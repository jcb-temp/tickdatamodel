import pandas as pd
import numpy as np

# ======================
# Functions Defined: ===
# ======================
# 
# calculate_mid_prices
# calculate_costs
# value_variance_in_window
# calculate_level_n_value_variance_by_day
# calculate_total_bid_size_by_day
# calculate_level_n_total_bid_size_by_day
# create_fake_orders
# fulfil_order
# fulfil_list_of_orders

def calculate_mid_prices(df, list_of_column_name_dictionaries):
    # Function to calculate all the mid prices for each level
    # 
    # expect a list of dictionaries. Each dictionary contains three entries:
    # bidpricecol = name of the bid price column
    # askpricecol = name of the ask price column
    # midpricecol = name for the new mid price column
    # 
    # e.g.
    # {"bidpricecol":"L1_BidPrice", "askpricecol": "L1_AskPrice", "midpricecol": "L1_MidPrice"}

    for colnamedict in list_of_column_name_dictionaries:
        # get the bid, ask, and mid column names
        bpc = colnamedict["bidpricecol"]
        apc = colnamedict["askpricecol"]
        mpc = colnamedict["midpricecol"]

        # find the average and assign to the new column
        df[mpc] = (df[bpc] + df[apc])/2

    return df


def calculate_costs(df, list_of_column_name_dictionaries):
    # Function to calculate the costs for each level
    # 
    # expect a list of dictionaries. Each dictionary contains three entries:
    # bidpricecol = name of the bid price column
    # midpricecol = name of the mid price column
    # costcol     = name for the new cost column
    # 
    # e.g.
    # {"bidpricecol":"L1_BidPrice", "midpricecol": "L1_MidPrice", "costcol": "L1_Cost"}

    for colnamedict in list_of_column_name_dictionaries:
        # get the bid, mid, and cost column names
        bpc = colnamedict["bidpricecol"]
        mpc = colnamedict["midpricecol"]
        cc = colnamedict["costcol"]

        # find the difference and assign to the new column
        df[cc] = df[mpc] - df[bpc]

    return df


def value_variance_in_window(df):
    # returns the variance of all values within the df

    values = []
    for ii in range (1,11):
        midpricecol = "L"+str(ii)+"_MidPrice"
        values += list(df.loc[:,midpricecol])
    return np.var(values)


def calculate_level_n_value_variance_by_day(df, n):
    #  Function to add a column containing the variance of level n values by day

    # Start with a new column containing the day to group by
    df.loc[:,'Time_Day'] = df['Time_Minute'].dt.date

    # Need the column with values
    value_col = "L"+str(n)+"_MidPrice"
    variance_col = "L"+str(n)+"_variance"

    # Copy across to a new column
    df.loc[:,variance_col] = df[value_col]

    # Next we want to get the variance of variance_col within each day
    # So we start by grouping by day
    df_g = df.loc[:,[variance_col, "Time_Day"]].groupby("Time_Day", as_index=False)
    # and aggregate
    df_g = df_g.agg({variance_col : np.var})

    # finally merge this back to the original data, drop the day column
    return pd.merge(df.drop(variance_col,axis=1), df_g, on=["Time_Day"], how='inner').drop("Time_Day",axis=1)


def calculate_total_bid_size_by_day(df):
    # Function to add a column containing the total daily sum of bid sizes across all levels

    # Start with a new column containing the day to group by
    df.loc[:,'Time_Day'] = df['Time_Minute'].dt.date

    # Need a list of the columns with bid sizes
    bid_size_cols = [ "L"+str(ii)+"_BidSize" for ii in range(1,11)]

    # Sum across all the bid size columns 
    df.loc[:,"TotalBidSize"] = df[bid_size_cols].sum(axis=1)

    # Next we want to sum TotalBidSize up within each day
    # So we start by grouping by day
    df_g = df.loc[:,["TotalBidSize", "Time_Day"]].groupby("Time_Day", as_index=False)
    # and aggregate
    df_g = df_g.agg({'TotalBidSize' : np.sum})

    # finally merge this back to the original data, drop the day column
    return pd.merge(df.drop("TotalBidSize",axis=1), df_g, on=["Time_Day"], how='inner').drop("Time_Day",axis=1)


def calculate_level_n_total_bid_size_by_day(df, n):
    # Function to add a column containing the daily sum of bid sizes for level n

    # Start with a new column containing the day to group by
    df.loc[:,'Time_Day'] = df['Time_Minute'].dt.date

    # Need the column with bid sizes
    bid_size_col = "L"+str(n)+"_BidSize"
    total_bid_size_col = "L"+str(n)+"_TotalBidSize"

    # Copy across to a total bid size column
    df.loc[:,total_bid_size_col] = df[bid_size_col]

    # Next we want to sum Ln_TotalBidSize up within each day
    # So we start by grouping by day
    df_g = df.loc[:,[total_bid_size_col, "Time_Day"]].groupby("Time_Day", as_index=False)
    # and aggregate
    df_g = df_g.agg({total_bid_size_col : np.sum})

    # finally merge this back to the original data, drop the day column
    return pd.merge(df.drop(total_bid_size_col,axis=1), df_g, on=["Time_Day"], how='inner').drop("Time_Day",axis=1)


def create_fake_orders(
    df,
    num_orders = 1, 
    min_quantity = 1, 
    max_quantity = 1, 
    min_horizon = 1,
    max_horizon = 1,
    min_datetime = "2018-02-05 09:00:00",
    max_datetime = "2018-06-12 16:00:00"):

    # This function makes a list of order dictionaries.
    # 
    # Each dictionary contains:
    # - id: integer id of this order
    # - quantity: quantity to sell
    # - horizon: maximum number of minutes over which the sale must take place
    # - start_datetime: starting datetime for the sale
    # 
    # these are made at random using the input min/max restrictions given
    # a total of num_orders are generated

    # preconvert min and max datetimes into minutes since 1970-01-01
    unix_epoch = pd.to_datetime("1970-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    min_datetime_minutes_since_unix_epoch = (pd.to_datetime(min_datetime, format="%Y-%m-%d %H:%M:%S") - unix_epoch) // pd.Timedelta('1m')
    max_datetime_minutes_since_unix_epoch = (pd.to_datetime(max_datetime, format="%Y-%m-%d %H:%M:%S") - unix_epoch) // pd.Timedelta('1m')
    
    list_of_orders = []
    for ii in range(int(num_orders)):
        # easy ones first: random quantity and horizon
        quantity = np.random.randint(min_quantity,max_quantity)
        horizon = np.random.randint(min_horizon,max_horizon)

        # Make a random start time for the trade that exists in our data
        date_in_index = False
        while not date_in_index:
            minutes_in_time_window = np.random.randint(min_datetime_minutes_since_unix_epoch,max_datetime_minutes_since_unix_epoch)
            start_datetime = unix_epoch + pd.DateOffset(minutes = minutes_in_time_window)

            date_in_index = start_datetime in df.index

        order_dict = {
            "id": ii,
            "quantity": quantity,
            "horizon": horizon,
            "start_datetime": start_datetime
        }

        list_of_orders += [order_dict]

    return list_of_orders


def fulfil_order(df, order_dict, debug_printing = False):
    # Take in a dictionary representation of an order, and calculate the cost associated with it

    # First pull out some key values
    quantity = order_dict["quantity"]
    start_datetime = order_dict["start_datetime"]
    end_datetime = order_dict["start_datetime"] + pd.DateOffset(minutes = order_dict["horizon"])

    # subset to the time window
    df_window = df.loc[start_datetime:end_datetime].copy()

    # find the actual min and max times
    actual_min_datetime = min(df_window["Time_Minute"])
    actual_max_datetime = max(df_window["Time_Minute"])

    # Find some extra info
    L1_variance = df_window.loc[start_datetime,"L1_variance"]
    L1_TotalBidSize = df_window.loc[start_datetime,"L1_TotalBidSize"]
    TotalBidSize = df_window.loc[start_datetime,"TotalBidSize"]

    # start to keep track of cost and price
    total_price = 0
    total_cost = 0
    
    if debug_printing: print("  Total Quantity "+str(quantity))

    # assuming perfect knowledge, we fulfil orders from the smallest cost upwards
    # loop upwards through the levels
    for ii in range(1,11):
        if debug_printing: print("  Level "+str(ii))

        current_cost_col = "L"+str(ii)+"_Cost"
        current_bidsize_col = "L"+str(ii)+"_BidSize"
        current_midprice_col = "L"+str(ii)+"_MidPrice"

        # sort by costs, low to high (here's our perfect knowledge)
        df_window.sort_values(current_cost_col, ascending=True, inplace = True)

        # work through all minutes in this level, and trade as much as possible in each
        for curr_index, curr_row in df_window.iterrows():
            num_on_offer = curr_row[current_bidsize_col]
            if num_on_offer < quantity:
                # Not enough to finish, so trade all of them and continue
                if debug_printing: print("    Traded "+str(num_on_offer)+" at cost "+str(curr_row[current_cost_col])+" for price "+str(curr_row[current_midprice_col]))

                # reduce quantity left to trade by the number available
                quantity -= num_on_offer

                # increase our total price and cost accordingly
                total_cost += num_on_offer*curr_row[current_cost_col]
                total_price += num_on_offer*curr_row[current_midprice_col]

            else:
                # There are enough here to finish, so trade what we need to, 
                # update totals using remaining quantity, set quantity to zero, and break
                if debug_printing: print("    Traded "+str(quantity)+" at cost "+str(curr_row[current_cost_col])+" for price "+str(curr_row[current_midprice_col]))

                total_cost += quantity*curr_row[current_cost_col]
                total_price += quantity*curr_row[current_midprice_col]
                quantity = 0

                break
        
        # check if we are done
        if quantity == 0:
            # nothing left to trade, so break
            if debug_printing: print("    Done!") 
            break

    # check whether we finished trading within the window
    success = quantity == 0

    fulfilment_dict = {
        "id": order_dict["id"],
        "success": success,
        "quantity": order_dict["quantity"],
        "horizon": order_dict["horizon"],
        "total_price": total_price,
        "total_cost": total_cost,
        "actual_min_datetime": actual_min_datetime,
        "actual_max_datetime": actual_max_datetime,
        "L1_variance": L1_variance,
        "L1_TotalBidSize": L1_TotalBidSize,
        "TotalBidSize": TotalBidSize
    }

    return fulfilment_dict


def fulfil_list_of_orders(df, list_of_order_dicts):
    # takes a list of order dictionaries, fulfils them, and returns 
    # a dataframe with the statistics of those fulfilments

    # cols_for_output = set()    
    # for order_dict in list_of_order_dicts:
    #     cols_for_output = cols_for_output.union(set(order_dict.keys()))

    

    # # add extras
    # cols_for_output = cols_for_output.union(set(['total_cost_percentage']))
    # print(list(cols_for_output))
    
    out_df = pd.DataFrame(columns=["id","success","quantity","horizon","total_price","total_cost","total_cost_percentage","actual_min_datetime","actual_max_datetime","L1_variance","L1_TotalBidSize","TotalBidSize"])

    for order_dict in list_of_order_dicts:

        fulfilment_dict = fulfil_order(df, order_dict)

        if fulfilment_dict["success"]:
            if fulfilment_dict["total_price"] == 0:
                print("issue with order id "+str(fulfilment_dict["id"])+": total_price is zero")

            fulfilment_dict["total_cost_percentage"] = fulfilment_dict["total_cost"]/fulfilment_dict["total_price"]
            out_df.loc[fulfilment_dict["id"]] = pd.Series(fulfilment_dict)

    return out_df