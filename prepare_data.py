# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:47:06 2022

@author: ling
"""

import pandas as pd
import numpy as np
import re
import os
import datetime

from itertools import product

def prepare_price(df_price):
    
    df_price[['Close', 'AdjustmentFactor']] = df_price.groupby('SecuritiesCode')[['Close', 'AdjustmentFactor']].fillna(method='ffill')
    for price_col in ['Open', 'High', 'Low']:
        df_price[price_col] = np.where(df_price.Volume>0, df_price[price_col], df_price.Close)
    
    df_price['Amplitude'] = (df_price['High'] - df_price['Low']) / df_price.groupby('SecuritiesCode')['Close'].shift(1)
    df_price['Volume(Yen)'] = df_price['Volume'] * df_price['Close']
    df_price['Volume_Rolling_5_mean'] = df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: s.rolling(5).mean())
    df_price['Volume_Rolling_5_rank'] = df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: s.rolling(5).rank(ascending=False))
    df_price['Volume_Rolling_5_zero_count'] = df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: (s==0).rolling(5).sum())
    df_price['Volume_Rolling_5_SE'] = (
        (df_price['Volume(Yen)'] - df_price['Volume_Rolling_5_mean'])
        / df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: s.rolling(5).std())
    )
    
    df_price['Volume_Rolling_20_mean'] = df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: s.rolling(20).mean())
    df_price['Volume_Rolling_20_rank'] = df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: s.rolling(20).rank(ascending=False))
    df_price['Volume_Rolling_20_zero_count'] = df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: (s==0).rolling(20).sum())
    df_price['Volume_Rolling_20_SE'] = (
        (df_price['Volume(Yen)'] - df_price['Volume_Rolling_20_mean'])
        / df_price.groupby('SecuritiesCode')['Volume(Yen)'].transform(lambda s: s.rolling(20).std())
    )
    
    df_price['Latest_daily_return_custom'] = (
        df_price['Close'] /
        df_price.groupby('SecuritiesCode')[['Close', 'AdjustmentFactor']].shift(1).product(min_count=2, axis=1)
        - 1
    )
    
    df_price['Latest_weekly_return_custom'] = (
        df_price['Close'] /
        df_price.groupby('SecuritiesCode')[['Close', 'AdjustmentFactor']].shift(5).product(min_count=2, axis=1)
        - 1
    )
    
    df_price['Daily_return_Rolling_5_mean'] = df_price.groupby('SecuritiesCode')['Latest_daily_return_custom'].transform(lambda s: s.rolling(5).mean())
    df_price['Daily_return_Rolling_5_rank'] = df_price.groupby('SecuritiesCode')['Latest_daily_return_custom'].transform(lambda s: s.rolling(5).rank(ascending=False))
    df_price['Daily_return_Rolling_5_std'] = df_price.groupby('SecuritiesCode')['Latest_daily_return_custom'].transform(lambda s: s.rolling(5).std())
    df_price['Daily_return_Rolling_5_SE'] = (
        (df_price['Latest_daily_return_custom'] - df_price['Daily_return_Rolling_5_mean']) /
        df_price['Daily_return_Rolling_5_std']
    )
    df_price['Daily_return_Rolling_20_mean'] = df_price.groupby('SecuritiesCode')['Latest_daily_return_custom'].transform(lambda s: s.rolling(20).mean())
    df_price['Daily_return_Rolling_20_rank'] = df_price.groupby('SecuritiesCode')['Latest_daily_return_custom'].transform(lambda s: s.rolling(20).rank(ascending=False))
    df_price['Daily_return_Rolling_20_std'] = df_price.groupby('SecuritiesCode')['Latest_daily_return_custom'].transform(lambda s: s.rolling(20).std())
    df_price['Daily_return_Rolling_20_SE'] = (
        (df_price['Latest_daily_return_custom'] - df_price['Daily_return_Rolling_20_mean']) /
        df_price['Daily_return_Rolling_20_std']
    )
    return df_price

def get_sector_return_from_price(df_price, sector_col):
    df_date_sector_return = (
     df_primary_price
     .assign(**{
         'Volume_return': lambda df: df['Volume(Yen)'] * df['Latest_daily_return_custom'],
         'Volume_return_5_mean': lambda df: df['Volume_Rolling_5_mean'] * df['Daily_return_Rolling_5_mean'],
         'Volume_return_20_mean': lambda df: df['Volume_Rolling_20_mean'] * df['Daily_return_Rolling_20_mean'],
     })
     .groupby(['Date', sector_col])[[
         'Volume_return', 'Volume_return_5_mean', 'Volume_return_20_mean', 'Volume(Yen)', 'Volume_Rolling_5_mean', 'Volume_Rolling_20_mean', 
     ]]
     .sum()
     .assign(**{
         f'Volume_weighted_{sector_col}_return': lambda df: df['Volume_return'] / df['Volume(Yen)'],
         f'Volume_weighted_{sector_col}_rolling_5_return': lambda df: df['Volume_return_5_mean'] / df['Volume_Rolling_5_mean'],
         f'Volume_weighted_{sector_col}_rolling_20_return': lambda df: df['Volume_return_20_mean'] / df['Volume_Rolling_20_mean']
     }).fillna(0).filter(regex=f'Volume_weighted_{sector_col}_.*')
    )
    return df_date_sector_return

def prepare_option(df_option):
    df_option["OptionsCode"] = df_option.OptionsCode.astype(str)
    df_option["DerivativeType"] = df_option['OptionsCode'].astype(str).str.slice(1,2)
    # All underlying is Nikkei Stock Average (Nikkei 225)
    most_traded_options_by_date = (
        df_option.groupby(['Date', 'StrikePrice'])['TradingVolume'].sum()
        .groupby(level=0).apply(lambda s: s.sort_values().index.get_level_values(1)[-1])
    ).to_frame()
    most_traded_options_by_date.columns = ["MostTradedStrike"]
    most_traded_options_by_date.reset_index(inplace=True)
    most_traded_options_by_date['NextContractMonth'] = (pd.to_datetime(most_traded_options_by_date.Date) + datetime.timedelta(days=30)).dt.strftime("%Y%m")
    
    df_option = pd.merge(df_option, most_traded_options_by_date, on=["Date"], how="left")
    df_option_most_traded = df_option[
        (df_option.StrikePrice==df_option.MostTradedStrike) &
        (df_option.ContractMonth.astype(str)==df_option.NextContractMonth)
    ]
    option_detail_columns = ['TradingValue','OpenInterest','ImpliedVolatility','BaseVolatility',]
    option_detail_rolling_columns = ['TradingValue','OpenInterest','ImpliedVolatility','BaseVolatility', 'ImpliedVolatilityDifference']
    df_put_option_most_traded = df_option_most_traded[df_option_most_traded.Putcall==1][['Date'] + option_detail_columns].set_index('Date')
    df_put_option_most_traded['ImpliedVolatilityDifference'] = df_put_option_most_traded['ImpliedVolatility'] - df_put_option_most_traded['BaseVolatility']
    df_put_option_most_traded[[f"{col}_Rolling_5_mean" for col in option_detail_rolling_columns]] = df_put_option_most_traded[option_detail_rolling_columns].rolling(5).mean()
    df_put_option_most_traded[[f"{col}_Rolling_5_rank" for col in option_detail_rolling_columns]] = df_put_option_most_traded[option_detail_rolling_columns].rolling(5).rank()
    df_put_option_most_traded[[f"{col}_Rolling_20_mean" for col in option_detail_rolling_columns]] = df_put_option_most_traded[option_detail_rolling_columns].rolling(20).mean()
    df_put_option_most_traded[[f"{col}_Rolling_20_rank" for col in option_detail_rolling_columns]] = df_put_option_most_traded[option_detail_rolling_columns].rolling(20).rank()
    df_put_option_most_traded.columns = [f"Put_{col}" for col in df_put_option_most_traded.columns]                       
       
    df_call_option_most_traded = df_option_most_traded[df_option_most_traded.Putcall==2][['Date'] + option_detail_columns].set_index('Date')
    df_call_option_most_traded['ImpliedVolatilityDifference'] = df_call_option_most_traded['ImpliedVolatility'] - df_call_option_most_traded['BaseVolatility']
    df_call_option_most_traded[[f"{col}_Rolling_5_mean" for col in option_detail_rolling_columns]] = df_call_option_most_traded[option_detail_rolling_columns].rolling(5).mean()
    df_call_option_most_traded[[f"{col}_Rolling_5_rank" for col in option_detail_rolling_columns]] = df_call_option_most_traded[option_detail_rolling_columns].rolling(5).rank()
    df_call_option_most_traded[[f"{col}_Rolling_20_mean" for col in option_detail_rolling_columns]] = df_call_option_most_traded[option_detail_rolling_columns].rolling(5).mean()
    df_call_option_most_traded[[f"{col}_Rolling_20_rank" for col in option_detail_rolling_columns]] = df_call_option_most_traded[option_detail_rolling_columns].rolling(5).rank()
    df_call_option_most_traded.columns = [f"Call_{col}" for col in df_call_option_most_traded.columns]                       
    
    df_option_most_traded = df_put_option_most_traded.join(df_call_option_most_traded)
    return df_option_most_traded


def prepare_stock_ref_data(df_stock_list):
    df_stock_list['Section/Products'] = df_stock_list['Section/Products'].str.replace('Foreign', 'Domestic').astype("category")
    df_stock_list['NewMarketSegment'] = df_stock_list['NewMarketSegment'].str.replace(' \(Foreign Stock\)', '').fillna('None').astype("category")
    df_stock_list['33SectorCode'] = df_stock_list['33SectorCode'].astype("category")
    df_stock_list['17SectorCode'] = df_stock_list['17SectorCode'].astype("category")
    df_stock_list['NewIndexSeriesSizeCode'] = df_stock_list['NewIndexSeriesSizeCode'].astype("category")
    df_stock_list = df_stock_list[[
        "SecuritiesCode", "Section/Products", "33SectorCode", "17SectorCode", "NewIndexSeriesSizeCode", "MarketCapitalization"
    ]]
    return df_stock_list

def join_primary_and_secondary_price(df_primary_price, df_secondary_price, sector_cols):
    # For secondary stock price, we only take sector level returns as features
    for sector_col in sector_cols:
        df_primary_sector_return = get_sector_return_from_price(df_primary_price, sector_col)
        df_secondary_sector_return = get_sector_return_from_price(df_secondary_price, sector_col)
        df_sector_return = df_primary_sector_return.join(
            df_secondary_sector_return,lsuffix="_primary", rsuffix="_secondary"
        )
        df_primary_price = df_primary_price.merge(
            df_sector_return.reset_index(),
            on=['Date', sector_col],
            how='left'
        )
    return df_primary_price

def add_rolling_feature_to_trade_flow_df(df_trade):
    
    steps = [
     ('Balance', [2, 4, 8], ['mean', 'rank']),
     ('BalanceNormed', [2, 4, 8], ['mean', 'rank']),
     ('Balance_rank', [2, 4, 8], ['mean']),
     ('BalanceNormed', [2, 4, 8], ['mean']),
    ]
    
    for col_name, rolling_options, agg_options in steps:
        for rolling_count, agg_func in product(rolling_options, agg_options):
            df_trade_with_col = df_trade.filter(regex=f'.*{col_name}$')
            col_map = {
                col: f"{col}_Rolling_{rolling_count}_{agg_func}"
                for col in df_trade_with_col.columns
            }
            df_trade[list(col_map.values())] = (
                df_trade_with_col.rolling(rolling_count).agg(agg_func)
            )
    
    return df_trade

def prepare_trade_flow(df_trade):
    df_trade_date_index = df_trade.Date.unique()
    
    df_trade = df_trade.drop(
        df_trade.filter(regex=".*(Sales|Purchases)").columns.tolist(),
        axis=1    
    )
    df_trade = df_trade.drop(['StartDate', 'EndDate'], axis=1)
    
    df_trade = df_trade.groupby('Date').sum(min_count=1).reset_index()
    
    institution_category_names = [
        "InvestmentTrusts", "BusinessCos", "OtherInstitutions", "InsuranceCos", "CityBKsRegionalBKs",
        "TrustBanks", "OtherFinancialInstitutions"
    ]
    flow_col_names = ["Total", "Balance"]
    for flow_col in flow_col_names:
        df_trade[f'Institution{flow_col}'] = (
            df_trade.filter(regex=f"({'|'.join(institution_category_names)}){flow_col}").sum(axis=1, min_count=1)
        )
    df_trade = df_trade.drop(
        df_trade.filter(regex=f"({'|'.join(institution_category_names)}).*").columns.tolist(),
        axis=1
    )
    
    df_trade_normed = (
        df_trade.filter(regex='.*Balance$')
        .div(df_trade.filter(regex='.*Total$').values)
        .rename(columns={
            col: col.replace('Balance', 'BalanceNormed')
            for col in df_trade.filter(regex='.*Balance$').columns
        })
    )
    df_trade[df_trade_normed.columns] = df_trade_normed.values
    df_trade = df_trade.drop(df_trade.filter(regex='.*Total$').columns.tolist(), axis=1)
    
    # Compare horizontally
    prop_broker_group = ["Proprietary", "Brokerage"]
    brokerage_group = ["Individuals", "Foreigners", "SecuritiesCos", "Institution"]
    compare_col_names = ["Balance", "BalanceNormed"]
    
    for group in [prop_broker_group, brokerage_group]:
        for compare_col_name in compare_col_names:
            df_rank_filter = df_trade.filter(regex=f"({'|'.join(group)}){compare_col_name}$")
            df_rank_columns_map = {
                col: f"{col}_rank"
                for col in df_rank_filter.columns[1:]
            }
            df_trade[list(df_rank_columns_map.values())] = (
                df_rank_filter
                .rank(axis=1)
                .iloc[:, 1:] # Degree of freedom decreases by 1
                .rename(columns=df_rank_columns_map)
            )
    
    df_trade = df_trade.dropna()
    df_trade = add_rolling_feature_to_trade_flow_df(df_trade)
    df_trade = (
        df_trade
        .set_index('Date')
        .reindex(df_trade_date_index)
        .fillna(method='ffill')
        .reset_index()
   )
        
    return df_trade

df_primary_price = prepare_price(pd.read_csv('train_files/stock_prices.csv'))
df_secondary_price = prepare_price(pd.read_csv('train_files/secondary_stock_prices.csv'))

df_option = prepare_option(pd.read_csv('train_files/options.csv'))
df_stock_ref_data = prepare_stock_ref_data(pd.read_csv('stock_list.csv'))

df_primary_price = df_primary_price.merge(df_option, on='Date', how='left')
df_primary_price = df_primary_price.merge(df_stock_ref_data, on="SecuritiesCode", how='left')
df_secondary_price = df_secondary_price.merge(df_stock_ref_data, on='SecuritiesCode', how='left')

df_price = join_primary_and_secondary_price(df_primary_price, df_secondary_price, ["33SectorCode", "17SectorCode"])

df_trade = prepare_trade_flow(pd.read_csv('train_files/trades.csv'))
df_price = df_price.merge(df_trade, on='Date', how='left')


