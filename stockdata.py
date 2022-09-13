# Activate enviornment
# conda activate myenv (stock_analysis for this project)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates
import datetime as dt
from datetime import date
import time

import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

import os
from os import listdir
from os.path import isfile, join
import threading

# Global Constants
PERIOD = "1y"
PATH = f"stocks/{PERIOD}"
TICKERFILEPATH = "Wilshire-5000-Stocks.csv"


# creates csv from web data and returns the dataframe
def create_csv(ticker, path):
    # start and end at in datetime format (pd.to_datetime(yyyy-mm-dd))
    #df = web.DataReader(ticker, 'yahoo', start, end)
    try:
        stock = yf.Ticker(ticker)
    except Exception as ex:
        print("Couldn't find " + ticker)
        return  
         
    df = stock.history(period=PERIOD)
    if df.empty:
        print("Failed to get data for ", ticker)
        return

    # add daily returns
    df['Daily Return'] = (df['Close'] / df['Close'].shift(1)) - 1
    # add cumulative returns
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    # add Ichimoku data
    df = add_ichimoku_data(df)

    df.to_csv(path + ticker + ".csv")

    outfile = path + ticker.replace(".", "_") + ".csv"
    df.to_csv(outfile)
    print(outfile, " saved")


def get_column_from_csv(file, col_name):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File doesn't exist")
    else:
        return df[col_name]


# gets pandas dataframe from csv file
def get_df(ticker):
    try: 
        df = pd.read_csv("stocks/" + ticker + ".csv")
    except FileNotFoundError:
        print("File for " + ticker + " doesn't exist")
    else:
        return df


def get_roi(df, start_date, end_date):
    # datetime format (pd.to_datetime(yyyy-mm-dd))
    df['Date'] = pd.to_datetime(df['Date'])
    init_value = df[df['Date'] == start_date]['Adj Close'].values[0]
    end_value = df[df['Date'] == end_date]['Adj Close'].values[0]
    print("Initial Price: ", init_value)
    print("Final Price: ", end_value)
    roi = (end_value - init_value) / init_value
    return roi


def add_bollinger_bands(df):
    df['middle_band'] = df['Close'].rolling(window=20).mean() # moving average over 20 days since its daily data
    df['upper_band'] = df['middle_band'] + 1.96 * df['Close'].rolling(window=20).std()
    df['lower_band'] = df['middle_band'] - 1.96 * df['Close'].rolling(window=20).std()
    return df


def add_ichimoku_data(df):
    if df.empty:
        print("Error reading dataframe")
        return
    # Conversion Line = (Highest Value in period + Lowest value in period)/2 (9 Sessions)
    high_value = df['High'].rolling(window=9).max() 
    low_value = df['Low'].rolling(window=9).min()
    df['Conversion'] = (high_value + low_value) / 2

    # Base Line = (Highest Value in period + Lowest value in period)/2 (26 Sessions)
    high_value2 = df['High'].rolling(window=26).max() 
    low_value2 = df['Low'].rolling(window=26).min()
    df['Baseline'] = (high_value2 + low_value2) / 2

    # Leading Span A = (Conversion Value + Base Value)/2 (26 Sessions)
    df['Leading_Span_A'] = ((df['Baseline'] + df['Conversion']) / 2).shift(26)
    
    # Leading Span B = (Conversion Value + Base Value)/2 (52 Sessions)
    high_value3= df['High'].rolling(window=52).max() 
    low_value3 = df['Low'].rolling(window=52).min()
    df['Leading_Span_B'] = ((high_value3 + low_value3) / 2).shift(26)

    # Lagging Span = Price shifted back 26 periods
    df['Lagging_Span'] = df['Close'].shift(-26)
    return df


def plot_with_bollinger_bands(df, ticker):
    fig = go.Figure()
    candle = go.Candlestick(x=df.index, open=df['Open'], high=['High'], 
    low=df['Low'], close=df['Close'], name='Candlestick')     # x = date

    upperLine = go.Scatter(x=df.index, y=df['upper_band'], 
    line=dict(color='rgba(250, 0, 0, 0.75)', width=1), name='Upper Band')

    midLine = go.Scatter(x=df.index, y=df['middle_band'], 
    line=dict(color='rgba(0, 0, 250, 0.75)', width=0.7), name='Middle Band')

    lowerLine = go.Scatter(x=df.index, y=df['lower_band'], 
    line=dict(color='rgba(0, 250, 0, 0.75)', width=1), name='Lower Band')

    fig.add_trace(candle)
    fig.add_trace(upperLine)
    fig.add_trace(midLine)
    fig.add_trace(lowerLine)

    fig.update_xaxes(title="Date", rangeslider_visible=True)  # rangeslider to zoom in on data on plot
    fig.update_yaxes(title="Price")

    fig.update_layout(title=ticker + " Bollinger Bands", height=1200, width=1800, showlegend=True)
    # plot(fig, auto_open=True) # plots in browser
    fig.show()


def plot_Ichimoku(df, ticker):  
    if df.empty:
        print("Error reading dataframe")
        return
    # draw fill shape between span A and span B
    candle = go.Candlestick(x=df.index, open=df['Open'], high=['High'], 
    low=df['Low'], close=df['Close'], name='Candlestick')

    close = go.Scatter(x=df.index, y=df['Close'], line=dict(color='orange', width=1), name='Closing Price')

    df1 = df.copy()
    fig = go.Figure()
    df['label'] = np.where(df['Leading_Span_A'] > df['Leading_Span_B'], 1, 0) # depends if span a is above span b
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()  # accumulating values of labels that are different
    df = df.groupby('group')

    dfs = []

    for name, data in df:
        dfs.append(data)
    
    for df in dfs:
        label = df['label'].iloc[0]
        if label >= 1:
            color = 'rgba(0, 250, 0, 0.4)'
        else:
            color = 'rgba(250, 0, 0, 0.4)'

        fig.add_traces(go.Scatter(x=df.index, y=df.Leading_Span_A, line=dict(color='rgba(0,0,0,0)')))
        fig.add_traces(go.Scatter(x=df.index, y=df.Leading_Span_B, line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty', fillcolor=color))

    baseline = go.Scatter(x=df1.index, y=df1['Baseline'], line=dict(color='pink', width=2), name='Baseline')
    conversion = go.Scatter(x=df1.index, y=df1['Conversion'], line=dict(color='black', width=1), name='Conversion')
    lagging = go.Scatter(x=df1.index, y=df1['Lagging_Span'], line=dict(color='purple', width=1), name='Lagging Span')
    span_a = go.Scatter(x=df1.index, y=df1['Leading_Span_A'], line=dict(color='green', width=2, dash='dot'), name='Span A')
    span_b = go.Scatter(x=df1.index, y=df1['Leading_Span_B'], line=dict(color='red', width=1, dash='dot'), name='Span B')

    fig.add_trace(close)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)
    fig.update_xaxes(title="Date", rangeslider_visible=True)  # rangeslider to zoom in on data on plot
    fig.update_layout(title=ticker + " Ichimoku Plot", height=1200, width=1800, showlegend=True)
    #fig.show()
    plot(fig)


def download_stocks_group(tickers, start, end):
    for x in range(start, end):
        create_csv(tickers[x], PATH)  

    print(f"Finished downloading {start} to {end}") 


def download_all_stocks():
    tickerlist = get_column_from_csv(TICKERFILEPATH, "Ticker")
    i = len(tickerlist)//12
    tlist = []
    for n in range(0, 12):
        th = threading.Thread(target=download_stocks_group, args=(tickerlist, i*n, i*(n+1)))
        tlist.append(th)
    for t in tlist:
        t.start()
    for t in tlist:
        t.join()


def Run():
    start = time.time()
    download_all_stocks()
    end = time.time()
    print("Finished downloading all stocks, %4.2f s elapsed" % (end-start))

# Uncomment to download data
# Run()
