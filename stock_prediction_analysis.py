"""
Robo Trading for ARIMA model (includes performance evaluation and prediction)

Analysis of Predicting Stock Prices
(Utilizes ARIMA & Time Series Data)
End-of-Semester Collaborative Group Project
[December 2023]
"""

import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import random
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
# from google.colab import drive

TICKER = "BP"
TICKER_NAME = "BP"
TECH = ["AAPL", "MSFT", "AMZN", "IBM", "INTC", "GOOGL", "META", "ORCL", "CSCO",
        "DELL"]
TECH_NAMES = ["Apple", "Microsoft", "Amazon", "IBM", "Intel", "Alphabet",
              "Meta", "Oracle", "Cisco", "Dell"]
AUTO = ["GM", "F", "TM", "BMWYY", "TSLA","HMC", "MBGYY", "VWAPY", "HYMTF",
        "NSANY"]
AUTO_NAMES = ["General Motors", "Ford", "Toyota", "BMW", "Tesla", "Honda",
              "Mercedes Benz", "Volkswagen", "Hyundai", "Nissan"]
HEALTH = ["JNJ", "CVS", "ABT", "PFE", "MRK", "CAH", "HUM", "ABBV", "NVS", "SYK"]
HEALTH_NAMES = ["Johnson & Johnson", "CVS", "Abbot Labs", "Pfizer",
                " Merk & Co.", "Cardinal Health", "Humana", "AbbVie",
                "Novartis", "Stryker"]
OIL = ["XOM", "SHEL", "CVX", "BP", "PSX", "TTE", "COP", "2222.SR", "VLO", "EOG"]
OIL_NAMES = ["Exxon Mobil", "Shell", "Chevron", "BP", "Philips 66",
             "Total Energies", "Conoco Philips", "Saudi Aramco", "Valero",
             "EOG Resources"]
HOLLY = ["DIS", "WBD", "PARA", "UVV", "SONY", "LGF-A", "APO", "NFLX", "AMC",
         "CMCSA"]
HOLLY_NAMES = ["Disney", "Warner Bros", "Paramount", "Universal", "Sony",
               "Lionsgate", "Legendary Entertainment", "Netflix", "AMC",
               "Comcast"]
SELECTED = HEALTH
SELECTED_NAMES = HEALTH_NAMES

PERIOD = "120mo" # data retrieval
START_DATE = '2020-10-30 00:00:00-04:00' # for df
COLUMNS_TO_REMOVE = ['Volume', 'Dividends', 'Stock Splits']
INTEREST_COLUMN = "Open"
# Number of Times to test random buy/sell.
RAN_BUYSELL = 1000


def z_score_normalize(df, column_name):
  '''
  Z-score normalize the columns of a dataframe
  '''
  mean = df[column_name].mean()
  std = df[column_name].std()
  df[column_name + '_normalized'] = (df[column_name] - mean) / std

  return df


def modify_df(df, interest_column = "Close",
              company_name = "Dow Jones Indus. Avg."):
  '''
  returns 3 different types of interest dataframes:
  nan_interest_df: The Dataframe with nans
  interest_df: The Dataframe without nans
  daily_closing_series: The column as a series
  '''
  interest_df = pd.DataFrame(df.loc[:, interest_column])
  daily_closing_series = df[interest_column] # series
  nan_interest_df = interest_df.asfreq('D') # create NaNs for empty cells

  return (nan_interest_df, interest_df, daily_closing_series)


def plot_differences(df, dependent_vars1, dependent_vars2,
                     label1, label2, name):
  '''
  Plot 2 columns of a dataframe (the difference between Yest and Pred Prices)
  '''
  plt.figure(figsize=(10, 6))
  plt.plot(df.index, dependent_vars1, label=label1)
  plt.plot(df.index, dependent_vars2, label=label2)
  plt.title("Difference Between Predicted and Yesterday's Price " + name)
  plt.xlabel('Index')
  plt.ylabel('Difference')
  plt.grid(True)
  plt.legend()
  plt.show()

  return None


def ad_test(company_name, interest_column, dataset):
  '''
  Check for Stationarry Test
  '''
  print("Here we Checking For Stationarity Using the Fixed statistical test")
  print("We are looking for a p < 0.05 to reject the null hypothesis")
  print("ie. Evidence for Data being Stationary: \n")
  print(company_name, interest_column)

  dftest = adfuller(dataset, autolag = 'AIC')
  print("1. ADF : ", dftest[0])
  print("2. P-Value : ", dftest[1])
  print("3. Num Of Lags : ", dftest[2])
  print("4. Num Of Observations Used For ADF Regression:", dftest[3])
  print("5. Critical Values :")
  for key, val in dftest[4].items():
      print("\t", key, ": ", val)

  return None


def choose_pdq(df):
  '''
  Choose the order (pqd) using the smallest AIC
  '''
  # Run this in order to see the best model fit for (p,q,d)
  stepwise_fit = auto_arima(df, trace=True, suppress_warnings=True)
  data_order = stepwise_fit.get_params()['order']

  print("\nGenerally try choose pqd with lower AIC Score.")
  p_val = int(input("Enter the preferred 'p' Value: "))
  d_val = int(input("Enter the preferred 'd' Value: "))
  q_val = int(input("Enter the preferred 'q' Value: "))

  return (p_val, d_val, q_val, data_order)


def money(df):
  '''
  money profited from using our model
  '''
  money_gained = 0

  for i in range(len(df)):
    row = df.iloc[i]

    if not (row).isna().any():
      pred = row['Predct']
      yest = row['Yesterdays Price']

      if pred > yest:
        act = row['Actual']
        money_gained += (act - yest)

  return money_gained


def random_money(df):
  '''
  money gained if you bought/sold at random once
  '''
  money_got = 0

  for i in range(len(df)):
    row = df.iloc[i]

    if not (row).isna().any():
      random_number = random.randint(0, 1)

      if random_number == 1:
        act = row['Actual']
        yest = row['Yesterdays Price']
        money_got += (act - yest)

  return money_got


def random_money_automated(df):
  '''
  money gained if you traded randomly for RAN_BUYSELL itterations
  '''
  ran_buysell = RAN_BUYSELL
  lst = []

  for i in tqdm(range(ran_buysell)):
    ran_money = random_money(df)
    lst.append(ran_money)

  mean = np.mean(lst)
  std_dev = np.std(lst)
  max_value = max(lst)

  print("If you just choose Randomly to buy or sell", ran_buysell, "times")
  print("The Mean is:", mean)
  print("The Standard Deviation is:", std_dev)
  print("The Max is:", max_value)
  if_rand_me_sd_max = [mean, std_dev, max_value]

  return if_rand_me_sd_max


def arima_pred_next(df, i, pdq_order):
  '''
  Evaluate the model for one day
  Given: Dataframe of data
         i for where to split the data
         pdq_order for the order of the model
  Returns: yest (yesterdays price)
           Pred (Tomorrows Predicted price)
           Actual (Today's Price)
           date (Today's date)
  '''
  train = df.iloc[:i]
  test = df.iloc[i]

  model = ARIMA(train, order=pdq_order)
  res = model.fit()

  yest = train.iloc[-1].values[0]
  pred = (res.predict(test.name)).values[0]
  actual = test.values[0]
  date = test.name

  return (yest, pred, actual, date)


def arima_pred_strat(name, test_size, nan_interest_df, interest_df,
                     series_daily_closing):
  '''
  Evaluate the model for a the test_size amount over many days
  '''
  interest_column_local = INTEREST_COLUMN
  actuals = []
  dates = []
  preds = []
  yests = []

  # Get Order___________________________________________________________________
  # Checking For Stationarity Using the Fixed statistical test
  # If p< 0.05 ; Data is stationary
  ad_test(name, interest_column_local, (series_daily_closing).dropna())
  # Get PDQ
  p_val, d_val, q_val, data_order = \
              choose_pdq(interest_df.iloc[:-1 * (test_size)])
  pdq_order = (p_val, d_val, q_val)
  print(pdq_order, "is the order")

  for i in tqdm(range(test_size)):

    i = -1 * (test_size + 1 - i)

    yest, pred, actual, date = arima_pred_next(nan_interest_df, i, pdq_order)

    dates.append(date)
    actuals.append(actual)
    preds.append(pred)
    yests.append(yest)

  return dates, actuals, preds, yests


def clean_data(ticker, name):
  '''
  will clean the data from Yahoo finance and gives 3 different types of cleaned
  dataframes

  nan_interest_df: df including nans for blank values
  interest_df: df with blanks instead of nans
  series_daily_closing: a series (with nans) of the interest collumn
  '''
  start_date = START_DATE
  columns_to_remove = COLUMNS_TO_REMOVE
  period_local = PERIOD
  interest_column_local = INTEREST_COLUMN

  # get all data associated with stock
  given_df = yf.Ticker(ticker)
  # get historical market data
  hist_data = given_df.history(period=period_local)

  # get only the columns we need
  total_df = pd.DataFrame(hist_data)
  df_all_needed = total_df[start_date:]
  df_all_needed = df_all_needed.drop(columns=columns_to_remove)

  # standardize all values in each column
  for column in list(df_all_needed.columns):
    z_score_normalize(df_all_needed, column)

  nan_interest_df, interest_df, series_daily_closing = \
      modify_df(df_all_needed, interest_column = interest_column_local, \
                company_name = name)
  return nan_interest_df, interest_df, series_daily_closing


def all(ticker, name):
  '''
  puts it all together to clean, evaluate, predict, and then plot the results
  for our model
  '''
  # Clean all the data__________________________________________________________
  nan_interest_df, interest_df, series_daily_closing = clean_data(ticker, name)

  # assess model performance of the data's specific ARIMA model_________________
  test_size = int(input("How large would you like your test size?"))
  dates, actuals, preds, yests = arima_pred_strat(name, test_size, nan_interest_df,
                                              interest_df, series_daily_closing)
  act_pred_df = pd.DataFrame(index=dates, data={
      'Actual': actuals,
      'Predct': preds,
      'Yesterdays Price': yests})

  # Plotting Section____________________________________________________________
  difference_1_3 = act_pred_df['Actual'] - act_pred_df['Yesterdays Price']
  difference_2_3 = act_pred_df['Predct'] - act_pred_df['Yesterdays Price']
  plot_differences(act_pred_df, difference_1_3, difference_2_3,
                   "Actual", "Predict", name)

  # How much Money would you make - i.e. your gained return_____________________
  money_gained = money(act_pred_df)
  random_mean_stddev_max = random_money_automated(act_pred_df)

  return(act_pred_df, money_gained, random_mean_stddev_max)


def main():
  '''
  Evaluate the model for an industry (selected) and (selected_names)
  '''
  while True:
    in_yn = \
        str(input("Would you like to evaluate ARIMA over the industry (y / n)"))
    if in_yn == "y" or in_yn == "n":
      break
    else:
      print("please put in y or n")

  if in_yn == "y":
    selected = SELECTED
    selected_names = SELECTED_NAMES
    # Mount your Drive to the Colab VM.
    drive.mount('/gdrive')
    # Suppress all warnings
    warnings.filterwarnings('ignore')

    dct = {}
    dct['Key'] = ["Expected Return from Model", "No Day-Trading",
      "Mean Return From Random Buy Sell",
      "SD of Random Return from Random Buy Sell",
      "Max Return from Random Buy Sell"]

    for i in range(len(selected)):
      ticker = selected[i]
      name = selected_names[i]

      print(name + "______________________________________________________________")
      # Evaluate the model
      act_pred_df, money_gain, if_rand_me_sd_max = all(ticker, name)
      # Get the first non-NaN value
      first_non_nan_index = act_pred_df['Actual'].first_valid_index()
      first_non_nan_value = act_pred_df['Actual'][first_non_nan_index] \
                          if first_non_nan_index is not None else None
      # Get the last non-NaN value
      last_non_nan_index = act_pred_df['Actual'].last_valid_index()
      last_non_nan_value = act_pred_df['Actual'][last_non_nan_index] \
                          if last_non_nan_index is not None else None
      buy_sell_at_end = last_non_nan_value - first_non_nan_value

      dct[name] = [money_gain] + [buy_sell_at_end] + if_rand_me_sd_max

      excel_file = '/gdrive/My Drive/ARIMA_output/' + \
                    'actual_vs_predicted_'+ name + '.csv'
      act_pred_df.to_csv(excel_file, index=True)
      print()
      print()


    # Convert dictionary to DataFrame
    industry_df = pd.DataFrame(dct)
    # Specify file name
    industry_file = '/gdrive/My Drive/ARIMA_output/HEALTH_Industry.csv'
    # Save to Excel
    industry_df.to_csv(industry_file, index=True)

  '''
  Evaluate the model for a specific stock
  '''
  while True:
    st_yn = str(input(f"""Would you like to ARIMA to predict tommorows
stock price for a selected stock? (y / n)"""))
    if st_yn == "y" or st_yn == "n":
      break
    else:
      print("please put in y or n")
  if st_yn == "y":
    ticker = TICKER
    name = TICKER_NAME
    print(name + "____________________________________________________________")

    nan_interest_df,interest_df,series_daily_closing = clean_data(ticker, name)
    # Get Order_________________________________________________________________
    # Checking For Stationarity Using the Fixed statistical test
    # If p< 0.05 ; Data is stationary
    ad_test(name, INTEREST_COLUMN, (series_daily_closing).dropna())
    # Get PDQ
    p_val, d_val, q_val, data_order = choose_pdq(interest_df)
    pdq_order = (p_val, d_val, q_val)
    print(pdq_order, "is the order")

    one_day = pd.Timedelta(days=1)
    train = nan_interest_df
    tod = train.iloc[-1].name
    tom = train.iloc[-1].name + one_day

    model = ARIMA(train, order=pdq_order)
    res = model.fit()

    today_stock = train.iloc[-1].values[0]
    tom_pred_stock = (res.predict(tom)).values[0]

    print("\n\nToday is \n", tod, "\nThe stock price is\n",
          today_stock, "\nand we predict that on\n", tom,
          "\nthe stock price will be\n", tom_pred_stock)

if __name__ == "__main__":
  main()

