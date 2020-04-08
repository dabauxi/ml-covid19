import datetime
from copy import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

confirmed_df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
death_df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
recovered_df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")

latest_data: pd.DataFrame = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/04-05-2020.csv")

columns = confirmed_df.keys()
print(columns)
confirmed_dates = confirmed_df.loc[:, columns[4]:columns[-1]]
death_dates = death_df.loc[:, columns[4]:columns[-1]]
recovered_dates = recovered_df.loc[:, columns[4]:columns[-1]]

np_confirmed = confirmed_df.to_numpy()[:, 4:]
np_death = death_df.to_numpy()[:, 4:]
np_recovered = recovered_df.to_numpy()[:, 4:]
total_np_confirmed = np.sum(np_confirmed, axis=0)
total_np_death = np.sum(np_death, axis=0)
total_np_recovered = np.sum(np_recovered, axis=0)

total_active = total_np_confirmed-total_np_death-total_np_recovered

test = [1, 2, 3, 4, 5]


def daily_increase(data):
    reversed_data = copy(data)
    reversed_data.reverse()
    d_increase = []
    for i in range(len(data)):
        if i < len(data) - 1:
            d_increase.append(reversed_data[i] - reversed_data[i + 1])
    d_increase.append(0)
    d_increase.reverse()
    return d_increase


#print(total_np_confirmed.tolist())
#print(daily_increase(total_np_confirmed.tolist()))

days_since_zero_outbreak = np.array([i for i in range(len(confirmed_dates.keys()))]).reshape(-1, 1)
total_np_confirmed = np.array(total_np_confirmed).reshape(-1, 1)
total_np_recovered = np.array(total_np_recovered).reshape(-1, 1)
total_np_death = np.array(total_np_death).reshape(-1, 1
                                                  )

future_days_to_predict = 10

future_forcast = np.array([i for i in range(len(confirmed_dates.keys())+future_days_to_predict)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


start = '22/1/2020'
start_date = datetime.datetime.strptime(start, '%d/%m/%Y')
future_prediction_days = []
for i in range(len(future_forcast)):
    future_prediction_days.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_zero_outbreak, total_np_confirmed, test_size=0.05, shuffle=False)