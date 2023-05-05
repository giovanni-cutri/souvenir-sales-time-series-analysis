#!/usr/bin/env python
# coding: utf-8

# # Souvenir Sales - Time Series Analysis
# 
# The following is a statistical analysis on a monthly time series which collects data about the sales of a souvenir shop in Australia in the period between 1987 and 1992.
# 
# The analysis will roughly follow the Box-Jenkins method and will focus on reaching stationarity for the time series, estimating a SARIMA model and predicting future values.

# ## Data exploration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")


# In[2]:


tseries = pd.read_csv("data/monthly_sales_queensland.csv", header = 0, parse_dates = ["date"], index_col = 0)


#     Now that we have imported the time series, let's have a first look at its values.

# In[3]:


tseries


# In[4]:


tseries.index.min(), tseries.index.max()


# In[5]:


tseries.index.max() - tseries.index.min()


# ## Check for non-stationarity

# Let's make a plot and see what we can say about it.

# In[6]:


fig, ax = plt.subplots()
ax.plot(tseries["sales"])
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_xticks(ax.get_xticks()[1::1])
plt.xticks(rotation = 90)
plt.show()


# From the plot of the series, we can already grasp that it is not stationary, as the expected value is not constant over time and neither is variance, which tends to increase. In particular, we can speculate the presence of an upward trend and a seasonal effect, which is comprehensible, considering the tourist vocation of the shop.

# In[7]:


df_years = tseries.copy(deep = True)
df_years.reset_index(inplace = True)
df_years["year"] = pd.to_datetime(df_years["date"]).dt.year
df_years["date"] = pd.to_datetime(df_years["date"]).dt.strftime("%m")
unstacked = df_years.set_index(["year", "date"])["sales"].unstack(-2)
unstacked.plot(xlabel = "months", xticks = pd.Series(range(0,12)))


# To better appreciate it, this plot shows the data for each year separately. The values of the series are clearly higher for the latest years and there is a recurring peak in the months of March and August, followed by a valley in October.

# In[8]:


plot_acf(tseries["sales"], lags = 24, zero = False);


# In[9]:


plot_pacf(tseries["sales"], lags = 24, zero = False, method = "ywm");


# Lastly, these are the *global* and *partial autocorrelation functions* for the series. The slow decay for the ACF suggests, once again, the existence of a trend, while the spikes at lag 12 indicate a probable seasonality.

# To formalize our guesses, let's resort to two statistical test:
# - The **Augmented Dickey-Fuller test** tests the null hypothesis of the presence of a unit root in our time series
# - The **KPSS test** tests the null hypothesis that our data is stationary

# In[10]:


adfuller(tseries["sales"])


# In[11]:


kpss(tseries["sales"])


# We were expecting to reject H0 for **KPSS** and not be able to reject it for **ADF** and that's exactly what happened, looking at the p-values.

# ## Reach stationarity

# In order to obtain stationarity in our time series, we need to perform a series of operations: we are going to stabilize the variance through the *Box-Cox transformation* and then apply differencing to treat trend and seasonality.

# In[12]:


tseries_novar = tseries.copy(deep = True)
bc = boxcox(tseries["sales"])[0]
lmbda = boxcox(tseries["sales"])[1]
tseries_novar["sales"] = bc


# In[13]:


fig, ax = plt.subplots()
ax.plot(tseries_novar["sales"])
ax.set_title("Time series with stabilized variance")
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_xticks(ax.get_xticks()[1::1])
plt.xticks(rotation = 90)
plt.show()


# In[14]:


tseries_diff = tseries_novar.copy(deep = True)
tseries_diff["sales"] = tseries_diff["sales"].diff(periods = 1)
tseries_diff = tseries_diff.iloc[1:]
tseries_diff


# In[15]:


fig, ax = plt.subplots()
ax.plot(tseries_diff["sales"])
ax.set_title("Time series with first order differencing")
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_xticks(ax.get_xticks()[1::1])
plt.xticks(rotation = 90)
plt.show()


# The first order differencing removes trend, but leaves seasonality.

# In[16]:


tseries_diff_s = tseries_novar.copy(deep = True)
tseries_diff_s["sales"] = tseries_diff_s["sales"].diff(periods = 12)
tseries_diff_s = tseries_diff_s.iloc[12:]
tseries_diff_s


# In[17]:


fig, ax = plt.subplots()
ax.plot(tseries_diff_s["sales"])
ax.set_title("Time series with first order seasonal differencing")
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_xticks(ax.get_xticks()[1::1])
plt.xticks(rotation = 90)
plt.show()


# The seasonal differencing removes seasonality, but leaves trend.

# In[18]:


tseries_diff_final = tseries_novar.copy(deep = True)
tseries_diff_final["sales"] = tseries_diff_final["sales"].diff(periods = 1).diff(periods = 12)
tseries_diff_final = tseries_diff_final.iloc[13:]
tseries_diff_final


# In[19]:


fig, ax = plt.subplots()
ax.plot(tseries_diff_final["sales"])
ax.set_title("Time series with first order differencing and seasonal differencing")
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_xticks(ax.get_xticks()[1::1])
plt.xticks(rotation = 90)
plt.show()


# The new series should be stationary now. Let's check with our two tests.

# In[20]:


adfuller(tseries_diff_final["sales"])


# In[21]:


kpss(tseries_diff_final["sales"])


# Our conclusions on the null hypothesis have now switched, as we expected.

# Let's look at the ACF and PACF.

# In[22]:


plot_acf(tseries_diff_final["sales"], lags = 24, zero = False);


# In[23]:


plot_pacf(tseries_diff_final["sales"], lags = 24, zero = False, method = "ywm");


# ## Estimate ARIMA model

# At this point, we should be able to estimate the values for the parameters of our ARIMA model by looking at these two plots and the spikes in them. However, the most reliable way to actually determine the parameters is using an objective procedure, for example a stepwise-like, and let a computer do it for us by choosing among many ARIMA models the "best" one, in terms of optimizing a certain indicator.

# In[24]:


fit = pm.auto_arima(tseries_novar, start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12,
                             start_P = 0, seasonal = True, d = 1, D = 1, trace = True,
                             error_action = "ignore",
                             suppress_warnings = True,
                             stepwise = True)
fit.summary()


# The chosen model appears to be *ARIMA(1,1,0)(0,1,1)[12]*. Along with it, we have also got a number of diagnostic tools: we can see that the parameters are significantly different than 0, while none of the residuals is significant.

# Let's look at other diagnostic measures through some plots.

# In[25]:


fit.plot_diagnostics();


# The residuals roughly follow a normal distribution, as deduced from the histogram and the Q-Q plot, but they seem to follow a pattern in their time series, which is not good for our model. 

# ## Forecast

# For the last step, let's try to forecast some future values, in particular 12 more observations, and plot the result.

# In[26]:


forecast = fit.predict(n_periods = 12)
df_forecast = pd.DataFrame({"sales": forecast.values})
df_forecast["sales"] = inv_boxcox(df_forecast["sales"], lmbda)
df_forecast.set_index(forecast.index, inplace = True)
df_forecast.index.name = "date"
tseries_forecast = pd.concat([tseries, df_forecast])
tseries_forecast


# In[27]:


fig, ax = plt.subplots()
ax.plot(tseries_forecast["sales"])
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_xticks(ax.get_xticks()[1::1])
plt.xticks(rotation = 90)
plt.show()

