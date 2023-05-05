library(tidyverse)
library(tsdl)
library(tseries)
library(tsibble)
library(feasts)
library(forecast)
library(lmtest)
library(ldsr)

data <- subset(tsdl, 12, "sales", description = "Queensland")[[1]]
attributes(data)

tseries <- data %>% as_tsibble() %>% head(n = 72) %>%
  rename(date = index, sales = value)
tseries

tseries %>% ggplot(aes(x = date, y = sales)) +
  geom_line() +
  theme_minimal()

tseries %>% gg_season(sales) +
  theme_minimal()

tseries %>% ACF(sales) %>% autoplot()
tseries %>% PACF(sales) %>% autoplot()

tseries %>% as.ts() %>% adf.test()
tseries %>% as.ts() %>% kpss.test()

lambda <- BoxCox.lambda(tseries$sales)
bc <- BoxCox(tseries$sales, lambda = lambda)
tseries.novar <- tseries %>% mutate(sales = bc)

tseries.novar %>% ggplot(aes(x = date, y = sales)) +
  geom_line() +
  theme_minimal()
 
tseries.diff <- tseries.novar %>%
  mutate(date = date, sales = difference(sales, lag = 1, differences = 1)) %>%
  slice_tail(n = -1)
tseries.diff

tseries.diff %>%
  ggplot(aes(x = date, y = sales)) +
  geom_line() +
  theme_minimal()

tseries.diff.s <- tseries.novar %>%
  mutate(sales = difference(sales, lag = 12, differences = 1)) %>%
  slice_tail(n= -12)
tseries.diff.s

tseries.diff.s %>%
  ggplot(aes(x = date, y = sales)) +
  geom_line() +
  theme_minimal()

tseries.diff.final <- tseries.novar %>%
  mutate(sales = difference(sales, lag = 1, differences = 1)) %>%
  slice_tail(n = -1) %>%
  mutate(sales = difference(sales, lag = 12, differences = 1)) %>%
  slice_tail (n = -12)
tseries.diff.final

tseries.diff.final %>%
  ggplot(aes(x = date, y = sales)) +
  geom_line() +
  theme_minimal()

tseries.diff.final %>% as.ts() %>% adf.test()
tseries.diff.final %>% as.ts() %>% kpss.test()

tseries.diff.final %>% ACF(sales, lag_max = 24) %>% autoplot()
tseries.diff.final %>% PACF(sales, lag_max = 24) %>% autoplot()

fit <- tseries %>%
  as.ts() %>%
  auto.arima(start.p = 1,start.q = 1, max.p = 3, max.q = 3,
             start.P = 0, seasonal = T, d = 1, D = 1, trace = T,
             stepwise = T, lambda = "auto")

coeftest(fit)

checkresiduals(fit)
qqnorm(fit$residuals)

tseries.forecast <- forecast(fit, h = 12)
tseries.forecast %>% autoplot()
