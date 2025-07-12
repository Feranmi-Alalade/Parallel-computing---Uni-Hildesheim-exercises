library(readxl)
library(lubridate)
library(forecast)

######################################################### Part 1
######### Question 1
path <- paste0(
  "C:/Users/hp/Downloads/First semester/Advanced marketing/",
  "Tasks/task_2_2_data.xlsx"
)

df <- read_excel(path)

head(df)

# Convert the values of the date column to date objects
df$Date <- as.Date(df$Date)

# Creating the time series graph
start_year <- year(min(df$Date))
start_month <- month(min(df$Date))

ts_df <- ts(df$Sales, start = c(start_year, start_month), frequency = 12)

# Save all plots to one pdf
pdf("decomposition_plots.pdf", width = 8, height = 6)

###### Question 2
# First plot showing original time series
plot(ts_df, main = "Time series showing Sales",
     ylab = "Sales", xlab = "Time", col = "blue")

###### Question 3
# second plot for additive decomposition
additive_decompo <- decompose(ts_df, type = "additive")
plot(additive_decompo)
title("Plot showing Additive Decomposition", line = -1)

# third plot for additive decomposition
multiplicative_decompo <- decompose(ts_df, type = "multiplicative")
plot(multiplicative_decompo)
title("Multiplicative Decomposition", line = -1)

dev.off()

############################################################### Part 2
####### Question 1, splitting the data into training and test sets
train_data <- window(ts_df, end = c(2023, 12))
test_data <- window(ts_df, start = c(2024, 1), end = c(2024, 12))

# PDF to save all the forecast plots
pdf("forecast_plots.pdf", width = 8, height = 6)

#### Question 2
# Naive forecast
naive_model <- naive(train_data, h = 12) # predict the next 12 periods
plot(naive_model, main = "Naive Forecast")
lines(test_data, col = "red")
legend("topleft", legend = c("Naive Forecast", "Actual"),
       col = c("darkblue", "red"), lty = 1)

# Average forecast
avg_model <- meanf(train_data, h = 12) # predict the next 12 periods
plot(avg_model, main = "Average Forecast")
lines(test_data, col = "red")
legend("topleft", legend = c("Average Forecast", "Actual"),
       col = c("darkblue", "red"), lty = 1)

# Moving average forecast (n=3)
# The previous 3 observed values
last_3 <- tail(train_data, 3)
# Calculating their mean
ma_value <- mean(last_3)

# Forecast next n periods where n is the length of the test data
ma_3_forecast <- ts(rep(ma_value, length(test_data)),
                    start = start(test_data),
                    frequency = frequency(train_data))

plot(train_data, xlim = c(start(train_data)[1], end(test_data)[1]),
     ylim = range(c(train_data, test_data)),
     main = "Moving Average Forecast (Order 3)",
     ylab = "Sales", col = "black")

# Add forecast and actual test data to plot
lines(ma_3_forecast, col = "blue", lty = 2) # Forecast line
lines(test_data, col = "red") # Actual test data

# Add legend
legend("topleft", legend = c("Training Data", "Test Data", "MA(3) Forecast"),
       col = c("black", "red", "blue"), lty = c(1, 1, 2))

# Moving average forecast (n=6)
# The previous 6 observed values
last_6 <- tail(train_data, 6)
# Calculating their mean
ma_6_value <- mean(last_6)

# Forecast next n periods where n is the length of the test data
ma_6_forecast <- ts(rep(ma_6_value, length(test_data)),
                    start = start(test_data),
                    frequency = frequency(train_data))

plot(train_data, xlim = c(start(train_data)[1], end(test_data)[1]),
     ylim = range(c(train_data, test_data)),
     main = "Moving Average Forecast (Order 6)",
     ylab = "Sales", col = "black")

# Add forecast and actual test data to plot
lines(ma_6_forecast, col = "blue", lty = 2) # Forecast line
lines(test_data, col = "red") # Actual test data

# Add legend
legend("topleft", legend = c("Training Data", "Test Data", "MA(6) Forecast"),
       col = c("black", "red", "blue"), lty = c(1, 1, 2))


# Exponential smoothing, alpha = 0.1
# Exponential smoothing with starting point set as first observation
es_01 <- ses(train_data, alpha = 0.1, initial = "simple", h = 12)
plot(es_01, main = "ES Forecast (alpha=0.1)", col = "darkblue")
lines(test_data, col = "red")
legend("topleft", legend = c("Alpha=0.1", "Actual"),
       col = c("darkblue", "red"), lty = 1)

# Exponential smoothing, alpha = 0.3
# Exponential smoothing with starting point set as first observation
es_03 <- ses(train_data, alpha = 0.3, initial = "simple", h = 12)
plot(es_03, main = "ES Forecast (alpha=0.3)", col = "darkblue")
lines(test_data, col = "red")
legend("topleft", legend = c("Alpha=0.3", "Actual"),
       col = c("darkblue", "red"), lty = 1)

# Exponential smoothing, alpha = 0.9
# Exponential smoothing with starting point set as first observation
es_09 <- ses(train_data, alpha = 0.9, initial = "simple", h = 12)
plot(es_09, main = "ES Forecast (alpha=0.9)", col = "darkblue")
lines(test_data, col = "red")
legend("topleft", legend = c("Alpha=0.9", "Actual"),
       col = c("darkblue", "red"), lty = 1)


########################################## Part 3: Long term forecasting method
### Question 1
### a: estimating trend component using 12-month centered moving avg
trend_comp <- stats::filter(train_data, filter = rep(1/12, 12), sides = 2)
### b: Seasonal component
# seasonal component = observed - trend
seas <- train_data - trend_comp

# months in the training data
months <- cycle(train_data)

### c: Average seasonal coefficients
# averaging the seasonal coefficients for each month across all years
seasonality <- tapply(seas, months, function(x) mean(x, na.rm = TRUE))
# average across months of the year
overall_seasonal_avg <- mean(seasonality, na.rm = TRUE)
# normalized seasonality = seasonality - overall_seasonal_avg
normalized_seasonality <- seasonality - overall_seasonal_avg

# Sum of normalized seasonalities
sum_norm <- sum(normalized_seasonality, na.rm = TRUE)

# Count of non-NA normalized seasonalities
count_norm <- sum(!is.na(normalized_seasonality))

# Print them
cat("Sum of normalized seasonalities:", sum_norm, "\n")
cat("Count of normalized seasonalities:", count_norm, "\n")

### d: linear regression model
# select only the periods with a value for linear trend
non_empty_trend <- trend_comp[!is.na(trend_comp)]

time_non_empty <- which(!is.na(trend_comp))

# Linear regression model
lin_reg <- lm(non_empty_trend ~ time_non_empty)
coef(lin_reg)

# linear regression model equation
intercept <- coef(lin_reg)[1]
slope <- coef(lin_reg)[2]

# print linear regression model equation
cat("Trend Equation: y =", round(intercept, 2),
    "+", round(slope, 2), "* t\n")
cat("Trend Equation: Sales =", round(intercept, 2),
    "+", round(slope, 2), "* t\n")

### QUestion 2
# To predict for 12 months after the train data,
# we need to define the start and stop time periods
t_start <- max(time_non_empty) + 1 #starting from the next month after train
t_end <- t_start + 11
prediction_time <- t_start:t_end

forecast_lin_alg <- intercept + (slope * prediction_time)

# adding normalized seasonal coefficients to the forecasts
final_forecast <- forecast_lin_alg + normalized_seasonality

# converting to time series object
final_forecast_ts <- ts(final_forecast, start = c(2024, 1), frequency = 12)

# Plotting the forecast with actual test data
plot(final_forecast_ts, col = "blue",
     main = "Forecast using Linear regression (Linear Trend + Seasonality)",
     ylab = "Sales", xlab = "Time",
     ylim = range(c(final_forecast_ts, test_data)))
lines(test_data, col = "red")
legend("topleft", legend = c("Forecast", "Actual"),
       col = c("blue", "red"), lty = 1)

dev.off()

### Question 3
pdf("Question_3.3.pdf", width = 8, height = 6)

# Original data with the trend line
plot(train_data, col = "blue", main = "Original data with the trend line",
     ylab = "Sales", xlab = "Time")
lines(trend_comp, col = "red")
legend("topleft", legend = c("Original Data", "Trend Line"),
       col = c("blue", "red"), lty = 1)

# Monthly seasonal coefficients
barplot(seasonality,
        names.arg = month.abb,
        col = "blue",
        main = "Monthly seasonal coefficients (not normalized)",
        ylab = "Seasonal coefficients",
        xlab = "Month")

# Combine train data and forecast
combined_series <- ts(c(train_data, final_forecast_ts),
                      start = start(train_data),
                      frequency = 12)

plot(combined_series, col = "blue", main = "Training + 12-Month Forecast",
     ylab = "Sales", xlab = "Time", lwd = 2)

lines(test_data, col = "red", lwd = 2)
legend("topleft",
       legend = c("Train data + Forecast",
                  "Actual Test Data"),
       col = c("blue", "red"),
       lty = 1, lwd = 2)
dev.off()


######################################## Part 4: Forest Accuracy Analysis
###### Question 1
# Function to calculate mae, mse, rmse, theil_u
model_evaluation <- function(pred_value, actual) {
  mae <- mean(abs(pred_value - actual), na.rm = TRUE)
  mse <- mean((pred_value - actual)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  theil_u <- rmse / sqrt(mean(actual^2, na.rm = TRUE))

  c(MAE = mae, MSE = mse, RMSE = rmse, Theil_U = theil_u)
}

# creating a list for the predicted values for all models
forecasts <- list(
  naive = naive_model$mean,
  average = avg_model$mean,
  ma_3 = ma_3_forecast,
  ma_6 = ma_6_forecast,
  es_01 = es_01$mean,
  es_03 = es_03$mean,
  es_09 = es_09$mean,
  add_seasonal = final_forecast_ts
)

##### Question 2
# applying the evaluation function to the forecasted values
# in forecasts and the actual test values
accuracies <- sapply(forecasts, model_evaluation, actual = test_data)

# Transpose and round to 2dp
accuracies <- t(round(accuracies, 2))
print(accuracies)