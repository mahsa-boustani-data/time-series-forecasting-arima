################################################################################
# Name: Mahsa Boustani
# ID: 153126336
# Column Assigned: V45
################################################################################


# Load required package
install.packages("tseries")
library(tseries)

# Read dataset
Full_Dataset <- read.csv("Case_study.csv")

# Select first 100 observations for training
train_data <- Full_Dataset[1:100, 45]

# Convert to time series object
yt_series <- ts(train_data)



################################################################################
# Step 1: Preliminary analysis of orders
################################################################################

# Plot the time series
plot(yt_series, 
     main = "Time Series Plot of train_data", 
     ylab = "Value (Yt)", 
     xlab = "Time Index")

# Load forecasting package
install.packages("forecast")
library(forecast)

# Compute first and second differences
diff1_yt <- diff(yt_series, differences = 1)
diff2_yt <- diff(yt_series, differences = 2)

# Plot ACF for original series and differenced series
par(mfrow=c(3,1))

Acf(yt_series, main="ACF of Yt (Original Series)")
Acf(diff1_yt, main="ACF of Original Series - First Difference (d=1)")
Acf(diff2_yt, main="ACF of Original Series - Second Difference (d=2)")

par(mfrow=c(1,1))

# Augmented Dickey-Fuller test for stationarity
adf_result <- adf.test(yt_series, alternative = "stationary")
print(adf_result)


# Analyze AR and MA orders using ACF and PACF
par(mfrow=c(1,2))

Acf(diff1_yt, main="ACF for diff1_yt (d=1)", lag.max=20)
Pacf(diff1_yt, main="PACF for diff1_yt (d=1)", lag.max=20)

par(mfrow=c(1,1))



################################################################################
# Step 2: Estimation and selection of ARIMA Models
################################################################################

# Create container to store model results
model_results <- data.frame(Model = character(), AIC = double(), BIC = double()
                            , stringsAsFactors = FALSE)

# Fit ARIMA(p,1,q) models for p,q = 1,...,4
for (p in 1:4) {
  for (q in 1:4) {
    
    # Fit model
    fit <- Arima(yt_series, order = c(p, 1, q))
    
    # Store AIC and BIC
    model_name <- paste0("ARIMA(", p, ",1,", q, ")")
    model_results <- rbind(model_results, data.frame(
      Model = model_name,
      AIC = fit$aic,
      BIC = fit$bic
    ))
  }
}

# Display comparison table
print(model_results)


# Fit selected best models
model_A <- Arima(yt_series, order = c(2, 1, 2)) # Best AIC
model_B <- Arima(yt_series, order = c(2, 1, 1)) # Best BIC
model_C <- Arima(yt_series, order = c(2, 1, 3)) # Adjacent model

# Print parameter estimates
cat("--- Estimates for ARIMA(2,1,2) [Best AIC] ---\n")
print(model_A$coef)
cat("AIC:", model_A$aic, " | BIC:", model_A$bic, "\n\n")

cat("--- Estimates for ARIMA(2,1,1) [Best BIC] ---\n")
print(model_B$coef)
cat("AIC:", model_B$aic, " | BIC:", model_B$bic, "\n\n")

cat("--- Estimates for ARIMA(2,1,3) [Adjacent] ---\n")
print(model_C$coef)
cat("AIC:", model_C$aic, " | BIC:", model_C$bic, "\n\n")



################################################################################
# Step 3: Diagnostic tests with in-sample data
################################################################################

# Store models in a list
models <- list(
  "ARIMA(2,1,2)" = model_A,
  "ARIMA(2,1,1)" = model_B,
  "ARIMA(2,1,3)" = model_C
)

# Residual diagnostics
for (name in names(models)) {
  
  cat("\n==============================================\n")
  cat("Diagnostic Tests for", name, "\n")
  
  res <- residuals(models[[name]])
  
  # Ljung-Box test for autocorrelation
  lb_test <- Box.test(res, lag = 10, type = "Ljung-Box")
  print(lb_test)
  
  # Plot ACF and PACF of residuals
  par(mfrow = c(1,2))
  Acf(res, main = paste("ACF of Residuals -", name))
  Pacf(res, main = paste("PACF of Residuals -", name))
  par(mfrow = c(1,1))
}



#-----------------------------------
# Normality check for residuals
#-----------------------------------

models <- list(
  "ARIMA(2,1,2)" = model_A,
  "ARIMA(2,1,1)" = model_B,
  "ARIMA(2,1,3)" = model_C
)

for (name in names(models)) {
  
  cat("\n==============================================\n")
  cat("Normality Tests for", name, "\n")
  
  res <- residuals(models[[name]])
  
  # Histogram of residuals
  hist(res, 
       main = paste("Histogram of Residuals -", name),
       col = "lightblue",
       breaks = 10)
  
  # QQ plot
  qqnorm(res, main = paste("QQ Plot -", name))
  qqline(res, col = "red")
  
  # Shapiro-Wilk normality test
  shapiro_result <- shapiro.test(res)
  print(shapiro_result)
}



#-----------------------------------
# Plot original series and fitted values
#-----------------------------------

# Final selected model
best_model <- model_B   # ARIMA(2,1,1)

# Extract fitted values
fitted_values <- fitted(best_model)

# Plot original series
plot(yt_series, 
     type = "l", 
     col = "black", 
     lwd = 2,
     main = "Original Series vs Fitted ARIMA(2,1,1)",
     ylab = "Value",
     xlab = "Time")

# Add fitted values
lines(fitted_values, 
      col = "red", 
      lwd = 2)

legend("bottomleft",
       legend = c("Original Series", "Fitted Values"),
       col = c("black", "red"),
       lwd = 2,
       bty = "n")



################################################################################
# Step 4: Forecast with out-of-sample data
################################################################################

# Train/Test split
Full_Dataset <- read.csv("Case_study.csv")
train <- ts(Full_Dataset[1:100, 45])
test  <- ts(Full_Dataset[-(1:100), 45])

# Fit final ARIMA model
best_fit <- Arima(train, order = c(2,1,1))

# Forecast horizons
h_values <- c(10, 25, 100)

par(mfrow = c(3, 1))

# Forecast plots
for (h in h_values) {
  
  h_use <- min(h, length(test))
  
  fc <- forecast(best_fit, h = h_use, level = 95)
  
  # Plot forecast with confidence interval
  plot(fc, main="Forecast ARIMA(2,1,1) with 95% CI",
       xlab="Time", ylab="Value")
  
  # Highlight forecast mean
  lines(fc$mean, col="red", lwd=3)
  
  # Plot actual test data
  lines(ts(test[1:h_use], start=101), col="blue", lwd=2)
  
  legend("bottomleft",
         legend = c("Train", "Forecast mean", "95% CI", "Test (actual)"),
         col = c("black", "red", "gray", "blue"),
         lwd = c(1, 2, 1, 2),
         bty = "n")
}

# Compute Mean Squared Error
mse_results <- data.frame(h = integer(), MSE = double())

for (h in h_values) {
  
  h_use <- min(h, length(test))
  fc <- forecast(best_fit, h = h_use, level = 95)
  
  mse <- mean((as.numeric(test[1:h_use]) - as.numeric(fc$mean))^2)
  
  mse_results <- rbind(mse_results, data.frame(h = h_use, MSE = mse))
}

print(mse_results)

