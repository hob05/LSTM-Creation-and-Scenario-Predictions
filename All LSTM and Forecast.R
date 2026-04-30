##### Packages and Reload Results #####

library(keras3)
library(tensorflow)
library(readxl)
library(tidyverse)
library(lubridate)
library(MLmetrics)

ae_model <- load_model("malta_ae_lstm.keras")

s1_results <- readRDS("s1_results.rds")
s1_summary <- readRDS("s1_summary.rds")
s1_fit <- readRDS("s1_fit.rds")
s1_linear_gradient <- readRDS("s1_linear_gradient.rds")

s2_results <- readRDS("s2_results.rds")
s2_summary <- readRDS("s2_summary.rds")

s3_results <- readRDS("s3_results.rds")
s3_summary <- readRDS("s3_summary.rds")
s3_fit <- readRDS("s3_fit.rds")
s3_gradient <- readRDS("s3_gradient.rds")

s4_results <- readRDS("s4_results.rds")
s4_summary <- readRDS("s4_summary.rds")

s5_results <- readRDS("s5_results.rds")
s5_summary <- readRDS("s5_summary.rds")

s6_results <- readRDS("s6_results.rds")
s6_summary <- readRDS("s6_summary.rds")
##### Build LSTM Model #####
## call dataframe ##
ae_df <- read_excel("Malta_AE.xlsx") 
ae_df$obs <- as.numeric(ae_df$obs) 

# Compare range and avg #

min(ae_df$MaxT)
max(ae_df$MaxT)
mean(ae_df$MaxT)
sd(ae_df$MaxT)

min(ae_df$Ad)
max(ae_df$Ad)
mean(ae_df$Ad)
sd(ae_df$Ad)

# Remove NA from first and final lags #
ae_df_noNA <- ae_df %>%
  drop_na()

# recode Day as a factor #
ae_df_model <- ae_df_noNA %>%
  arrange(Date) %>%
  mutate(
    Year = year(Date),
    Day = factor(Day, levels = 1:7)
  ) %>%
  filter(!is.na(Ad), !is.na(MaxT), !is.na(Day))

# define train and val data years #
train_ae_df <- ae_df_model %>% filter(Year <= 2023)
val_ae_df   <- ae_df_model %>% filter(Year >= 2024)

## look at distribution ##
min(train_ae_df$MaxT)
max(train_ae_df$MaxT)
mean(train_ae_df$MaxT)
sd(train_ae_df$MaxT)

min(train_ae_df$Ad)
max(train_ae_df$Ad)
mean(train_ae_df$Ad)
sd(train_ae_df$Ad)

#

min(val_ae_df$MaxT)
max(val_ae_df$MaxT)
mean(val_ae_df$MaxT)
sd(val_ae_df$MaxT)

min(val_ae_df$Ad)
max(val_ae_df$Ad)
mean(val_ae_df$Ad)
sd(val_ae_df$Ad)

#

ks.test(train_ae_df$MaxT, val_ae_df$MaxT)
ks.test(train_ae_df$Ad, val_ae_df$Ad)

# Dummy variables for Day #
day_train <- model.matrix(~ Day - 1, data = train_ae_df)
day_val   <- model.matrix(~ Day - 1, data = val_ae_df)

# Scaling #
ad_mean   <- mean(train_ae_df$Ad)
ad_sd     <- sd(train_ae_df$Ad)
if (ad_sd == 0) ad_sd <- 1

maxT_mean <- mean(train_ae_df$MaxT)
maxT_sd   <- sd(train_ae_df$MaxT)
if (maxT_sd == 0) maxT_sd <- 1

train_ad_sc   <- (train_ae_df$Ad   - ad_mean) / ad_sd
train_maxT_sc <- (train_ae_df$MaxT - maxT_mean) / maxT_sd

val_ad_sc   <- (val_ae_df$Ad   - ad_mean) / ad_sd
val_maxT_sc <- (val_ae_df$MaxT - maxT_mean) / maxT_sd

x_train_mat <- cbind(
  Ad = train_ad_sc,
  MaxT = train_maxT_sc,
  day_train
)

x_val_mat <- cbind(
  Ad = val_ad_sc,
  MaxT = val_maxT_sc,
  day_val
)

# Create sequences for time events #

create_sequences <- function(x, y, timesteps = 7) {
  n <- nrow(x)
  n_features <- ncol(x)
  
  X <- array(NA_real_, dim = c(n - timesteps, timesteps, n_features))
  Y <- numeric(n - timesteps)
  
  for (i in 1:(n - timesteps)) {
    X[i, , ] <- as.matrix(x[i:(i + timesteps - 1), ])
    Y[i] <- y[i + timesteps]
  }
  
  list(X = X, Y = Y)
}


seq_train <- create_sequences(
  x = x_train_mat,
  y = train_ad_sc,
  timesteps = 7
)

seq_val <- create_sequences(
  x = x_val_mat,
  y = val_ad_sc,
  timesteps = 7
)

x_train <- seq_train$X
y_train <- seq_train$Y

x_val <- seq_val$X
y_val <- seq_val$Y


dim(x_train)
dim(x_val)

n_features <- dim(x_train)[3]

# Build model  #

ae_model <- keras_model_sequential() %>%
  layer_lstm(units = 16, input_shape = c(7, n_features)) %>%
  layer_dense(units = 1)

ae_model %>% compile(
  optimizer = "adam",
  loss = "mse"
)

# Loss history and model performance #

history <- ae_model %>% fit(
  x_train, y_train,
  epochs = 30,
  batch_size = 16,
  validation_data = list(x_val, y_val),
  verbose = 1
)

history_df <- as.data.frame(history)

ggplot(history_df, aes(x = epoch, y = value, color = data)) +
  geom_point() +
  geom_line(linewidth = 1) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(x = "Epoch", y = "Loss", color = NULL, tag = "B") +
  scale_color_manual(values = c("training" = "#2196F3", "validation" = "#FF5722")) +
  theme_minimal() +
  theme(
    legend.key.size = unit(1.25, "cm"),  
    legend.text = element_text(size = 13), 
    plot.tag = element_text(size = 30, face = "bold")
  ) 

history_df_wide <- history_df %>%
  pivot_wider(
    names_from = "data",
    values_from = "value",
  ) %>%
  select(!`metric`)

min(history_df_wide$training)
min(history_df_wide$validation)

pred_sc <- ae_model %>%
  predict(x_val)
pred <- as.numeric(pred_sc) * ad_sd + ad_mean
actual <- y_val * ad_sd + ad_mean

# Metrics #

mse <- mean((pred - actual)^2)
mse

rmse <- sqrt(mse)
rmse

mean(train_ae_df$Ad)

sd(train_ae_df$Ad)

range(train_ae_df$Ad)

r_sq <- 1 - sum((pred - actual)^2) / sum((actual - mean(actual))^2)
r_sq

mape <- MAPE(pred, actual)
mape

# save model #

save_model(ae_model, "malta_ae_lstm.keras")

# actual vs pred # 

ac_pred_df <- data.frame(
  observation = rep(1:length(actual), 2),
  value       = c(actual, pred),
  type        = rep(c("Actual", "Predicted"), each = length(actual))
)

ggplot(ac_pred_df, aes(x = observation, y = value, colour = type)) +
  geom_line(linewidth = 1) +
  scale_colour_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(y = "Admissions", x = "Observation", colour = NULL, tag = "A") +
  theme_minimal() +
  theme(
    legend.position = "right",
    legend.text  = element_text(size = 12), 
    legend.key.size = unit(1.5, "cm"),
    plot.tag = element_text(size = 30, face = "bold")
  )

##### Naive Forecast #####
##### Naive Forecasting Models (Baseline Comparison) #####

val_actual_full <- val_ae_df$Ad  # full unscaled validation series

# Index of the actual values y_val maps to:
timesteps <- 7
n_val     <- length(y_val)

# Lag-1 # 

naive1_pred <- val_actual_full[(timesteps):(timesteps + n_val - 1)]  

naive1_mse  <- mean((naive1_pred - actual)^2)
naive1_rmse <- sqrt(naive1_mse)
naive1_r2   <- 1 - sum((naive1_pred - actual)^2) / sum((actual - mean(actual))^2)
naive1_mape <- MAPE(naive1_pred, actual)

# Lag-7 #

naive7_pred <- val_actual_full[(timesteps - 6):(timesteps + n_val - 7)]  

naive7_mse  <- mean((naive7_pred - actual)^2)
naive7_rmse <- sqrt(naive7_mse)
naive7_r2   <- 1 - sum((naive7_pred - actual)^2) / sum((actual - mean(actual))^2)
naive7_mape <- MAPE(naive7_pred, actual)

# summary table #
comparison_df <- data.frame(
  Model = c("LSTM", "Naive Lag-1", "Naive Lag-7"),
  RMSE  = round(c(rmse, naive1_rmse, naive7_rmse), 3),
  R2    = round(c(r_sq, naive1_r2,   naive7_r2),   4),
  MAPE  = round(c(mape, naive1_mape, naive7_mape), 4)
)

print(comparison_df)

# pred vs actual #
plot_df <- data.frame(
  observation = 1:n_val,
  Actual      = actual,
  LSTM        = pred,
  Naive_Lag1  = as.numeric(naive1_pred),
  Naive_Lag7  = as.numeric(naive7_pred)
) %>%
  pivot_longer(-observation, names_to = "Model", values_to = "Admissions")

ggplot(plot_df, aes(x = observation, y = Admissions, colour = Model)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c(
    "Actual"     = "black",
    "LSTM"       = "#2196F3",
    "Naive_Lag1" = "#FF5722",
    "Naive_Lag7" = "#4CAF50"
  )) +
  labs(x = "Observation", y = "Admissions", colour = NULL, tag = "C") +
  theme_minimal() +
  theme(
    legend.text     = element_text(size = 12),
    legend.key.size = unit(1.5, "cm"),
    plot.tag        = element_text(size = 30, face = "bold")
  )

###### Scenario 1: Constant Temps #####

# Set forecast days #
forecast_days <- factor(c(5, 6, 7, 1, 2, 3, 4), levels = 1:7) 

# Create prediction function #
forecast_scenarios <- function(scenario_temps, seed_df, forecast_days) {
  
  window_ad   <- (seed_df$Ad   - ad_mean) / ad_sd
  window_maxT <- (seed_df$MaxT - maxT_mean) / maxT_sd
  window_day  <- model.matrix(~ Day - 1, data = seed_df)
  
  preds_scaled <- numeric(7)
  
  for (step in 1:7) {
    x_mat <- cbind(
      Ad   = window_ad,
      MaxT = window_maxT,
      window_day
    )
    
    x_arr <- array(x_mat, dim = c(1, 7, n_features))
    
    p_sc <- as.numeric(ae_model %>% predict(x_arr))
    preds_scaled[step] <- p_sc
    
    new_maxT_sc <- (scenario_temps[step] - maxT_mean) / maxT_sd
    
    new_day_df  <- data.frame(Day = forecast_days[step])
    new_day_row <- model.matrix(~ Day - 1, data = new_day_df)
    
    missing_cols <- setdiff(colnames(window_day), colnames(new_day_row))
    for (mc in missing_cols) new_day_row <- cbind(new_day_row, setNames(data.frame(0), mc))
    new_day_row <- new_day_row[, colnames(window_day), drop = FALSE]
    
    window_ad   <- c(window_ad[-1],   p_sc)
    window_maxT <- c(window_maxT[-1], new_maxT_sc)
    window_day  <- rbind(window_day[-1, ], as.numeric(new_day_row))
  }
  
  # Unscale predictions
  preds <- preds_scaled * ad_sd + ad_mean
  
  data.frame(
    forecast_day = 1:7,
    day_of_week  = as.numeric(as.character(forecast_days)),
    MaxT         = scenario_temps,
    predicted_Ad = round(preds, 1),
    weekly_total = NA_real_
  ) %>%
    mutate(weekly_total = ifelse(forecast_day == 7, sum(predicted_Ad), NA))
}

# Run prediction #

temp_cols <- grep("^Temp_", names(constant_temp), value = TRUE)

set.seed(1)

s1_results <- lapply(temp_cols, function(col) {
  temps <- as.numeric(constant_temp[[col]])
  res <- forecast_scenarios(temps, aug.2025.wk1, forecast_days)
  res$scenario  <- col
  res$temperature <- as.numeric(gsub("Temp_", "", col))
  res
}) %>% bind_rows()

# Summary. results and fit #
s1_summary <- s1_results %>%
  group_by(scenario, temperature) %>%
  summarise(
    weekly_total = sum(predicted_Ad),
    daily_mean = mean(predicted_Ad),
    mean_sd = sd(predicted_Ad),
    .groups = "drop"
  ) %>%
  arrange(temperature)

print(s1_summary)

# 

s1_results %>%
  filter(scenario == "Temp_35") %>%
  select(forecast_day, day_of_week, MaxT, predicted_Ad)

# Fit # 

s1_peak <- s1_summary[which.max(s1_summary$weekly_total), ]
s1_min <- s1_summary[which.min(s1_summary$weekly_total), ]

s1_pre_peak <- s1_summary[1:which.max(s1_summary$weekly_total), ]

s1_fit <- lm(weekly_total ~ temperature, data = s1_summary)
s1_coefs <- coef(s1_fit)

eq_label <- paste0("y = ", round(s1_coefs[2], 2), "x + ", round(s1_coefs[1], 1),
                   "\nR² = ", round(summary(s1_fit)$r.squared, 3))

summary(s1_fit)

s1_linear_gradient <- coef(s1_fit)["temperature"]
cat("Average linear gradient:", round(s1_linear_gradient, 2), "admissions per °C\n")

# Plot Weekly Total Admissions #

ggplot(s1_summary, aes(x = temperature, y = weekly_total)) +
  geom_smooth(method = "lm", formula = y ~ x,
              colour = "grey50", linetype = "dashed",
              se = TRUE, linewidth = 0.8) +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_point(colour = "steelblue", size = 2.5) +
  annotate("text",
           x = 28.2, y = round(s1_peak$weekly_total) - 0.5,  
           label = eq_label,
           hjust = 0, size = 3.5, colour = "grey30") +
  scale_x_continuous(
    breaks = seq(28, 40, by = 2),
    minor_breaks = seq(28, 40, by = 1)
  ) +
  scale_y_continuous(
    limits = c((round(s1_min$weekly_total-6)), round((s1_peak$weekly_total+6))),
    breaks = seq((round(s1_min$weekly_total-6)), round((s1_peak$weekly_total+6)), by = 5),
    minor_breaks = seq((round(s1_min$weekly_total-6)), (round(s1_peak$weekly_total+6)), by = 1)
  ) +
  labs(
    x = "Constant Daily Maximum Temperature (°C)",
    y = "Predicted Weekly Admissions",
    tag = "A"
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_line(colour = "grey70", linewidth = 0.5),
    panel.grid.minor = element_line(colour = "grey90", linewidth = 0.3),
    plot.tag = element_text(size = 30, face = "bold")
  )

# Save results #

saveRDS(s1_results,  "s1_results.rds")
saveRDS(s1_summary, "s1_summary.rds")
saveRDS(s1_fit,   "s1_fit.rds")
saveRDS(s1_linear_gradient, "s1_linear_gradient.rds")

##### Scenario 2: Variation around an increasing mean #####

# Set scenario temperatures #
s2_temps <- lapply(1:nrow(var_mean), function(i) {
  as.numeric(var_mean[i, day_cols])
})

# Run prediction #
set.seed(1)

s2_results <- lapply(seq_along(s2_temps), function(i) {
  res <- forecast_scenarios(s2_temps[[i]], aug.2025.wk1, forecast_days)
  res$scenario       <- paste0("Scenario_", var_mean$Scenario[i]) 
  res$scenario_label <- paste0(round(scenario_means[i]), "°C mean")
  res$date_name      <- factor(res$forecast_day, levels = 1:7,
                               labels = c("Fri 8th", "Sat 9th", "Sun 10th",
                                          "Mon 11th", "Tue 12th", "Wed 13th", "Thu 14th"))
  res
}) %>% bind_rows()

# Results and Summary # 

s2_results <- s2_results %>%
  group_by(scenario) %>%
  mutate(cumulative_Ad = cumsum(predicted_Ad)) %>%
  ungroup()

s2_summary <- s2_results %>%
  group_by(scenario, scenario_label) %>%
  summarise(
    weekly_total = sum(predicted_Ad),
    daily_mean   = mean(predicted_Ad),
    mean_sd = sd(predicted_Ad),
    .groups = "drop"
  ) 

print(s2_summary)

# Plot weekly prediction #
ggplot(s2_summary, aes(x = as.numeric(gsub("°C mean", "", scenario_label)), 
                       y = weekly_total)) +
  geom_point(colour = "steelblue", size = 2) +
  geom_smooth(method = "loess", formula = y ~ x,
              colour = "steelblue", linetype = "solid",
              se = FALSE, linewidth = 0.8) +
  geom_smooth(method = "lm", formula = y ~ x,
              colour = "grey50", linetype = "dashed",
              se = TRUE, linewidth = 0.8) +
  scale_x_continuous(
    breaks = seq(28, 40, by = 2),
    minor_breaks = seq(28, 40, by = 1)
  ) +
  scale_y_continuous(
    breaks = seq(2700, 3000, by = 5),
    minor_breaks = seq(2700, 3000, by = 1)
  ) +
  labs(
    x = "Mean Maximum Daily Temperature (°C)",
    y = "Predicted Weekly Admissions",
    tag = "B"
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_line(colour = "grey70", linewidth = 0.7),
    panel.grid.minor = element_line(colour = "grey90", linewidth = 0.3),
    plot.tag = element_text(size = 30, face = "bold")
  )

# fit #

s2_fit <- lm(weekly_total ~ as.numeric(gsub("°C mean", "", scenario_label)), 
             data = s2_summary)
s2_gradient <- coef(s2_fit)[2]
s2_eq_label <- paste0("y = ", round(coef(s2_fit)[2], 2), "x + ", round(coef(s2_fit)[1], 1),
                      "\nR² = ", round(summary(s2_fit)$r.squared, 3))

s2_peak <- s2_summary[which.max(s2_summary$weekly_total), ]
s2_min <- s2_summary[which.min(s2_summary$weekly_total), ] 

# Save results #


saveRDS(s2_results,   "s2_results.rds")
saveRDS(s2_summary,   "s2_summary.rds")
saveRDS(s2_fit, "s2_fit.rds")
saveRDS(s2_gradient, "s2_gradient.rds")

##### Scenario 3: 1 Extreme Day #####

# set forecast days and temperatures #

day_order <- c("Friday_8th", "Saturday_9th", "Sunday_10th",
               "Monday_11th", "Tuesday_12th", "Wednesday_13th", "Thursday_14th")

s3_temps <- lapply(1:nrow(one_extreme), function(i) {
  as.numeric(one_extreme[i, c("Friday_8th", "Saturday_9th", "Sunday_10th",
                              "Monday_11th", "Tuesday_12th", "Wednesday_13th",
                              "Thursday_14th")])
})

# run prediction #

set.seed(1)

s3_results <- lapply(seq_along(s3_temps), function(i) {
  res <- ext_forecast_scenarios(s3_temps[[i]], aug.2025.wk1, forecast_days)
  res$scenario    <- paste0("Scenario_", i)
  res$extreme_day <- names(one_extreme)[-1][which(one_extreme[i, -1] == 40)]
  res$date_name   <- factor(res$forecast_day, levels = 1:7,
                            labels = c("Fri 8th", "Sat 9th", "Sun 10th",
                                       "Mon 11th", "Tue 12th", "Wed 13th", "Thu 14th"))
  res
}) %>% bind_rows()

# format results with baseline #

s3_results <- s3_results %>% filter(scenario != "Baseline")

s3_results <- bind_rows(
  baseline_res %>% mutate(scenario = "Baseline", extreme_day = "None (31°C Baseline)"),
  s3_results
)

s3_results <- s3_results %>%
  mutate(extreme_day = factor(extreme_day,
                              levels = c("None (31°C Baseline)", day_order),
                              labels = c("None (31°C Baseline)", gsub("_", " ", day_order))))

# summary results #

s3_summary <- s3_results %>%
  group_by(scenario, extreme_day) %>%
  summarise(
    weekly_total = sum(predicted_Ad),
    daily_mean   = round(mean(predicted_Ad), 1),
    .groups = "drop"
  )

print(s3_summary)

# plot single day #

ggplot(s3_results, aes(x = date_name, y = predicted_Ad,
                       colour = extreme_day,
                       group = extreme_day)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_colour_manual(
    values = c("None (31°C Baseline)" = "grey50",
               setNames(RColorBrewer::brewer.pal(7, "Dark2"),
                        gsub("_", " ", day_order))),
    name = "Day of 40°C"
  ) +
  labs(
    x = "Forecast Day",
    y = "Predicted Daily Admissions",
    tag = "A"
  ) +
  scale_y_continuous(
    limits = c(390, 450),
    breaks = seq(390, 450, by = 10),
    minor_breaks = seq(390, 450, by = 5)
  ) +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    legend.title.position = "top"
  )

# Plot weekly total # 

ggplot(s3_summary, aes(x = fct_rev(extreme_day), y = weekly_total)) +
  geom_col(width = 0.5) +
  coord_flip(ylim = c(2800, NA)) +
  geom_text(aes(label = round(weekly_total, 1)), hjust = 1.2, colour = "white") +
  labs(
    x = "Day of 40°C",
    y = "Predicted Weekly Admissions",
    tag = "A"
  ) + 
  theme(
    plot.tag = element_text(size = 30, face = "bold")
  )

# Save results #

saveRDS(s3_results, "s3_results.rds")
saveRDS(s3_summary, "s3_summary.rds")

##### Scenario 4: 2 Extreme Days #####

# set temperatures #

s4_temps <- lapply(1:nrow(two_extreme), function(i) {
  as.numeric(two_extreme[i, c("Friday_8th", "Saturday_9th", "Sunday_10th",
                              "Monday_11th", "Tuesday_12th", "Wednesday_13th",
                              "Thursday_14th")])
})

# run prediction #

set.seed(1)

s4_results <- lapply(seq_along(s4_temps), function(i) {
  res <- ext_forecast_scenarios(s4_temps[[i]], aug.2025.wk1, forecast_days)
  
  res$scenario <- paste0("Scenario_", i)
  res$extreme_day <- {
    hot <- names(two_extreme)[-1][which(two_extreme[i, -1] == 40)]
    if (length(hot) == 0) "None (31°C Baseline)" else paste(hot, collapse = ", ")
  }
  res$date_name <- factor(res$forecast_day, levels = 1:7,     
                          labels = c("Fri 8th", "Sat 9th", "Sun 10th",
                                     "Mon 11th", "Tue 12th", "Wed 13th", "Thu 14th"))
  res
}) %>% bind_rows()

# format results with baseline #

s4_results <- s4_results %>% filter(scenario != "Baseline")

s4_results <- bind_rows(
  baseline_res %>% mutate(scenario = "Baseline", extreme_day = "None (31°C Baseline)"),
  s4_results
)

# summary results #

s4_summary <- s4_results %>%
  mutate(extreme_day = as.character(extreme_day)) %>%
  group_by(scenario, extreme_day) %>%
  summarise(
    weekly_total = sum(predicted_Ad),
    daily_mean   = round(mean(predicted_Ad), 1),
    .groups = "drop"
  ) %>%
  mutate(
    extreme_day = str_replace(extreme_day, ", ", ",\n"),
    extreme_day = factor(extreme_day, levels = paired_levels)
  )

print(s4_summary)

# plot daily predictions # 

paired_levels <- c(
  "Wednesday 13th,\nThursday 14th",
  "Tuesday 12th,\nWednesday 13th",
  "Monday 11th,\nTuesday 12th",
  "Sunday 10th,\nMonday 11th",
  "Saturday 9th,\nSunday 10th",
  "Friday 8th,\nSaturday 9th",
  "None (31°C Baseline)"
)

s4_summary <- s4_summary %>%
  mutate(extreme_day = factor(extreme_day,
                              levels = paired_levels))

ggplot(s4_results, aes(x = date_name, y = predicted_Ad,
                       colour = extreme_day,
                       group = extreme_day)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_colour_manual(
    values = c("None (31°C Baseline)" = "grey50",
               setNames(RColorBrewer::brewer.pal(6, "Dark2"),
                        gsub("_", " ", paired_levels[-1]))),
    name = "Days of 40°C"
  ) +
  labs(
    x = "Forecast Day",
    y = "A&E Admissions",
    tag = "B"
  ) +
  scale_y_continuous(
    limits = c(380, 460),
    breaks = seq(380, 460, by = 10),
    minor_breaks = seq(380, 460, by = 5)
  ) +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    legend.title.position = "top",
    plot.tag = element_text(size = 30, face = "bold")
  )

# plot weekly total #

ggplot(s4_summary, aes(x = extreme_day, y = weekly_total)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(weekly_total, 1)), hjust = 1.2, colour = "white") +
  coord_flip(ylim = c(2800, NA)) +
  labs(
    x = "Days of 40°C",
    y = "Predicted Weekly Admissions",
    tag = "B"
  ) +
  theme(
    plot.tag = element_text(size = 30, face = "bold")
  )

# Save model #

saveRDS(s4_results, "s4_results.rds")
saveRDS(s4_summary, "s4_summary.rds")

##### Scenario 5: 3 Extreme Days #####

# set days and temperatures #

triplet_labels <- c(
  "Baseline (31°C)",
  "Friday 8th, Saturday 9th, Sunday 10th",
  "Saturday 9th, Sunday 10th, Monday 11th",
  "Sunday 10th, Monday 11th, Tuesday 12th",
  "Monday 11th, Tuesday 12th, Wednesday 13th",
  "Tuesday 12th, Wednesday 13th, Thursday 14th"
)

s5_temps <- lapply(1:nrow(three_extreme), function(i) {
  as.numeric(three_extreme[i, c("Friday_8th", "Saturday_9th", "Sunday_10th",
                                "Monday_11th", "Tuesday_12th", "Wednesday_13th",
                                "Thursday_14th")])
})

# run prediction #
set.seed(1)
s5_results <- lapply(seq_along(s5_temps), function(i) {
  res <- ext_forecast_scenarios(s5_temps[[i]], aug.2025.wk1, forecast_days)
  res$scenario <- paste0("Scenario_", i)
  res$extreme_day <- paste(
    gsub("_", " ", names(three_extreme)[-1][which(three_extreme[i, -1] == 40)]),
    collapse = ", "
  )
  res$date_name <- factor(res$forecast_day, levels = 1:7,
                          labels = c("Fri 8th", "Sat 9th", "Sun 10th",
                                     "Mon 11th", "Tue 12th", "Wed 13th", "Thu 14th"))
  res
}) %>% bind_rows()

# format results with baseline and levels  #
s5_results <- bind_rows(
  baseline_res %>% mutate(scenario = "Baseline", extreme_day = "Baseline (31°C)"),
  s5_results
) %>%
  mutate(extreme_day = factor(extreme_day, levels = triplet_labels))

s5_paired_levels <- c(
  "Baseline (31°C)",
  "Friday 8th,\nSaturday 9th,\nSunday 10th",
  "Saturday 9th,\nSunday 10th,\nMonday 11th",
  "Sunday 10th,\nMonday 11th,\nTuesday 12th",
  "Monday 11th,\nTuesday 12th,\nWednesday 13th",
  "Tuesday 12th,\nWednesday 13th,\nThursday 14th"
)

# summary results #
s5_summary <- s5_results %>%
  mutate(extreme_day = as.character(extreme_day)) %>%
  filter(!is.na(extreme_day)) %>%
  group_by(scenario, extreme_day) %>%
  summarise(
    weekly_total = sum(predicted_Ad),
    daily_mean   = round(mean(predicted_Ad), 1),
    .groups = "drop"
  ) %>%
  mutate(
    extreme_day = str_replace_all(extreme_day, ", ", ",\n"),
    extreme_day = factor(extreme_day, levels = rev(s5_paired_levels))
  )

print(s5_summary)

# plot daily predictions #
ggplot(s5_results, aes(x = date_name, y = predicted_Ad,
                       colour = extreme_day,
                       group = extreme_day)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_colour_manual(
    values = c("Baseline (31°C)" = "grey50",
               setNames(RColorBrewer::brewer.pal(5, "Dark2"),
                        triplet_labels[-1])),
    name = "Days of 40°C"
  ) +
  labs(
    x        = "Forecast Day",
    y        = "Predicted Daily Admissions",
    tag = "C"
  ) +
  scale_y_continuous(
    limits = c(390, 460),
    breaks = seq(350, 500, by = 10),
    minor_breaks = seq(350, 500, by = 5)
  ) +
  theme_bw() +
  theme(
    axis.text.x     = element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    legend.title.position = "top",
    plot.tag = element_text(size = 30, face = "bold")
  )

# plot weekly totals # 
ggplot(s5_summary, aes(x = extreme_day, y = weekly_total)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(weekly_total, 1)), hjust = 1.2, colour = "white") +
  coord_flip(ylim = c(2800, NA)) +
  labs(
    x = "Days of 40°C",
    y = "Predicted Weekly Admissions",
    tag = "C"
  ) +
  theme(
    plot.tag = element_text(size = 30, face = "bold")
  )

# Save model #

saveRDS(s5_results, "s5_results.rds")
saveRDS(s5_summary, "s5_summary.rds")

s5_results %>% pull(extreme_day) %>% as.character() %>% unique()

##### ANOVA for Scenarios 3, 4 and 5 #####

# difference between each day #
summary(aov(predicted_Ad ~ extreme_day, data = s3_results))

summary(aov(predicted_Ad ~ extreme_day, data = s4_results))

summary(aov(predicted_Ad ~ extreme_day, data = s5_results))

# combine datasets #
s3_comp <- s3_results %>%
  select(predicted_Ad) %>%
  mutate(n_hw = "1")

s4_comp <- s4_results %>%
  select(predicted_Ad) %>%
  mutate(n_hw = "2")

s5_comp <- s5_results %>%
  select(predicted_Ad) %>%
  mutate(n_hw = "3")

s345_comp <- bind_rows(s3_comp,s4_comp,s5_comp)

# difference between scenarios # 
summary(aov(predicted_Ad ~ n_hw, data = s345_comp))


