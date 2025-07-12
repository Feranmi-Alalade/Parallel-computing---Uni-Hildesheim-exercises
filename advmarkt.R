pdf("Exercise 10_2.pdf", width = 8, height = 6)

# Data vectors
x_t <- c(154, 150, 145, 145, 156, 170, 191, 224, 245, 254, 255, 243,
         232, 213, 200, 206, 215, 232, 247, 276, 300, 308, 311, 300)

l_t_star <- c(167, 172, 177, 182, 187, 192, 198, 203, 208, 213, 218, 223,
              228, 233, 237, 242, 246, 251, 255, 259, 263, 267, 272, 276)

s_t_star <- c(-12, -22, -32, -37, -31, -22, -7, 21, 37, 41, 37, 20,
              4, -20, -37, -36, -31, -19, -8, 17, 37, 41, 39, 24)

# Create monthly time axis from July 2021 to June 2023
dates <- seq(as.Date("2021-07-01"), by = "month", length.out = length(x_t))

# Time index t
t <- seq_along(x_t)

# Linear regression line: l_hat_t = 164.08 + 4.77 * t
l_hat_t <- 164.08 + 4.77 * t

# Extract first letters of each month
month_labels <- substr(format(dates, "%b"), 1, 1)

# Start plot without x-axis
plot(dates, x_t, type = "l", col = "blue", lwd = 2,
     ylim = range(c(x_t, l_t_star, s_t_star)),
     xaxt = "n", xlab = "Month", ylab = "Value",
     main = "Sales vs Trend vs Seasonal Component")

# Add custom x-axis with first letters of months
axis(1, at = dates, labels = month_labels)

abline(h = 0, col = "black", lwd = 1)

# Add other series
lines(dates, l_t_star, col = "red", lwd = 2, lty = 2)
lines(dates, s_t_star, col = "green", lwd = 2, lty = 3)
lines(dates, l_hat_t, col = "purple", lwd = 2, lty = 4)  # Regression line

text(x = dates[18], y = max(x_t) - 10, 
     labels = expression(hat(l)[t] == 164.08 + 4.77 * t),
     col = "purple", cex = 1.1)

# Add legend
legend("topleft", legend = c("Actual Sales (x_t)", "Linear Trend (l*_t)",
                             "Seasonal Coefficients (s*_t)",
                             "Regression Line (l̂_t)"),
       col = c("blue", "red", "green", "purple"), lty = c(1, 2, 3, 4), lwd = 2)

dev.off()