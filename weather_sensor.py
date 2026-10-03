import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

df = pd.DataFrame({
    "Hour": range(1, 16),
    "Temperature": [22, 23, np.nan, np.nan, np.nan, 25, 26, 58,
                    24, 23, 22, 21, 23, 24, 25]
})

# (a) stats BEFORE cleaning
print("Stats BEFORE cleaning:")
print(df["Temperature"].describe())
raw = df["Temperature"].dropna()
print("Skewness:", stats.skew(raw))
print("Kurtosis:", stats.kurtosis(raw))
mean_before = df["Temperature"].mean()
std_before = df["Temperature"].std()
print("Mean:", mean_before, "Std:", std_before)

# (b) linear interpolation for the 3 consecutive missing values
df["Temp_raw"] = df["Temperature"]
df["Temp_clean"] = df["Temperature"].interpolate(method="linear")
print("After interpolation, missing values:", df["Temp_clean"].isnull().sum())

# (c) Z-score outlier detection (threshold = 2), replace ONLY flagged value
z = zscore(df["Temp_clean"])
mask = np.abs(z) > 2
median_val = df["Temp_clean"].median()
for _, row in df[mask].iterrows():
    print(f"Z-score(>2) outlier detected at Hour: {int(row['Hour'])} (value: {row['Temp_clean']})")
print("Replaced with column median:", median_val)
df.loc[mask, "Temp_clean"] = median_val

# (d) standardize cleaned column and compare
df["Temp_Zscore"] = zscore(df["Temp_clean"])
print("Before/After comparison:")
print(df[["Hour", "Temp_raw", "Temp_clean", "Temp_Zscore"]])

print("Stats AFTER cleaning:")
print("Mean:", df["Temp_clean"].mean(), "Std:", df["Temp_clean"].std())
print(f"(compare to BEFORE: Mean {mean_before:.2f}, Std {std_before:.2f})")