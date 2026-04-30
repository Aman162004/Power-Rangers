import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

import os

import pandas as pd

import os
os.makedirs("plots", exist_ok=True)
df = pd.read_csv("data/historical/raw/demand_load/load_sldc_2021-01-01_to_2026-04-06.csv")
df.rename(columns={"load_mw": "load_MW"}, inplace=True)

print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"\nFirst few rows:")
print(df.head(5))
print(f"\nLast few rows:")
print(df.tail(5))
print(f"\nData Types:")
print(df.dtypes)
print(f"\nBasic Statistics:")
print(df.describe())


df["timestamp"] = pd.to_datetime(df["timestamp"])

# checking for missing, duplicates and negatives
print(f"Missing Values:")
print(df.isna().sum())
print(f"\nMissing Percentage:")
print(df.isna().sum() / len(df) * 100)

duplicates = df.duplicated(subset=["timestamp"]).sum()
print(f"\nDuplicate timestamps: {duplicates}")

negative_values = (df["load_MW"].dropna() <= 0).sum()
print(f"\nNegative or zero values: {negative_values}")


# checking consistent time-step
df_cpy = df.copy()
# df_cpy = df_cpy.dropna() # to avoid warning from calculations
df_cpy = df_cpy.sort_values("timestamp").reset_index(drop=True)
df_cpy.set_index("timestamp", inplace=True)

time_diff = df_cpy.index.to_series().diff()
expected_diff = pd.Timedelta("5 minutes")
irregular_gaps = (time_diff != expected_diff).sum() - 1
print(f"\nIrregular time gaps detected: {irregular_gaps}")


# checking for outliers
Q1 = df_cpy["load_MW"].quantile(0.25)
Q3 = df_cpy["load_MW"].quantile(0.75)
IQR = Q3 - Q1

outliers = (
    (df_cpy.dropna()["load_MW"] < (Q1 - 3 * IQR))
    | (df_cpy.dropna()["load_MW"] > (Q3 + 3 * IQR))
).sum()

print(f"\nOutliers detected (3*IQR method): {outliers}")
print(f"\nOutliers percentage: {outliers / len(df_cpy) * 100:.2f}%")


def categorize_time(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


df_cpy["year"] = df_cpy.index.year
df_cpy["month"] = df_cpy.index.month
df_cpy["day"] = df_cpy.index.day
df_cpy["hour"] = df_cpy.index.hour
df_cpy["minute"] = df_cpy.index.minute
df_cpy["day_of_week"] = df_cpy.index.dayofweek
df_cpy["day_name"] = df_cpy.index.day_name()
df_cpy["week_of_year"] = df_cpy.index.isocalendar().week
df_cpy["quarter"] = df_cpy.index.quarter

df_cpy["is_weekend"] = df_cpy["day_of_week"].isin([5, 6]).astype(int)
# df_cpy['is_missing'] = df_cpy['load_MW'].isna().astype(int)

df_cpy["time_of_day"] = df_cpy["hour"].apply(categorize_time)
df_cpy["season"] = df_cpy["month"].apply(get_season)

print("\nTemporal features created.\n")
print(df_cpy.sample(3))



weather_df = pd.read_csv("data/historical/raw/weather/weather_openmeteo_2021-01-01_to_2026-04-06.csv")
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
weather_df.set_index("timestamp", inplace=True)
df_cpy = df_cpy.join(weather_df[["temperature", "humidity"]], how="left")



print(f"\nLoad Statistics (MW):")
print(f" Mean: {df_cpy['load_MW'].mean():.2f}")
print(f" Median: {df_cpy['load_MW'].median():.2f}")
print(f" Std Dev: {df_cpy['load_MW'].std():.2f}")
print(f" Min: {df_cpy['load_MW'].min():.2f}")
print(f" Max: {df_cpy['load_MW'].max():.2f}")
print(f" Range: {df_cpy['load_MW'].max() - df_cpy['load_MW'].min():.2f}")
print(
    f" CV (Coefficient of Variation): {(df_cpy['load_MW'].std() / df_cpy['load_MW'].mean() * 100):.2f}%"
)

print(f"\nDistribution Characteristics:")
print(f" Skewness: {df_cpy['load_MW'].skew():.3f}")
print(f" Kurtosis: {df_cpy['load_MW'].kurtosis():.3f}")


# complete timeseries
fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(x=df_cpy.index, y=df_cpy["load_MW"], linewidth=0.5, alpha=0.5, ax=ax)
ax.set_title(
    "Delhi Electricity Load - Complete Time Series", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Load (MW)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/01_complete_timeseries.png", dpi=300, bbox_inches="tight")



# monthly average load
monthly_avg = df_cpy.groupby([df_cpy.index.year, df_cpy.index.month])["load_MW"].mean()
monthly_avg.index.names = [None, None]
monthly_avg = monthly_avg.reset_index()
monthly_avg.columns = ["year", "month", "avg_load"]

monthly_avg["date"] = pd.to_datetime(monthly_avg[["year", "month"]].assign(day=1))

fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(x="date", y="avg_load", data=monthly_avg, marker="o", linewidth=2)
ax.set_title("Monthly Average Electricity Load", fontsize=14, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Average Load (MW)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/02_monthly_average.png", dpi=300, bbox_inches="tight")



# distribution of load
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Calculate statistics once
mean_val = df_cpy["load_MW"].mean()
median_val = df_cpy["load_MW"].median()
std_dev_val = df_cpy["load_MW"].std()
min_val = df_cpy["load_MW"].min()
max_val = df_cpy["load_MW"].max()
skewness_val = df_cpy["load_MW"].skew()

# Histogram
sns.histplot(
    x="load_MW",
    data=df_cpy,
    bins=50,
    edgecolor="black",
    alpha=0.7,
    ax=axs[0],
    color="Dodgerblue",
)
axs[0].axvline(
    mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}"
)
axs[0].axvline(
    median_val,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {median_val:.2f}",
)
axs[0].axvline(
    min_val, color="blue", linestyle=":", linewidth=2, label=f"Min: {min_val:.2f}"
)
axs[0].axvline(
    max_val, color="purple", linestyle=":", linewidth=2, label=f"Max: {max_val:.2f}"
)

axs[0].set_title("Load Distribution - Histogram", fontsize=12, fontweight="bold")
axs[0].set_xlabel("Load (MW)", fontsize=11)
axs[0].set_ylabel("Frequency", fontsize=11)

# Text box for statistics on histogram
stats_text = (
    f"Mean: {mean_val:.2f}\n"
    f"Median: {median_val:.2f}\n"
    f"Std Dev: {std_dev_val:.2f}\n"
    f"Min: {min_val:.2f}\n"
    f"Max: {max_val:.2f}\n"
    f"Skewness: {skewness_val:.3f}"
)
axs[0].text(
    0.98,
    0.98,
    stats_text,
    transform=axs[0].transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
)
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Box Plot
sns.boxplot(y="load_MW", data=df_cpy, ax=axs[1], color="Dodgerblue")
axs[1].set_title("Load Distribution - Box Plot", fontsize=12, fontweight="bold")
axs[1].set_ylabel("Load (MW)", fontsize=11)
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./plots/03_load_distribution.png", dpi=300, bbox_inches="tight")



# Histogram of Load Distribution with Statistics

# Calculate statistics
mean_val = df_cpy["load_MW"].mean()
median_val = df_cpy["load_MW"].median()
std_dev_val = df_cpy["load_MW"].std()
min_val = df_cpy["load_MW"].min()
max_val = df_cpy["load_MW"].max()
skewness_val = df_cpy["load_MW"].skew()

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    x="load_MW",
    data=df_cpy,
    bins=50,
    edgecolor="black",
    alpha=0.7,
    ax=ax,
    color="Dodgerblue",
)
ax.axvline(
    mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}"
)
ax.axvline(
    median_val,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {median_val:.2f}",
)
ax.axvline(
    min_val, color="blue", linestyle=":", linewidth=2, label=f"Min: {min_val:.2f}"
)
ax.axvline(
    max_val, color="purple", linestyle=":", linewidth=2, label=f"Max: {max_val:.2f}"
)

ax.set_title("Load Distribution - Histogram", fontsize=14, fontweight="bold")
ax.set_xlabel("Load (MW)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

stats_text = (
    f"Mean: {mean_val:.2f}\n"
    f"Median: {median_val:.2f}\n"
    f"Std Dev: {std_dev_val:.2f}\n"
    f"Min: {min_val:.2f}\n"
    f"Max: {max_val:.2f}\n"
    f"Skewness: {skewness_val:.3f}"
)
ax.text(
    0.98,
    0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./plots/03_load_distribution_histogram.png", dpi=300, bbox_inches="tight")



import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 8))

# Calculate statistics for horizontal lines
median_val = df_cpy["load_MW"].median()
mean_val = df_cpy["load_MW"].mean()
min_val = df_cpy["load_MW"].min()
max_val = df_cpy["load_MW"].max()
Q1_val = df_cpy["load_MW"].quantile(0.25)
Q3_val = df_cpy["load_MW"].quantile(0.75)
# Using plt.boxplot to show mean, median, min, max, and quartiles directly
ax.boxplot(df_cpy["load_MW"].dropna(), showmeans=True, whis=[0, 100])

# Add horizontal lines for key statistics
ax.axhline(
    median_val,
    color="green",
    linestyle="-",
    linewidth=1.5,
    label=f"Median: {median_val:.2f}",
)
ax.axhline(
    mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}"
)
ax.axhline(
    min_val, color="blue", linestyle=":", linewidth=1.5, label=f"Min: {min_val:.2f}"
)
ax.axhline(
    max_val, color="purple", linestyle=":", linewidth=1.5, label=f"Max: {max_val:.2f}"
)
ax.axhline(
    Q1_val, color="orange", linestyle="-.", linewidth=1.5, label=f"Q1: {Q1_val:.2f}"
)
ax.axhline(
    Q3_val, color="brown", linestyle="-.", linewidth=1.5, label=f"Q3: {Q3_val:.2f}"
)

ax.set_title("Load Distribution - Box Plot", fontsize=14, fontweight="bold")
ax.set_ylabel("Load (MW)", fontsize=12)
ax.set_xticks([])  # Hide x-axis ticks for a single box plot
ax.set_ylim(1000, 9000)  # Squeeze the y-axis
ax.set_yticks(range(1000, 9001, 2000))  # Set y-axis spacing to 2000
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")  # Add legend to show what each line represents

plt.tight_layout()
plt.savefig("./plots/03_load_distribution_boxplot.png", dpi=300, bbox_inches="tight")


# Distribution of day_of_week
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x="day_name", data=df_cpy, palette="viridis", ax=ax, order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
ax.set_title(
    "Distribution of Load Records by Day of Week", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Day of Week", fontsize=12)
ax.set_ylabel("Number of Records", fontsize=12)
plt.xticks(rotation=45, ha="right")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/13_day_of_week_distribution.png", dpi=300, bbox_inches="tight")


# Distribution of season
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x="season", data=df_cpy, palette="plasma", ax=ax, order=["Winter", "Summer", "Monsoon", "Post-Monsoon"])
ax.set_title("Distribution of Load Records by Season", fontsize=14, fontweight="bold")
ax.set_xlabel("Season", fontsize=12)
ax.set_ylabel("Number of Records", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/14_season_distribution.png", dpi=300, bbox_inches="tight")


# Distribution of is_weekend
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="is_weekend", data=df_cpy, palette="coolwarm", ax=ax)
ax.set_title(
    "Distribution of Load Records by Weekday/Weekend", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Day Type", fontsize=12)
ax.set_ylabel("Number of Records", fontsize=12)
ax.set_xticklabels(["Weekday", "Weekend"])
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/15_is_weekend_distribution.png", dpi=300, bbox_inches="tight")


# hourly pattern (average load by hour of day)
hourly_pattern = df_cpy.groupby("hour")["load_MW"].agg(["mean", "std", "min", "max"])

fig, ax = plt.subplots(figsize=(15, 6))

sns.lineplot(
    x=hourly_pattern.index,
    y=hourly_pattern["mean"],
    marker="o",
    linewidth=2,
    label="Mean",
    color="steelblue",
    ax=ax,
)
ax.fill_between(
    hourly_pattern.index,
    hourly_pattern["mean"] - hourly_pattern["std"],
    hourly_pattern["mean"] + hourly_pattern["std"],
    alpha=0.3,
    label="+-1 Std Dev",
    color="Coral",
)
sns.lineplot(
    x=hourly_pattern.index,
    y=hourly_pattern["min"],
    linestyle="--",
    alpha=0.5,
    label="Min",
    color="green",
    ax=ax,
)
sns.lineplot(
    x=hourly_pattern.index,
    y=hourly_pattern["max"],
    linestyle="--",
    alpha=0.5,
    label="Max",
    color="red",
    ax=ax,
)

ax.set_title("Daily Load Pattern - Average by Hour", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day", fontsize=12)
ax.set_ylabel("Load (MW)", fontsize=12)
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig("./plots/04_hourly_pattern.png", dpi=300, bbox_inches="tight")



# weekly pattern
weekly_pattern = df_cpy.groupby("day_of_week")["load_MW"].agg(["mean", "std"])
day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    x=np.arange(7),
    height=weekly_pattern["mean"].values,
    yerr=weekly_pattern["std"].values,
    capsize=5,
    alpha=0.7,
    color="dodgerblue",
)
ax.set_title(
    "Weekly Load Pattern - Average by Day of Week", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Day of Week", fontsize=12)
ax.set_ylabel("Average Load (MW)", fontsize=12)
ax.set_xticks(range(7))
ax.set_xticklabels(day_names, rotation=45)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/05_weekly_pattern.png", dpi=300, bbox_inches="tight")



# seasonal pattern
seasonal_pattern = df_cpy.groupby("season")["load_MW"].agg(["mean", "std", "count"])
season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
seasonal_pattern = seasonal_pattern.reindex(season_order)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    x=seasonal_pattern.index,
    height=seasonal_pattern["mean"].values,
    yerr=seasonal_pattern["std"].values,
    capsize=5,
    alpha=0.7,
    color="coral",
)
ax.set_title("Seasonal Load Pattern", fontsize=14, fontweight="bold")
ax.set_xlabel("Season", fontsize=12)
ax.set_ylabel("Average Load (MW)", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/06_seasonal_pattern.png", dpi=300, bbox_inches="tight")


# weekday vs weekend
weekend_comparison = df_cpy.groupby("is_weekend")["load_MW"].agg(["mean", "std"])
labels = ["Weekday", "Weekend"]

# yerr is headache
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(
    x=labels,
    height=weekend_comparison["mean"].values,
    yerr=weekend_comparison["std"].values,
    capsize=5,
    alpha=0.7,
    color=["coral", "dodgerblue"],
)
ax.set_title("Weekday vs Weekend Load Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Average Load (MW)", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/07_weekday_vs_weekend.png", dpi=300, bbox_inches="tight")



# hour vs day-of-week heatmap

pivot_hourly_daily = df_cpy.groupby(["day_of_week", "hour"])["load_MW"].mean().unstack()

fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(
    pivot_hourly_daily,
    cmap="YlOrRd",
    annot=False,
    fmt=".0f",
    cbar_kws={"label": "Average Load (MW)"},
    ax=ax,
)
ax.set_title("Load Heatmap - Hour vs Day of Week", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day", fontsize=12)
ax.set_ylabel("Day of Week", fontsize=12)
ax.set_yticklabels(day_names, rotation=0)
plt.tight_layout()
plt.savefig("./plots/08_heatmap_hour_day.png", dpi=300, bbox_inches="tight")


# monthly heatmap
pivot_monthly = (
    df_cpy.groupby([df_cpy.index.year, df_cpy.index.month])["load_MW"].mean().unstack()
)
month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    pivot_monthly,
    cmap="YlOrRd",
    annot=True,
    fmt=".0f",
    cbar_kws={"label": "Average Load (MW)"},
    ax=ax,
)
ax.set_title("Average Load by Year and Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Year", fontsize=12)
ax.set_xticklabels(month_names)
plt.tight_layout()
plt.savefig("./plots/09_heatmap_year_month.png", dpi=300, bbox_inches="tight")



# load by time of day

time_of_day_pattern = df_cpy.groupby("time_of_day")["load_MW"].agg(["mean", "std"])
time_order = ["Night", "Morning", "Afternoon", "Evening"]
time_of_day_pattern = time_of_day_pattern.reindex(time_order)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    x=time_of_day_pattern.index,
    height=time_of_day_pattern["mean"].values,
    yerr=time_of_day_pattern["std"].values,
    capsize=5,
    alpha=0.7,
    color="Coral",
)
ax.set_title("Load Pattern by Time of Day", fontsize=14, fontweight="bold")
ax.set_xlabel("Time of Day", fontsize=12)
ax.set_ylabel("Average Load (M@)", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("./plots/10_time_of_day_pattern.png", dpi=300, bbox_inches="tight")


# using daily data
daily_data = df_cpy["load_MW"].resample("D").mean()
daily_data = daily_data.fillna(daily_data.mean())

decomposition = seasonal_decompose(daily_data, model="additive", period=7)

fig, axs = plt.subplots(4, 1, figsize=(15, 12))

axs[0].plot(decomposition.observed, linewidth=1)
axs[0].set_title("Original Time Series (Daily Average)", fontsize=12, fontweight="bold")
axs[0].set_ylabel("Load (MW)")
axs[0].grid(True, alpha=0.3)

axs[1].plot(decomposition.trend, linewidth=2, color="orange")
axs[1].set_title("Trend Component", fontsize=12, fontweight="bold")
axs[1].set_ylabel("Load (MW)")
axs[1].grid(True, alpha=0.3)

axs[2].plot(decomposition.seasonal, linewidth=1, color="green")
axs[2].set_title("Seasonal Component (Weekly)", fontsize=12, fontweight="bold")
axs[2].set_ylabel("Load (MW)")
axs[2].grid(True, alpha=0.3)

axs[3].plot(decomposition.resid, linewidth=0.5, color="red")
axs[3].set_title("Residual Component", fontsize=12, fontweight="bold")
axs[3].set_ylabel("Load (MW)")
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./plots/12_decomposition.png", dpi=300, bbox_inches="tight")


peak_hour = hourly_pattern["mean"].idxmax()
min_hour = hourly_pattern["mean"].idxmin()
peak_day = weekly_pattern["mean"].idxmax()
peak_season = seasonal_pattern["mean"].idxmax()

print(f"\nTemporal Patterns:")
print(f"  Peak load hour: {peak_hour}:00 ({hourly_pattern['mean'][peak_hour]:.2f} MW)")
print(f"  Minimum load hour: {min_hour}:00 ({hourly_pattern['mean'][min_hour]:.2f} MW)")
print(
    f"  Peak load day: {day_names[peak_day]} ({weekly_pattern['mean'][peak_day]:.2f} MW)"
)
print(f"  Peak season: {peak_season} ({seasonal_pattern['mean'][peak_season]:.2f} MW)")

weekday_avg = df_cpy[df_cpy["is_weekend"] == 0]["load_MW"].mean()
weekend_avg = df_cpy[df_cpy["is_weekend"] == 1]["load_MW"].mean()
weekend_diff = (weekend_avg - weekday_avg) / weekday_avg * 100

print(f"\nWeekend Effect:")
print(f"  Weekday average: {weekday_avg:.2f} MW")
print(f"  Weekend average: {weekend_avg:.2f} MW")
print(f"  Difference: {weekend_diff:+.2f}%")

# variability analysis
print(f"\nVariability Analysis:")
print(
    f"  Hour with highest variability: {hourly_pattern['std'].idxmax()}:00 (Std: {hourly_pattern['std'].max():.2f} MW)"
)
print(
    f"  Hour with lowest variability: {hourly_pattern['std'].idxmin()}:00 (Std: {hourly_pattern['std'].min():.2f} MW)"
)

if df_cpy["year"].nunique() > 1:
    yearly_avg = df_cpy.groupby("year")["load_MW"].mean()
    print(f"\n\nYear-over-Year Comparison:")
    for year, avg in yearly_avg.items():
        print(f"  {year}: {avg:.2f} MW")

# Box plots of load_MW grouped by season
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(
    x="season", y="load_MW", data=df_cpy, order=season_order, palette="viridis", ax=ax
)
ax.set_title("Load Distribution by Season", fontsize=14, fontweight="bold")
ax.set_xlabel("Season", fontsize=12)
ax.set_ylabel("Load (MW)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/16_load_by_season_boxplot.png", dpi=300, bbox_inches="tight")


# Quantify the difference between summer peaks and winter minimums
summer_max_load = df_cpy[df_cpy["season"] == "Summer"]["load_MW"].max()
winter_min_load = df_cpy[df_cpy["season"] == "Winter"]["load_MW"].min()

print(f"\nSeasonal Extremes:")
print(f"  Summer Peak Load: {summer_max_load:.2f} MW")
print(f"  Winter Minimum Load: {winter_min_load:.2f} MW")
print(
    f"  Difference (Summer Peak - Winter Minimum): {(summer_max_load - winter_min_load):.2f} MW"
)

# Also consider average differences for context
summer_avg_load = seasonal_pattern.loc["Summer", "mean"]
winter_avg_load = seasonal_pattern.loc["Winter", "mean"]
print(f"\nAverage Seasonal Comparison:")
print(f"  Summer Average Load: {summer_avg_load:.2f} MW")
print(f"  Winter Average Load: {winter_avg_load:.2f} MW")
print(
    f"  Difference (Summer Avg - Winter Avg): {(summer_avg_load - winter_avg_load):.2f} MW"
)


# 6.1 Load vs. Temperature Scatter Plot
plt.figure(figsize=(12, 7))
sns.scatterplot(
    x="temperature",
    y="load_MW",
    data=df_cpy,
    alpha=0.3,
    color="dodgerblue",
    edgecolor=None,
)
plt.title("Load (MW) vs. Temperature (°C)", fontsize=14, fontweight="bold")
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Load (MW)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("./plots/17_load_vs_temperature_scatter.png", dpi=300, bbox_inches="tight")



# 5.2 Temperature Distribution Histogram
plt.figure(figsize=(10, 6))
sns.histplot(
    df_cpy["temperature"].dropna(),
    bins=30,
    kde=True,
    color="lightcoral",
    edgecolor="black",
)
plt.title("Distribution of Temperature (°C)", fontsize=14, fontweight="bold")
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("./plots/18_temperature_distribution.png", dpi=300, bbox_inches="tight")



# 5.3 Humidity Distribution Histogram
plt.figure(figsize=(10, 6))
sns.histplot(
    df_cpy["humidity"].dropna(),
    bins=30,
    kde=True,
    color="mediumseagreen",
    edgecolor="black",
)
plt.title("Distribution of Relative Humidity (%)", fontsize=14, fontweight="bold")
plt.xlabel("Relative Humidity (%)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("./plots/19_humidity_distribution.png", dpi=300, bbox_inches="tight")


import matplotlib.pyplot as plt
import seaborn as sns

# Box plot of humidity grouped by season
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(
    x="season", y="humidity", data=df_cpy, order=season_order, palette="mako", ax=ax
)
ax.set_title("Humidity Distribution by Season", fontsize=14, fontweight="bold")
ax.set_xlabel("Season", fontsize=12)
ax.set_ylabel("Relative Humidity (%)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/20_humidity_by_season_boxplot.png", dpi=300, bbox_inches="tight")


import matplotlib.pyplot as plt
import seaborn as sns

# Box plot of temperature grouped by season
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(
    x="season",
    y="temperature",
    data=df_cpy,
    order=season_order,
    palette="viridis",
    ax=ax,
)
ax.set_title("Temperature Distribution by Season", fontsize=14, fontweight="bold")
ax.set_xlabel("Season", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "./plots/21_temperature_by_season_boxplot.png", dpi=300, bbox_inches="tight"
)



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure 'temp_bin' is created if not already present from previous steps
# Using qcut to create bins with roughly equal number of observations
if "temp_bin" not in df_cpy.columns:
    df_cpy["temp_bin"] = pd.qcut(
        df_cpy["temperature"], q=20, duplicates="drop", precision=1
    )

# Calculate average load per temperature bin
binned_avg_load = df_cpy.groupby("temp_bin")["load_MW"].mean().reset_index()
# Convert temp_bin to a numerical representation (midpoint) for plotting
binned_avg_load["temperature_mid"] = binned_avg_load["temp_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(12, 7))
sns.lineplot(
    x="temperature_mid",
    y="load_MW",
    data=binned_avg_load,
    marker="o",
    color="purple",
    linewidth=2,
)
plt.title(
    "Average Electricity Load (MW) by Temperature Bin", fontsize=14, fontweight="bold"
)
plt.xlabel("Temperature (°C) (Midpoint of Bin)", fontsize=12)
plt.ylabel("Average Load (MW)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "./plots/22_binned_load_vs_temperature_line.png", dpi=300, bbox_inches="tight"
)


print("\n--- Insights from Binned Line Plot ---")
print(
    "This plot clearly shows the average electricity demand across different temperature ranges. We can observe how the load generally behaves, confirming the increasing trend at higher temperatures, often resembling a U-shape where load also tends to rise slightly at very low temperatures (though less pronounced in this dataset for Delhi). This visualization effectively smooths out the noise of individual data points to highlight the core relationship."
)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
plt.hexbin(
    df_cpy["temperature"], df_cpy["load_MW"], gridsize=50, cmap="viridis", mincnt=1
)
plt.colorbar(label="Count of Observations")
plt.title(
    "Hexbin Plot: Temperature vs. Electricity Load (Density)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Load (MW)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/23_hexbin_temperature_load.png", dpi=300, bbox_inches="tight")


print("\n--- Insights from Hexbin Plot ---")
print(
    "The hexbin plot addresses overplotting by representing the density of data points with color intensity. Darker hexagons indicate a higher concentration of temperature-load pairs. This plot reinforces the idea that most data points fall within moderate load ranges at moderate temperatures, but also clearly shows denser areas corresponding to higher loads at higher temperatures, particularly visible in the 'tail' stretching towards high load and high temperature, indicating significant demand during hot periods."
)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.regplot(
    x="temperature",
    y="load_MW",
    data=df_cpy,
    scatter=False,
    lowess=True,
    color="dodgerblue",
    line_kws={"lw": 3, "alpha": 0.8},
)
plt.title(
    "Smoothed Regression (LOWESS): Electricity Load vs. Temperature Trend",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Load (MW)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/24_lowess_load_vs_temperature.png", dpi=300, bbox_inches="tight")


print("\n--- Insights from LOWESS Plot ---")
print(
    "This plot focuses solely on the estimated non-linear relationship between temperature and load using Locally Weighted Scatterplot Smoothing (LOWESS). It effectively highlights the 'U-shaped' or, more specifically for Delhi, the strong positive correlation at higher temperatures without the distraction of individual data points. This smooth curve clearly illustrates that load is relatively stable at moderate temperatures and rises sharply as temperatures increase, consistent with increased cooling demands. This also implicitly suggests that very low temperatures, if present in the data, might also lead to increased load due to heating."
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure 'temp_bin' is created if not already present
if "temp_bin" not in df_cpy.columns:
    df_cpy["temp_bin"] = pd.qcut(
        df_cpy["temperature"], q=20, duplicates="drop", precision=1
    )

plt.figure(figsize=(16, 8))
sns.boxplot(x="temp_bin", y="load_MW", data=df_cpy, palette="coolwarm")
plt.title(
    "Electricity Load (MW) Distribution Across Temperature Bins",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Temperature Bins (°C)", fontsize=12)
plt.ylabel("Load (MW)", fontsize=12)
plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(
    "./plots/25_boxplot_load_across_temperature_bins.png", dpi=300, bbox_inches="tight"
)


print("\n--- Insights from Box Plot Across Temperature Bins ---")
print(
    "This box plot provides a statistical summary of load distribution within each temperature bin. It clearly shows how the median load (the central line in each box), the interquartile range (the box itself), and the overall spread of load (whiskers) change with temperature. We can observe not only the consistent increase in median and average load as temperature rises but also how its variability (the height of the box and whiskers) might change, indicating more volatile demand at certain temperature extremes, likely due to varying consumption patterns under hot conditions."
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create temperature bins and calculate mean load per bin
# Using qcut to create bins with roughly equal number of observations
# This helps in handling potential non-uniform distribution of temperatures
df_cpy["temp_bin"] = pd.qcut(df_cpy["temperature"], q=20, duplicates="drop")
aggregated_data = df_cpy.groupby("temp_bin")["load_MW"].mean().reset_index()

# Convert temp_bin to a numerical representation for plotting (e.g., midpoint)
# For simplicity, taking the mean of the interval as the bin's temperature value
aggregated_data["temperature_mid"] = aggregated_data["temp_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(14, 8))

# Plot the aggregated mean loads
sns.scatterplot(
    x="temperature_mid",
    y="load_MW",
    data=aggregated_data,
    s=100,  # Size of points
    color="purple",
    alpha=0.7,
    label="Average Load per Temperature Bin",
)

# Add a regression line to show the trend
sns.regplot(
    x="temperature",
    y="load_MW",
    data=df_cpy,
    scatter=False,  # Do not plot individual points again
    lowess=True,  # Use LOESS regression for non-linear relationships
    color="dodgerblue",
    line_kws={"lw": 3, "alpha": 0.8},
    label="LOESS Trend Line",
)

plt.title(
    "Electricity Load (MW) vs. Temperature (°C) with Aggregation and Trend",
    fontsize=16,
    fontweight="bold",
)
plt.xlabel("Temperature (°C)", fontsize=13)
plt.ylabel("Load (MW)", fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    "./plots/17_load_vs_temperature_enhanced_scatter.png", dpi=300, bbox_inches="tight"
)


# Discussing the insights from the enhanced plot
print("\n--- Insights from Load vs. Temperature Plot ---")
print(
    "The enhanced scatter plot clearly illustrates the non-linear relationship between electricity load and temperature."
)
print(
    "We observe a 'U-shaped' or, more predominantly, an increasing trend at higher temperatures. This suggests:"
)
print("1. At moderate temperatures, load is relatively stable.")
print(
    "2. As temperatures rise significantly (typically above 25-30°C), there's a sharp increase in electricity demand."
)
print(
    "   This strong positive correlation at higher temperatures is primarily driven by increased cooling load from air conditioning."
)
print(
    "3. While less pronounced in this dataset's range for Delhi, in colder climates, low temperatures would also drive up demand for heating."
)
print(
    "This relationship confirms that temperature is a critical factor influencing electricity consumption and peak demand."
)
