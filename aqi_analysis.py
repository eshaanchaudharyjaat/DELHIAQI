"""
Delhi Air Quality (AQI) Analysis and Visualization
---------------------------------------------------
Mini Project - Python Programming
Student : Eshaan Chaudhary
Course  : BCA (Data Science), 3rd Semester, Chandigarh University

What this program does:
  1. Loads AQI data from a CSV file (or makes sample data if none is found)
  2. Cleans the data (dates, missing values)
  3. Analyses monthly / yearly trends and pollutants
  4. Classifies each day into an AQI category
  5. Saves charts in the 'charts' folder and prints conclusions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = os.path.join("data", "delhi_aqi.csv")
CHART_DIR = "charts"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

sns.set_style("whitegrid")


# ---------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------
def generate_sample_data(path):
    """Creates realistic SAMPLE data (2018-2022) so the project runs
    without a download. Replace it with the real Kaggle dataset."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
    month = dates.month.to_numpy()

    # Delhi pattern: very high in Nov/Dec/Jan, low in Jul/Aug (monsoon)
    seasonal = {1: 280, 2: 200, 3: 150, 4: 140, 5: 150, 6: 140,
                7: 80, 8: 70, 9: 100, 10: 220, 11: 340, 12: 310}
    base = np.array([seasonal[m] for m in month], dtype=float)
    aqi = np.clip(base + rng.normal(0, 40, len(dates)), 30, 500)

    df = pd.DataFrame({
        "Date": dates.strftime("%d-%m-%Y"),
        "PM2.5": aqi * 0.45 + rng.normal(0, 8, len(dates)),
        "PM10": aqi * 0.85 + rng.normal(0, 15, len(dates)),
        "NO2": aqi * 0.12 + rng.normal(0, 5, len(dates)) + 20,
        "SO2": aqi * 0.03 + rng.normal(0, 2, len(dates)) + 8,
        "CO": aqi * 0.006 + rng.normal(0, 0.2, len(dates)) + 0.5,
        "O3": 60 - aqi * 0.05 + rng.normal(0, 8, len(dates)),
        "AQI": aqi,
    })
    num_cols = POLLUTANTS + ["AQI"]
    df[num_cols] = df[num_cols].clip(lower=0)

    # Put some missing values in, like real-world data
    for col in POLLUTANTS + ["AQI"]:
        df.loc[rng.choice(len(df), 40, replace=False), col] = np.nan

    df.round(2).to_csv(path, index=False)
    print(f"[INFO] No dataset found. Sample data created at: {path}")


def load_data(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        generate_sample_data(path)
    df = pd.read_csv(path)
    # Kaggle 'city_day.csv' has many cities - keep only Delhi
    if "City" in df.columns:
        df = df[df["City"] == "Delhi"]
    print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------
# STEP 2: CLEAN DATA
# ---------------------------------------------------------------
def clean_data(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    print("\nMissing values BEFORE cleaning:")
    print(df[cols].isnull().sum().to_string())

    # Fill gaps with the median of the same month (keeps seasonality)
    df["Month"] = df["Date"].dt.month
    for c in cols:
        df[c] = df[c].fillna(df.groupby("Month")[c].transform("median"))
    df = df.dropna(subset=["AQI"])

    df["Year"] = df["Date"].dt.year
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    print("\nMissing values AFTER cleaning:",
          int(df[cols].isnull().sum().sum()))
    return df


# ---------------------------------------------------------------
# STEP 3: AQI CATEGORY (Indian CPCB standard)
# ---------------------------------------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


# ---------------------------------------------------------------
# STEP 4: ANALYSIS + CHARTS
# ---------------------------------------------------------------
def save(fig_name):
    os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, fig_name), dpi=150)
    plt.close()


def plot_yearly_trend(df):
    yearly = df.groupby("Year")["AQI"].mean()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(yearly.index.astype(str), yearly.values, color="#d9534f")
    plt.bar_label(bars, fmt="%.0f")
    plt.title("Average AQI per Year - Delhi")
    plt.xlabel("Year")
    plt.ylabel("Average AQI")
    save("1_yearly_trend.png")
    return yearly


def plot_monthly_trend(df):
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = df.groupby("Month_Name")["AQI"].mean().reindex(order)
    plt.figure(figsize=(9, 5))
    plt.plot(order, monthly.values, marker="o", color="#0275d8", linewidth=2)
    plt.fill_between(order, monthly.values, alpha=0.15, color="#0275d8")
    plt.title("Average AQI by Month - Delhi")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")
    save("2_monthly_trend.png")
    return monthly


def plot_pollutants(df):
    cols = [c for c in POLLUTANTS if c in df.columns]
    means = df[cols].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=means.index, y=means.values, hue=means.index,
                palette="viridis", legend=False)
    plt.title("Average Pollutant Levels - Delhi")
    plt.ylabel("Concentration")
    save("3_pollutants.png")


def plot_heatmap(df):
    pivot = df.pivot_table(values="AQI", index="Year",
                           columns="Month", aggfunc="mean")
    plt.figure(figsize=(10, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                cbar_kws={"label": "AQI"})
    plt.title("AQI Heatmap (Year vs Month)")
    save("4_heatmap.png")


def plot_correlation(df):
    cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation: Pollutants vs AQI")
    save("5_correlation.png")
    return df[cols].corr()["AQI"].drop("AQI").sort_values(ascending=False)


def plot_categories(df):
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    counts = df["AQI_Category"].value_counts().reindex(order, fill_value=0)
    colors = ["#00b050", "#92d050", "#ffff00", "#ffc000", "#ff0000", "#800000"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(order, counts.values, color=colors, edgecolor="black")
    plt.bar_label(bars)
    plt.title("Number of Days in Each AQI Category")
    plt.ylabel("Days")
    plt.xticks(rotation=20)
    save("6_categories.png")
    return counts


# ---------------------------------------------------------------
# STEP 5: MAIN PROGRAM
# ---------------------------------------------------------------
def main():
    print("=" * 55)
    print("   DELHI AIR QUALITY (AQI) ANALYSIS")
    print("=" * 55)

    df = clean_data(load_data(DATA_FILE))
    df["AQI_Category"] = df["AQI"].apply(aqi_category)

    yearly = plot_yearly_trend(df)
    monthly = plot_monthly_trend(df)
    plot_pollutants(df)
    plot_heatmap(df)
    corr = plot_correlation(df)
    counts = plot_categories(df)

    print("\n" + "=" * 55)
    print("   KEY FINDINGS")
    print("=" * 55)
    print(f"Period analysed      : {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")
    print(f"Total days           : {len(df)}")
    print(f"Overall average AQI  : {df['AQI'].mean():.1f} "
          f"({aqi_category(df['AQI'].mean())})")
    print(f"Most polluted month  : {monthly.idxmax()} "
          f"(avg AQI {monthly.max():.0f})")
    print(f"Cleanest month       : {monthly.idxmin()} "
          f"(avg AQI {monthly.min():.0f})")
    print(f"Most polluted year   : {yearly.idxmax()} "
          f"(avg AQI {yearly.max():.0f})")
    worst = df.loc[df["AQI"].idxmax()]
    print(f"Worst single day     : {worst['Date'].date()} "
          f"(AQI {worst['AQI']:.0f})")
    print(f"Pollutant most linked to AQI: {corr.idxmax()} "
          f"(correlation {corr.max():.2f})")
    bad = counts[["Very Poor", "Severe"]].sum()
    print(f"Days 'Very Poor' or 'Severe': {bad} "
          f"({bad / len(df) * 100:.1f}% of days)")

    df.to_csv(os.path.join("data", "cleaned_delhi_aqi.csv"), index=False)
    print(f"\nCharts saved in '{CHART_DIR}' folder.")
    print("Cleaned data saved to data/cleaned_delhi_aqi.csv")


if __name__ == "__main__":
    main()
