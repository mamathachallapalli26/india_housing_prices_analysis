"""
EDA for the Indian housing dataset used by the Streamlit app.

This script can be saved and run locally to explore the property market data.
It reads the cleaned dataset from the same folder as this file.

Usage:
    python eda_housing.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px


DATA_FILE = "cleaned_housing_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "eda_outputs"


def load_data():
    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}.\n"
            "Please place '{DATA_FILE}' in the same folder as this EDA script."
        )

    df = pd.read_csv(data_path)
    df = df.copy()
    df["Good_Investment_%"] = df["Good_Investment"] * 100
    return df


def show_basic_info(df):
    print("=" * 80)
    print("Dataset Overview")
    print("=" * 80)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nMissing values:\n", df.isna().sum())


def show_numeric_summary(df):
    print("\n" + "=" * 80)
    print("Numeric Summary")
    print("=" * 80)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    print(df[numeric_cols].describe().T.round(2))


def show_categorical_summary(df):
    print("\n" + "=" * 80)
    print("Categorical Summary")
    print("=" * 80)
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    for col in cat_cols:
        print(f"\n{col}:\n{df[col].value_counts().head(10)}")


def plot_missing_values(df):
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing = missing[missing["missing_count"] > 0]

    if missing.empty:
        print("No missing values to visualize.")
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(data=missing, x="missing_count", y="column", palette="viridis")
    plt.title("Missing Values by Column")
    plt.xlabel("Missing Count")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=200)
    plt.close()


def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Price_in_Lakhs"], kde=True, ax=axes[0], color="purple")
    axes[0].set_title("Distribution of Price (Lakhs)")
    axes[0].set_xlabel("Price in Lakhs")

    sns.histplot(df["Price_per_SqFt_INR"], kde=True, ax=axes[1], color="indigo")
    axes[1].set_title("Distribution of Price per Sq Ft")
    axes[1].set_xlabel("Price per Sq Ft (INR)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=200)
    plt.close()


def plot_quality_by_city(df):
    city_quality = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=city_quality.values * 100, y=city_quality.index, palette="magma")
    plt.title("Top Cities by Good Investment Share")
    plt.xlabel("Good Investment (%)")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "good_investment_by_city.png", dpi=200)
    plt.close()


def plot_property_type_and_state(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    type_counts = df["Property_Type"].value_counts()
    sns.barplot(x=type_counts.values, y=type_counts.index, palette="viridis", ax=axes[0])
    axes[0].set_title("Property Type Frequency")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Property Type")

    state_prices = df.groupby("State")["Price_per_SqFt_INR"].mean().sort_values(ascending=False)
    sns.barplot(x=state_prices.values, y=state_prices.index, palette="plasma", ax=axes[1])
    axes[1].set_title("Average Price per Sq Ft by State")
    axes[1].set_xlabel("Price per Sq Ft (INR)")
    axes[1].set_ylabel("State")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "property_and_state_summary.png", dpi=200)
    plt.close()


def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr = df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def plot_size_vs_price(df):
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        hue="Good_Investment",
        palette={0: "#6b21a8", 1: "#f472b6"},
        alpha=0.75,
    )
    plt.title("Size vs Price (Colored by Investment Quality)")
    plt.xlabel("Size in Sq Ft")
    plt.ylabel("Price in Lakhs")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "size_vs_price.png", dpi=200)
    plt.close()


def plot_transport_and_investment(df):
    transport_summary = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=transport_summary, x="Public_Transport_Accessibility", y="Good_Investment", palette="magma")
    plt.title("Good Investment Rate by Transport Accessibility")
    plt.ylabel("Good Investment Rate")
    plt.xlabel("Transport Accessibility")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transport_vs_investment.png", dpi=200)
    plt.close()


def plot_interactive_charts(df):
    fig = px.scatter(
        df,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        color="City",
        hover_data=["Property_Type", "BHK"],
        title="Interactive: Size vs Price by City",
    )
    fig.write_html(str(OUTPUT_DIR / "size_vs_price_interactive.html"))

    fig2 = px.box(
        df,
        x="City",
        y="Price_per_SqFt_INR",
        title="Interactive: Price per Sq Ft by City",
    )
    fig2.write_html(str(OUTPUT_DIR / "price_per_sqft_by_city.html"))


def print_insights(df):
    print("\n" + "=" * 80)
    print("Key EDA Insights")
    print("=" * 80)

    pps = df["Price_per_SqFt_INR"]
    print(f"Average Price per Sq Ft: ₹{pps.mean():,.0f}")
    print(f"Median Price per Sq Ft: ₹{pps.median():,.0f}")
    print(f"Min Price per Sq Ft: ₹{pps.min():,.0f}")
    print(f"Max Price per Sq Ft: ₹{pps.max():,.0f}")

    print(f"\nAverage Property Price: ₹{df['Price_in_Lakhs'].mean():,.1f} Lakhs")
    print(f"Average Size: {df['Size_in_SqFt'].mean():,.0f} sq ft")
    print(f"Good Investment Share: {df['Good_Investment'].mean():.2%}")

    corr_size_price = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    corr_size_ppsf = df["Size_in_SqFt"].corr(df["Price_per_SqFt_INR"])
    print(f"Correlation(Size, Price): {corr_size_price:.3f}")
    print(f"Correlation(Size, Price per Sq Ft): {corr_size_ppsf:.3f}")

    city = df.groupby("City")["Good_Investment"].mean().idxmax()
    city_value = df.groupby("City")["Good_Investment"].mean().max()
    print(f"Best City for Investment (by Good Investment rate): {city} ({city_value:.2%})")

    transport = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean()
    print("\nInvestment quality by transport access:")
    print(transport.round(4))


def run_eda():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df = load_data()

    show_basic_info(df)
    show_numeric_summary(df)
    show_categorical_summary(df)
    print_insights(df)

    plot_missing_values(df)
    plot_price_distribution(df)
    plot_quality_by_city(df)
    plot_property_type_and_state(df)
    plot_correlation_heatmap(df)
    plot_size_vs_price(df)
    plot_transport_and_investment(df)
    plot_interactive_charts(df)

    print("\n" + "=" * 80)
    print("EDA completed.")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    sns.set_style("whitegrid")
    run_eda()
