import pandas as pd
from data_loader import PricingDataLoader
import logging

# Set up reporting format
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_exploration():
    # 1. Load data using our previously created class
    loader = PricingDataLoader(raw_data_path='data/raw')
    df = loader.load_data()
    
    if df is None:
        return

    print("\n" + "="*50)
    print("BASIC PROJECT DATA ANALYSIS")
    print("="*50)

    # 2. Missing values analysis (Data quality)
    print(f"\n1. Missing Values (NaN):")
    print(df[['price', 'product_category_name']].isnull().sum())

    # 3. Price statistics
    print(f"\n2. Price statistics:")
    stats = df['price'].describe()
    print(f"- Average price: {stats['mean']:.2f} R$")
    print(f"- Minimum price: {stats['min']:.2f} R$")
    print(f"- Maximum price: {stats['max']:.2f} R$")

    # 4. Top 5 Most popular categories
    print(f"\n3. Top 5 categories by sales volume:")
    top_categories = df['product_category_name'].value_counts().head(5)
    print(top_categories)

    # 5. Time Range
    print(f"\n4. Data period:")
    print(f"From: {df['shipping_limit_date'].min()}")
    print(f"To:   {df['shipping_limit_date'].max()}")
    print("="*50)

if __name__ == "__main__":
    run_exploration()