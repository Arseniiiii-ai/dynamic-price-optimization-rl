import pandas as pd 
import numpy as np 
import logging

#Set up logging sowe can track the process in the terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_features(df):
    """Translates raw data into mathematical signals for the Machine Learning model."""
    if df is None:
        logging.error("No data provided to feature engineering!")
        return None 

    logging.info("Starting feature Engineering...")
    # 1. Temporal Features (Time signals)
    # We extract the day (0=Monday, 6=Sunday) and the month.
    df['day_of_week'] = df['shipping_limit_date'].dt.dayofweek
    df['month'] = df['shipping_limit_date'].dt.month

    # 2. Binary Flags (Weekend signal)
    # Logic: If day is 5 (Sat) or 6 (Sun), it's a weekend.
    # Why? E-commerce behavior changes when people are off work.
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Pricing Context (The "Expensive vs Cheap" signal)
    # We calculate how this specific price compares to the average of its category.
    # Logic: 1.0 means average, 1.2 means 20% more expensive, 0.8 means 20% cheaper.
    category_avg = df.groupby('product_category_name')['price'].transform('mean')
    df['price_vs_category_avg'] = df['price'] / category_avg

    # 4. Data cleaning
    # If a category is missing, we label it 'unknown' so the math doesn't break.
    df['product_category_name'] = df['product_category_name'].fillna('unknown')
    
    logging.info(f"Features created. Data shape: {df.shape}")
    return df

if __name__ == "__main__":
    # This part allows us to test the script individually 
    from data_loader import PricingDataLoader

    loader = PricingDataLoader()
    raw_data = loader.load_data()

    if raw_data is not None: 
        features_df = create_features(raw_data)
        print( "\n--- Feature Priview (New Columns) ---")
        # We only show the columns we just created to verify they work 
        cols_to_show = ['shipping_limit_date', 'day_of_week', 'is_weekend', 'price_vs_category_avg']
        print(features_df[cols_to_show].head())
    