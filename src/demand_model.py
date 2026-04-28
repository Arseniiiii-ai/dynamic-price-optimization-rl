import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class DemandForecaster:
    def __init__(self):
        # XGBRegressor is a powerful AI that handles non-linear relationships
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

    def prepare_training_data(self, df):
        """
        Groups data to see how many items sold each day at what price.
        """
        logging.info("Aggregating data for daily demand...")
        
        # We simplify the date to just the day (ignoring hours/minutes)
        df['date'] = df['shipping_limit_date'].dt.date
        
        # Grouping by Date and Category
        daily_sales = df.groupby(['date', 'product_category_name']).agg({
            'price': 'mean',
            'is_weekend': 'first',
            'month': 'first',
            'order_id': 'count' # This is our TARGET (Sales Volume)
        }).reset_index()
        
        daily_sales.rename(columns={'order_id': 'sales_volume'}, inplace=True)
        return daily_sales

    def train(self, df):
        logging.info("Training the Demand Forecaster (XGBoost)...")
        
        # Features (X) are the things the AI looks at
        # Target (y) is what the AI is trying to guess (Sales Volume)
        features = ['price', 'is_weekend', 'month']
        X = df[features]
        y = df['sales_volume']
        
        # We split: 80% of data to learn, 20% to test its 'honesty'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Check accuracy
        preds = self.model.predict(X_test)
        error = mean_absolute_error(y_test, preds)
        logging.info(f"Model Trained! Average Prediction Error: {error:.2f} units.")
        return error

if __name__ == "__main__":
    from data_loader import PricingDataLoader
    from feature_engineering import create_features
    
    # Run the full pipeline
    loader = PricingDataLoader()
    raw_df = loader.load_data()
    featured_df = create_features(raw_df)
    
    forecaster = DemandForecaster()
    train_data = forecaster.prepare_training_data(featured_df)
    forecaster.train(train_data)