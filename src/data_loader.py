import pandas as pd
import os
import logging

class PricingDataLoader:
    def __init__(self, raw_data_path='data/raw'):
        self.raw_data_path = raw_data_path

    def load_data(self):
        """Loads and merges order items and products datasets."""
        try:
            # We assume the user executes from the root directory
            # raw_data_path might be relative
            order_items_path = os.path.join(self.raw_data_path, 'olist_order_items_dataset.csv')
            products_path = os.path.join(self.raw_data_path, 'olist_products_dataset.csv')
            
            logging.info(f"Loading order items from {order_items_path}")
            order_items = pd.read_csv(order_items_path)
            
            logging.info(f"Loading products from {products_path}")
            products = pd.read_csv(products_path)
            
            logging.info("Merging datasets...")
            # Merge datasets on product_id
            df = pd.merge(order_items, products, on='product_id', how='left')
            
            df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'])
            
            return df
        except FileNotFoundError as e:
            logging.error(f"Error loading datasets: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None
