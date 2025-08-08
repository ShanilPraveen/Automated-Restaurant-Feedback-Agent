import pandas as pd
from data import ReviewDatabase
from tools import (
    plot_stacked_bar_chart,
    plot_line_chart,
    plot_pie_chart,
    plot_simple_bar_chart
)

def run_plotting_tests():
    print("Initializing ReviewDatabase...")
    try:
        database = ReviewDatabase('European Restaurant reviews.csv')
    except FileNotFoundError:
        print("Error: The CSV file was not found. Please ensure it is in the project root.")
        return
        
    print("Database loaded successfully.")
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2019-12-31')
    print(f"Fetching reviews from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    reviews_in_range = database.get_reviews(start_date, end_date)
    
    if reviews_in_range.empty:
        print("No reviews found for the specified date range. Please adjust the dates.")
        return
        
    print(f"Found {len(reviews_in_range)} reviews. Generating plots...")

    plot_stacked_bar_chart(reviews_in_range)
    plot_line_chart(reviews_in_range)
    plot_pie_chart(reviews_in_range)
    plot_simple_bar_chart(reviews_in_range)
    
    print("All plots generated successfully. Check your project directory for the image files.")

if __name__ == "__main__":
    run_plotting_tests()