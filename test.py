import pandas as pd
from data import ReviewDatabase
from tools import (
    plot_stacked_bar_chart,
    plot_line_chart,
    plot_pie_chart,
    plot_simple_bar_chart,
    analyze_sentiment,
    generate_recommendations_report
)
from main import app

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


def run_sentiment_analysis_tests():
    feedback = analyze_sentiment("The food was amazing and the service was excellent!")
    print(f"Sentiment Analysis Feedback: {feedback}")

def run_recommendations_report():
    print("Generating Recommendations Report...")
    database = ReviewDatabase('European Restaurant reviews.csv')  # Create instance
    start_date = pd.to_datetime('2019-10-01')
    end_date = pd.to_datetime('2019-12-31')
    report = generate_recommendations_report(database, start_date, end_date)
    print(report)


def check_workflow(prompt: str):
    print("Checking workflow...")
    response = app.invoke({"input": prompt})
    print("\n" + "="*50)
    print("Agent Response:")
    print("="*50)
    print(response["agent_outcome"])
    print("Workflow check complete.")

if __name__ == "__main__":
    check_workflow("Respond to this review: The service was fantastic and the food was delicious!")
    # check_workflow("Respond to this review: The food was cold and the waiter was very rude.")
    # check_workflow("Show me the sentiment trend for the year 2019.")
    # check_workflow("What is the sentiment distribution for the third quarter of 2018?")
    # check_workflow("how is the sentiments of customers are changed in 2023?")
    
                        # Report Generation May Take a while
    # check_workflow("Create a report for the last half of 2018 and save it as well.")
    # check_workflow("Generate a strategic recommendations report for the period from 2019-01-01 to 2019-06-30 and save it as a PDF.")