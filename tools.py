import pandas as pd
import matplotlib.pyplot as plt

def get_sentiment_counts_by_date(reviews_df: pd.DataFrame):
    """
    Groups reviews by date and counts the number of positive, negative,
    and neutral sentiments for each day.
    """
    sentiment_counts = reviews_df.groupby([reviews_df['Review Date'].dt.date, 'Sentiment']).size().unstack(fill_value=0)
    return sentiment_counts

def get_total_sentiment_counts(reviews_df: pd.DataFrame):
    """
    Calculates the total count for each sentiment category across all reviews.
    """
    return reviews_df['Sentiment'].value_counts()

def plot_stacked_bar_chart(reviews_df: pd.DataFrame, file_name: str = "sentiment_stacked_bar.png"):
    """
    Generates a stacked bar chart showing sentiment distribution over time.
    """
    sentiment_counts = get_sentiment_counts_by_date(reviews_df)
    sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Sentiment Trends Over Time (Stacked Bar Chart)')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    return f"Plot saved to {file_name}"

def plot_line_chart(reviews_df: pd.DataFrame, file_name: str = "sentiment_line_chart.png"):
    """
    Generates a line chart to show sentiment trends over time.
    """
    sentiment_counts = get_sentiment_counts_by_date(reviews_df)
    sentiment_counts.plot(kind='line', figsize=(12, 6), marker='o')
    plt.title('Sentiment Trends Over Time (Line Chart)')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    return f"Plot saved to {file_name}"

def plot_pie_chart(reviews_df: pd.DataFrame, file_name: str = "sentiment_pie_chart.png"):
    """
    Generates a pie chart showing the overall sentiment distribution.
    """
    total_counts = get_total_sentiment_counts(reviews_df)
    plt.figure(figsize=(8, 8))
    plt.pie(total_counts, labels=total_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
    plt.title('Overall Sentiment Distribution')
    plt.axis('equal') 
    plt.savefig(file_name)
    plt.close()
    return f"Plot saved to {file_name}"

def plot_simple_bar_chart(reviews_df: pd.DataFrame, file_name: str = "sentiment_simple_bar.png"):
    """
    Generates a simple bar chart comparing total sentiment counts.
    """
    total_counts = get_total_sentiment_counts(reviews_df)
    plt.figure(figsize=(10, 6))
    total_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title('Total Sentiment Count')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    return f"Plot saved to {file_name}"