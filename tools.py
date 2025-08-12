import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph,Spacer
from reportlab.lib.enums import TA_CENTER

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def save_report_as_pdf(report_text: str, filename: str):
    """
    Saves the given report text to a PDF file with improved formatting.
    """
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        styles['Heading1'].alignment = TA_CENTER
        styles['Heading1'].spaceAfter = 12
        styles['Normal'].spaceAfter = 6
        styles['BodyText'].fontName = 'Helvetica'
        
        story.append(Paragraph("Strategic Recommendations Report", styles['Heading1']))
        story.append(Spacer(1, 12))

        for line in report_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.endswith(':'):
                story.append(Paragraph(f"<b>{line}</b>", styles['Normal']))
            elif re.match(r'^\d+\.\s', line):
                story.append(Paragraph(f"  {line}", styles['Normal'])) 
            elif line.startswith('- ') or line.startswith('* '):
                 story.append(Paragraph(f"  • {line[2:]}", styles['Normal'])) 
            else:
                story.append(Paragraph(line, styles['Normal']))
                
        doc.build(story)
        print(f"Report successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving report to PDF file: {e}")




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



def analyze_sentiment(review_text:str)->str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert. Analyze the following restaurant review and classify its sentiment as 'Positive', 'Negative', or 'Neutral'. Do not add any other text, just the classification.",
                },
                {
                    "role": "user",
                    "content": review_text,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=10,
        )
        sentiment = chat_completion.choices[0].message.content.strip()
        if sentiment in ['Positive', 'Negative', 'Neutral']:
            return sentiment
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error calling Groq API for sentiment analysis: {e}")
        return "Neutral"
    

def summarize_themes_with_llm(text: str, sentiment_type: str) -> str:
    """
    Internal helper function to call the LLM for theme summarization.
    """
    try:
        prompt = (
            f"You are a sentiment analysis assistant. Summarize the top 3-5 recurring themes "
            f"or topics from the following list of {sentiment_type} restaurant reviews. "
            "List the themes concisely in a bulleted list. Do not add any extra commentary."
        )
            
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API for theme summarization: {e}")
        return "An error occurred while summarizing themes."

def get_top_themes(reviews_list: list[str], sentiment_type: str) -> str:
    """
    Identifies and summarizes the most common themes from a list of reviews of a specific sentiment.
    This function handles large numbers of reviews by processing them in batches.
    """
    all_themes = []
    current_batch = []
    current_token_count = 0
    BATCH_TOKEN_LIMIT = 1000
    
    for review in reviews_list:
        review_tokens = len(review.split())
        
        if current_token_count + review_tokens > BATCH_TOKEN_LIMIT:
            batch_text = "\n".join(current_batch)
            themes = summarize_themes_with_llm(batch_text, sentiment_type)
            all_themes.append(themes)
            
            current_batch = [review]
            current_token_count = review_tokens
        else:
            current_batch.append(review)
            current_token_count += review_tokens
            
    if current_batch:
        batch_text = "\n".join(current_batch)
        themes = summarize_themes_with_llm(batch_text, sentiment_type)
        all_themes.append(themes)
    
    return "\n".join(all_themes)

def generate_recommendations_report(database: object, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    """
    Generates a full strategic report for a given date range by performing all necessary steps.
    """
    all_reviews = database.get_reviews(start_date, end_date)

    if all_reviews.empty:
        return "No reviews found for the specified date range. Please try a different range."

    sentiment_counts_df = all_reviews.groupby(all_reviews['Review Date'].dt.date)['Sentiment'].value_counts().unstack(fill_value=0)

    positive_reviews = all_reviews[all_reviews['Sentiment'].str.lower() == 'positive']['Review'].tolist()
    negative_reviews = all_reviews[all_reviews['Sentiment'].str.lower() == 'negative']['Review'].tolist()

    positive_themes = get_top_themes(positive_reviews, 'Positive')
    #print(f"Positive Themes:\n{positive_themes}\n")
    negative_themes = get_top_themes(negative_reviews, 'Negative')
    #print(f"Negative Themes:\n{negative_themes}\n")

    data_summary = sentiment_counts_df.to_string()
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a business strategist and data analyst for a restaurant chain. "
                        "Analyze the provided sentiment data, including key positive and negative themes, "
                        "and provide a concise, actionable report. Focus on identifying trends and giving "
                        "specific, business-oriented recommendations based on the themes. Your output should be a professional, actionable report, not just a simple classification."
                        "Format the output with clear headings like 'Executive Summary:', 'Key Findings:', and 'Recommendations:'.Use numbered lists for items within these sections. Do not use any special formatting characters like Markdown (** or *)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the sentiment data:\n\n{data_summary}\n\n"
                        f"Key Positive Themes:\n{positive_themes}\n\n"
                        f"Key Negative Themes:\n{negative_themes}"
                    ),
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=5000,
        )
        report = chat_completion.choices[0].message.content.strip()
        #save_report_as_pdf(report, "recommendations_report.pdf")
        return report
    except Exception as e:
        print(f"Error calling Groq API for recommendations: {e}")
        return "An error occurred while generating recommendations."