import pandas as pd

class ReviewDatabase:
    def __init__(self, file_path='European Restaurant reviews.csv'):
        dataset = pd.read_csv(file_path)
        dataset = dataset.dropna()
        dataset = dataset.drop_duplicates()
        dataset.drop(columns=["Country", "Restaurant Name", "Review Title"], inplace=True)
        dataset['Review Date'] = dataset['Review Date'].str.replace(' •', '')
        dataset['Review Date'] = dataset['Review Date'].str.replace('Sept', 'Sep')
        dataset['Review Date'] = pd.to_datetime(dataset['Review Date'], format='%b %Y')
        self.database = dataset

    def get_data(self):
        return self.database

    def get_reviews(self, start_date, end_date):
        filtered = self.database[
            (self.database['Review Date'] >= start_date) &
            (self.database['Review Date'] <= end_date)
        ]
        return filtered


newdb = ReviewDatabase()