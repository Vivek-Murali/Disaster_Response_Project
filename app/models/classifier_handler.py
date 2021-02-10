import json
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import joblib
from sqlalchemy import create_engine

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


class Classifier:
    def __init__(self):
        self.engine = create_engine('sqlite:///../data/disaster_response.db')
        self.df = pd.read_sql_table('messages1', self.engine)
        self.model = joblib.load("../models/XGB_Model.pkl")

    def classify(self,text:str)->dict:
        classification_labels = self.model.predict([text])[0]
        classification_results = dict(zip(self.df.columns[4:], classification_labels))
        return classification_results

    def create_wordcloud(self):
        stopwords = set(STOPWORDS) 
        # iterate through the csv file 
        text = " ".join(self.df['message'].tolist()[:100])
        wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(text) 
  
        # plot the WordCloud image                        
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.savefig('wordcloud.png') 

    def make_data(self):
        genre_counts = self.df.groupby('genre').count()['message']
        genre_names = [i.capitalize() for i in list(genre_counts.index)]
        label_counts = self.df.astype(bool).sum(axis=0).iloc[4:]
        label_names = list(self.df.astype(bool).sum(axis=0).iloc[4:].index)
        msg_length = self.df['message'].str.len()
        msg_ids = [i for i in range(len(self.df))]
        return genre_names,label_counts,label_names,msg_length,msg_ids,genre_counts

    def tokenize(self,text:str)->list:
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens


if __name__ == "__main__":
    api = Classifier()
    api.create_wordcloud()



