import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''Loads data from the database given. 
       Returns the input variables, output variables and the category names'''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from messages', engine)
    X = df['message']
    y = df.iloc[:, 4:39]
    y['related'].replace(2, 1, inplace=True)
    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    '''
    Removes whitespaces, reduces each word to its base form, converts to lowercase and return an array of each word
    in a message
    '''
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Builds a machine learning pipeline to train. Uses Grid Search to find the best parameters.'''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(xgb.XGBClassifier()))
        ])
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'tfidf__use_idf': (True, False)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Calculates the precision, recall and f1-score for each of the category names'''
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    evaluation = {}
    for column in Y_test.columns:
        evaluation[column] = []
        evaluation[column].append(precision_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(recall_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(f1_score(Y_test[column], y_pred_df[column]))
    print(pd.DataFrame(evaluation))


def save_model(model, model_filepath):
    '''Saves the model to a pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))