from flask import Flask
from flask import render_template, request, jsonify
from models.classifier_handler import Classifier

import joblib
import plotly
from plotly.graph_objs import Bar
import json
from sqlalchemy import create_engine
import pandas as pd

import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pickle


app = Flask(__name__)  # '__main__'
app.secret_key = "%&@Y@*9921QW((!!!@@344323621"
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+file_path
# db = SQLAlchemy(app)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


PARENT_Folder = os.getcwd()
DATABASE = "data/disaster_response.db"
DATABASE_FILE = os.path.join(PARENT_Folder,DATABASE)
print("Database path :{}".format(DATABASE_FILE))
print(os.getcwd())
engine = create_engine('sqlite:///'+DATABASE_FILE)
df = pd.read_sql_table('messages', engine)

MODEL = 'models/classifier.pkl'
MODEL_FILE = os.path.join(PARENT_Folder,MODEL)
with open(MODEL_FILE,'rb') as fp:
    model = pickle.load(fp)


# model = joblib.load("app/models/XGB_Model.pkl")
global api 
api = Classifier(df,model)



@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    genre_names,label_counts,label_names,msg_length,msg_ids,genre_counts = api.make_data()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Category',
                'yaxis': {
                    'title': "No of Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_counts,
                    y=label_names,
                    orientation='h',
                    marker_color="#6C7C32",
                )
            ],

            'layout': {
                'title': 'Count of each topic of message',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "No of Messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=msg_ids,
                    y=msg_length,
                    marker_color="#EA553B"
                )
            ],

            'layout': {
                'title': 'Length of each message',
                'yaxis': {
                    'title': "Length"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('index.html', ids=ids, graphJSON=graphJSON)

@app.route('/fetch',methods=['POST'])
def get_category():
    query = request.form['search']
    classification_labels = api.model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(api.df.columns[6:], classification_labels))

    # This will render the results.html Please see that file. 
    return render_template(
        'results.html',
        query=query,
        classification_result=classification_results
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)