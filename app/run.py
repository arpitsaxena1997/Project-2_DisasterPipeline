import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genere distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #message category distribution
    messageCategory_counts = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)
    
    #Distribution of Words in Emergency Messages
    messageData = ""
    for index,series in df.iterrows():
        messageData = messageData + series["message"]
    
    #getting list of words by removing punctuations and converting all to lowercase 
    li = re.sub(r"[^a-zA-Z0-9]",  " ",messageData).lower().strip().split()
  
    #getting top 10 used words for graph
    top100 = Counter(li).most_common(100)
    top100_clean = [item for item in top100 if item[0] not in stopwords.words("english")]


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
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        {
            'data': [
                Bar(
                    x=list(messageCategory_counts.index),
                    y=messageCategory_counts 
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[item[0] for item in top100_clean[:10]],
                    y=[item[1] for item in top100_clean[:10]]
                )
            ],

            'layout': {
                'title': 'Distribution of Words in Emergency Messages (Top 10)',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()