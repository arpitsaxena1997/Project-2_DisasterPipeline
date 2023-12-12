import sys

import re
import pickle
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


def load_data(database_filepath):
    '''
    load data from database

    Args:
        database_filepath: path to db
    Returns:
        X: training data
        Y: testing Data
        category_names: list of category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, con=engine)
    X = df.message
    Y = df[df.columns[4:]]

    return X, Y, Y.columns



def tokenize(text):
    '''
    Method to process the text data into lemmatized tokens

    Args:
        text: text data to be processed
    Returns:
        list: clean_tokens list with tokens extracted from the processed text data 
    '''
    # normalize case and remove leading/trailing white space and punctuation
    text = re.sub("\W"," ", text.lower().strip())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # initiate stopword
    stop_words = stopwords.words("english")
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token to
    # lemmatize and remove stopwords  
    clean_tokens = []
    
    for tok in tokens:
        if tok not in stop_words:

            clean_tok = lemmatizer.lemmatize(tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds ML pipeline using grid search

    Args:
        None
    Results:
        model which is optimised    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_features': (None, 10000),        
        
        'clf__estimator__n_estimators': [50,100]     
    }

    return GridSearchCV(pipeline, param_grid=parameters,verbose = 3, n_jobs=-1, scoring='f1_samples')                       


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evalute and fing the best model

    Args:
        model: model to be evaluated
        X_test: messages to predict from
        Y_test: categories of the messages
        category_names: 
    Results:
        None
    '''
    pred = model.predict(X_test)
    eval = classification_report(Y_test, pred, target_names=category_names)

    print(eval)


def save_model(model, model_filepath):
    '''
    Saves model in pickle file

    Args:
        model: model to be saved
        model_filepath: model saving path
    Results:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()