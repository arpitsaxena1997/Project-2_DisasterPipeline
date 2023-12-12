# Disaster Response Pipeline Project

https://github.com/arpitsaxena1997/Project-2_DisasterPipeline/assets/17746091/26aa8c71-522c-4fa3-b9c8-96b4d756ff10

## Table of Contents
- [Installation](#installation)
- [Project Motivation](#motivation)
- [File Description](#file)
- [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
- Numpy
- Pandas
- Matplotlib
- Seaborn
- nltk
- Scikit-Learn
- SQLalchemy
- Pickle
- Flask
- Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/

## Project Motivation <a name="motivation"></a>
This project is made for Udacity Data Science NanoDegree 'Disaster Response' assignment.

The goal of this project is to classify disaster events from messages using ML pipeline. The dataset for project is provided by 'Figure Eight', it contains real messages.

A web app is provided in the project, in which emergency message can be entered as input, and it classify it it into various emergency categories. A demo video is attached in the starting of this file.

## File Description <a name="file"></a>
### app
    - templates:                Files for running the app, provided in boiler plate code from Udacity
    - run.py:                   Flask file used to run the app 

### data
    - disaster_categories.csv: File containing data to process, provided in boiler plate code from Udacity
    - disaster_messages.csv:   File containing data to process, provided in boiler plate code from Udacity
    - process_data.py:         ETL pipeline, used to load and process data
    - DisasterResponse.db:     Database containing processed data

### models
    - train_classifier.py:     ML pipeline, used to create the best fitting model
    - classifier.pkl:          Pickle file containing the saved model 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Acknowledgment should go to Udacity for provinding the dataset



