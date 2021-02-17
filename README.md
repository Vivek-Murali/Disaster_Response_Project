# Disaster_Response_Project
This is the Simple implementation of Disaster Response Pipeline
#### Folders:
1: Notebook -> Consistis the working IPYNB Files for this pipeline
2: app -> Contains the Flask API app 
3: data-> Contains the ETL pipline code and process
4. model -> Contains the ML pipline code

File Structure
    app
    | - template
    | |- index.html # main page of web app
    | |- results.html # Serving post request to the frontend. 
    | - models
    | |- classifier_handler.py
    | - static
    | |- wordcloud.png
    | - app.py # Flask file that runs app
    data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py
    |- disaster_response.db # database to save clean data to
    models
    |- train_classifier.py
    |- classifier.pkl # saved model
    notebooks
    |- ETL_PIPELINE.ipynb
    |- ML_PIPELINE.ipynb
    README.md


#### Introduction
This ML model used for flask api was based on XGBoot as it gave better accuracy and F1-score compared ADAboost
For the ML Pipe i have used ADAboost as XBoost have dependency issue to install in Udacity ENV 


##### To Install XGboot
pip install xgboost

##### How to run the file
ETL Pipeline : python data/preprocess_data.py data/disaster_messages.csv data/disaster_categories.csv <Databasename>.db
<Databasename> = path of the Database name
Example : - python data/preprocess_data.py data/disaster_messages.csv data/disaster_categories.csv data/message.db

ML Pipeline : python models/classifier.py data/something.db <modelname>.pkl
<modelname> = path of the new model name
Example : python models/classifier.py data/message.db model/classifier.pkl

Flask API : 
1. Run the following command in the app's directory to run your web app.
    `python app/app.py`

2. Go to http://0.0.0.0:3001/






