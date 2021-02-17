# Disaster_Response_Project
This is the Simple implementation of Disaster Response Pipeline
#### Folders:
1: Notebook -> Consistis the working IPYNB Files for this pipeline
2: app -> Contains the Flask API app 
3: data-> Contains the ETL pipline code and process
4. model -> Contains the ML pipline code

File Structure<br>
    app<br>
    | - template<br>
    | |- index.html # main page of web app<br>
    | |- results.html # Serving post request to the frontend. <br>
    | - models<br>
    | |- classifier_handler.py<br>
    | - static<br>
    | |- wordcloud.png<br>
    | - app.py # Flask file that runs app<br>
    data<br>
    |- disaster_categories.csv # data to process<br>
    |- disaster_messages.csv # data to process<br>
    |- process_data.py<br>
    |- disaster_response.db # database to save clean data to<br>
    models<br>
    |- train_classifier.py<br>
    |- classifier.pkl # saved model<br>
    notebooks<br>
    |- ETL_PIPELINE.ipynb<br>
    |- ML_PIPELINE.ipynb<br>
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






