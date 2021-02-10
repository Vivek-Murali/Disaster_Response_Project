import sys
import pandas as pd
from sqlalchemy import *
from typing import List,Dict

def load_data(messages_filepath:str, categories_filepath:str)->pd.DataFrame:
    """
    Merges the data from messages and categories csv files

    Args:
        messages_filepath (str): Path of Message File 
        categories_filepath (str): Path of Categories File 

    Returns:
        pd.DataFrame: [description]
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='outer')
    return df


def process_text(sent):
        if sent not in [" ", "\n", ""]:
            sent = sent.strip("\n")            
            sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
            sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", "", sent)
            sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
            sent = re.sub("^ +", "", sent) # remove space in front
            sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
            sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
            sent = re.sub(r"^ +(.*)", r"\1", sent) # remove space at beginning
            sent = re.sub(' {2,}', ' ', sent)
            sent = sent[0].upper() + sent[1:]
            sent = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent) # Replace all CAPS with capitalize
            return sent
        return

def clean_data(df):
    '''Splits categories into separate category columns. 
       Converts category values to just numbers 0 or 1.
       Replaces categories column in df with new category columns.
       Removes duplicates
    '''
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x.split("-")[0], categories.iloc[0].values.tolist()))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    del df['categories']
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='first')
    df['clean_message'] = df.message.apply(lambda x:process_text(x))
    df['clean_original'] = df.original.apply(lambda x: process_text(x))
    return df
    

def save_data(df, database_filename):
    '''Saves the cleaned dataframe to a table messages in the database given'''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False) 