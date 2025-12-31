import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract:
    def __init__(self, mongo_db_url=MONGO_DB_URL):
        try:
            self.mongo_db_url = mongo_db_url
            self.client = pymongo.MongoClient(self.mongo_db_url, tlsCAFile=ca)
            logging.info("Successfully connected to MongoDB")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def cv_to_json(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            records=list(json.loads(df.T.to_json()).values())
            return records       
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_vlient=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_vlient[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__=='__main__':
    FILE_PATH=r"D:\Machine Learning\Network_Security\network_data\phisingData.csv"
    DATABASE="NetworkSecurityDB"
    COLLECTION="PhishingDataCollection"
    obj=NetworkDataExtract()
    records=obj.cv_to_json(file_path=FILE_PATH)
    print(records)
    no_of_records=obj.insert_data_mongodb(records,DATABASE,COLLECTION)
    print(f"Number of records inserted: {no_of_records}")