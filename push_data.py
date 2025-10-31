
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from dotenv import load_dotenv

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

print(MONGO_DB_URL)
ca=certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def cv_to_json(self, file_path):
        try:
            data=pd.read_csv(file_path)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        


FILE_PATH = 'Network_Data/phisingData.csv'
DATABASE  = 'Alvin'
COLLECTION = 'NetworkData'
network_obj = NetworkDataExtract()
records = network_obj.cv_to_json(FILE_PATH)

print(records)
length = network_obj.insert_data_mongodb(records, DATABASE, COLLECTION)

print(length)




