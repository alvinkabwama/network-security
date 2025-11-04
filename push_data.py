import os, sys, json
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise NetworkSecurityException("MONGO_DB_URL is not set", sys)

class NetworkDataExtract:
    def csv_to_json(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            return df.to_dict(orient="records")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database: str, collection: str) -> int:
        try:
            client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where()
            )
            db = client[database]
            col = db[collection]
            res = col.insert_many(records)
            return len(res.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

FILE_PATH = "Network_Data/phisingData.csv"
DATABASE = "Alvin"
COLLECTION = "NetworkData"

network = NetworkDataExtract()
records = network.csv_to_json(FILE_PATH)
logging.info("Loaded %d records", len(records))
count = network.insert_data_mongodb(records, DATABASE, COLLECTION)
print(count)
