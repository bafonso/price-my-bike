#%%
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
import pymongo
from eBayPriceEstimator import ebaypriceestimator as epe
import importlib
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

import time
start_time = time.time()

# import myfuncs as myfn

# Mongo DB info
client = MongoClient()
db = client['ebay-db']
ebay_items_collection = db['ebay_items']
ebay_searches_collection = db['ebay_searches']
ebi = ebay_items_collection
ebs = ebay_searches_collection
from gensim import models

category_id = 177831

c = ebi.find({ "$or": [{ 'PrimaryCategoryID': str(category_id) ,
                        'ListingStatus' : 'Completed'
                        }]}, )#.limit(1000)

df = pd.DataFrame.from_records(c) #.iloc[0:2000]

#importlib.reload(epe)

texts = df.Description.to_list()

num_of_topics = 6
lda_model = epe.build_lda_model_from_text(texts,num_of_topics)
model_fname = 'lda_' + str(num_of_topics) + '.model'
lda_model.save(model_fname)
print("--- %s seconds ---" % (time.time() - start_time))


#%%
