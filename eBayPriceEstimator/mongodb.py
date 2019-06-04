from pymongo import MongoClient

# Mongo DB info
client = MongoClient()
db = client['ebay-db']
ebay_items_collection = db['ebay_items']
ebay_searches_collection = db['ebay_searches']

ebi = ebay_items_collection


def get_completed_ebay_item_category(category_id, limit_value):
    ebi = ebay_items_collection
    if limit_value is not 0:
        items = ebi.find({ "$or": [{ 'PrimaryCategoryID': str(category_id) ,
                                'ListingStatus' : 'Completed'
                                }]}, ).limit(limit_value)
    else:
        items = ebi.find({ "$or": [{ 'PrimaryCategoryID': str(category_id) ,
                        'ListingStatus' : 'Completed'
                        }]}, )
    return items


def get_ebay_item_category(category_id, limit_value):
    ebi = ebay_items_collection
    # bike_categories = ['177831', '9355']
    if limit_value is not 0:
        items = ebi.find( { "$or": [{ 'PrimaryCategoryID': str(category_id) }, 
                            # { 'PrimaryCategoryID': '9355' }
                            ] } ).limit(limit_value)
    else:
        items = ebi.find( { "$or": [{ 'PrimaryCategoryID': str(category_id) }, 
                        # { 'PrimaryCategoryID': '9355' }
                        ] } )
    return items

