# from ebaypriceestimator import mongodb as mdb
import pandas as pd
from . import mongodb as mdb
import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# Text related
def sanitize_text(input):
    """Sanitize text from ebay listings by removing space and dashes
    
    Arguments:
        input {string} -- string to sanitize
    
    Returns:
        string -- lower case string without spaces or dashes
    """
    input = input.lower()
    input = input.replace(' ','')
    input = input.replace('-','')
    return input

# order and clean up dictionary entries

def prune_categories(item_specific_details,min_occurrences):
    final_cats_to_use = {}
    for cat,_ in item_specific_details.items():
        _feat_group = item_specific_details[cat]
        # what = [i for i in item_specific_details[key]]
        _cats = []
        _values = []
        for entry in item_specific_details[cat] :
            for key, text in entry.items():
                _cats.append(key)
                _values.append(int(text))
                # d = {key:text for key, text in entry.items()}    
                # print(d)
        # pprint.pprint(_cats)
        # pprint.pprint(_values)
        _df = pd.DataFrame([_cats,_values] ).transpose()
        _df.columns = ['categories','occurrences']
        _df = _df.sort_values('occurrences', ascending=False)
        _df = _df[_df.occurrences > min_occurrences]
        if len(_df.index)> 1 :
            final_cat_dict = _df.set_index('categories').to_dict()
            final_cats_to_use[cat] = final_cat_dict
    return final_cats_to_use


def extract_all_features(category_id,number_of_items):
    
    items = mdb.get_ebay_item_category(category_id,number_of_items)
    # we will populate these with what we find
    item_specific_feature_name = []
    item_specific_count = dict()
    item_specific_types = dict()

    item_specific_details = dict()

    # f = open('temp_out.txt', 'w')

    for item in items:
        if 'ItemSpecifics' in item:
            _name_values = item['ItemSpecifics']['NameValueList']
            # pprint.pprint(_name_values)
            # try: # We do this for now instead of checking things manually
            if isinstance(_name_values,dict): # if we don't get a list, we simply get a dictionary
                # print('we tweak it')
                _name_values = [_name_values]
            # print(type(_name_values))
            for _name in _name_values:
                # if not isinstance(_name,dict): # we get out of the loop if we don't have a dictionary
                #     print('We have one!!')
                #     # pprint.pprint(_name_values)
                #     break
                # else:
                #     pprint.pprint(_name_values)
                # print(type(_name))
                # pprint.pprint(_name)
                _feature = _name['Name'] # the type of item specific feature, ie. "Wheel Size"
                _feature = sanitize_text(_feature)
                if isinstance(_name['Value'], str) :
                    _feature_entry = _name['Value'] # ie,  700C
                    _feature_entry = sanitize_text(_feature_entry)
                    # Trying to also get how many there are
                    if _feature not in item_specific_details: # we run into a new specific, ie, new Wheel
                        # print('yep, never seen ' + _feature)
                        # this would need to be expanded to take into account arrays of features
                        feature_list = list()
                        feature_list.append({_feature_entry: 1})
                        item_specific_details[_feature] = feature_list
                    else: # Here we need to figure out if we have seen this feature item (700c) before
                        # print('I have seen you before ' + _feature)
                        # Going to just go through it all <sigh>
                        have_seen_before = False
                        # print ('checking ' + _feature_entry + ' in ' + _feature)
                        # pprint.pprint(item_specific_details[_feature])
                        for feat_desc in item_specific_details[_feature]: # Looking for 700c in wheels
                            # print('feat_desc: ' + feat_desc ) #+ ' and item_specific_details[_feature]: ' + item_specific_details[_feature] )
                            for key,text in feat_desc.items():
                                # print ('key: ' + key + ' _feature_entry: ' + _feature_entry )
                                if key == _feature_entry:
                                    # print(feat_desc[key])
                                    feat_desc[key] += 1
                                    # print('WOHOOOO we have you! key is ' + key )
                                    have_seen_before = True
                                    break

                        if not have_seen_before:
                            item_specific_details[_feature].append({_feature_entry : 1})
    return item_specific_details

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def preprocess_df(pruned_df):
    my_df = pruned_df
    cols = my_df.columns
    num_cols = my_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    # We do not want to pre process our final price column
    dataset = my_df
    X = dataset.drop('price', axis=1)
    
    if len(X.drop(cat_cols, axis=1).columns) > 0:
        X_scaled = X.drop(cat_cols, axis=1)
        X_scaled = pd.DataFrame(sc.fit_transform(X_scaled), columns=X_scaled.columns.values)
        #Add back in string columns
        X_scaled[cat_cols] = X[cat_cols]

    else:
        X_scaled = X

    X_scaled = pd.get_dummies(X_scaled, columns=cat_cols, drop_first=True)
    return X_scaled



def remove_cols_below_null_pct(df, null_pct):
    # min_pct_cutoff = 50. 
    aha = df.notnull().sum()
    cols_to_drop = []
    for ind in aha.index.to_list():
        # print ( float(aha[ind]) / float(len(df)) )
        if ( float(aha[ind]) / float(len(df)) ) < (null_pct/100.) :
            cols_to_drop.append(ind)
        # print(ind)

    pruned_df = df.drop(cols_to_drop, axis=1)
    return pruned_df

def convert_df_to_number(_df):
    for col in _df:
        # print(col)
        # _df[col].head()
        # print(_df[col])
        _temp_df = _df[col]
        # _temp_df = _temp_df[np.isfinite(_df[col])]
        _temp_df = _temp_df[pd.notnull(_df[col])] # remove NaNs
        _df_size = len(_temp_df)
        how_many_numbers = sum(_temp_df.apply(is_number_tryexcept))
        ratio = how_many_numbers/_df_size
        if ratio > 0.9 :
            _df[col] = _df[col].apply(float)
    return _df

def get_all_items_list(items,final_cats_to_use):
    all_items_list = []
    expected_cats = len(final_cats_to_use)
    for item in items: # for every item
        _item_list = [] # list where I will append things
        # count = 1
        for model_category in final_cats_to_use: # (~10) We organize this way so we can can assemble our dataframe more easily
            # print('count is ' + str(count))
            # count +=1
            _is_nan = True
            if 'ItemSpecifics' in item:
                # print('we have specifics')
                _item_specifics_list = item['ItemSpecifics']['NameValueList']
                if isinstance(_item_specifics_list,dict):
                    # print('we tweak it')
                    _item_specifics_list = [_item_specifics_list]
                for _item_spec in _item_specifics_list : # this is a list
                    # print(_item_spec)
                    if isinstance(_item_spec,str):
                        print(_item_specifics_list)
                        print('previous')
                        print(prev)
                        print(_item_spec)
                        print(type(_item_spec))
                    else:
                        prev = _item_specifics_list
                        # print(_item_specifics_list)
                    _item_feature = _item_spec['Name'] # the type of item specific feature, ie. "Wheel Size"
                    _item_feature = sanitize_text(_item_feature)
                    # print('item_feature: ' + _item_feature + ' model_cat: ' + model_category) 
                    if _item_feature == model_category: # feature is present, let's see if contains the categories entries we want to use
                        # we now find if it has one of the categories we want, otherwise we have a missing value
                        # print(type(_item_spec['Value']))
                        if isinstance(_item_spec['Value'], str) :
                            _feat_value = sanitize_text(_item_spec['Value'])
                            for key,_model_cat in final_cats_to_use[model_category].items():
                                if _feat_value in _model_cat: # if this exists in the dictionary, we append it, otherwise we have no value
                                    # print('_feat_value: ' + _feat_value)
                                    # pprint.pprint(_model_cat)
                                    # print('_feat_value: ' + _feat_value+ ' is present in ' + _model_cat)
                                    _item_list.append(_feat_value)
                                    _is_nan = False
                                    break

            if _is_nan:
                _item_list.append(np.NaN)

        if len(_item_list) is not expected_cats:
            print('WHYYYYYYY')
            print(_item_list)
        else:    
            final_price = float(item['ConvertedCurrentPrice']['value'])
            _item_list.append(final_price)
            if len(_item_list) is not (expected_cats + 1):
                print('wait a minute')
            all_items_list.append(_item_list)
    
    return all_items_list


        