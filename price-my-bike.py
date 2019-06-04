#%%
import importlib
from eBayPriceEstimator import ebaypriceestimator as epe 
from eBayPriceEstimator import mongodb as mdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

importlib.reload(epe)
importlib.reload(mdb)

#%%
category_id = 177831
number_of_bikes = 0 # 0 is get them all
items = mdb.get_completed_ebay_item_category(category_id,number_of_bikes)
item_specific_details = epe.extract_all_features(items,category_id,number_of_bikes)
#%%
# if we set to 0 here, we get them all
item_count = items.count()
min_occurrence_pct = 1 # what's the minimum percentag
min_occurrences = item_count*(min_occurrence_pct/100)
final_cats_to_use = {}

final_cats_to_use,all_categories = epe.prune_categories(item_specific_details,min_occurrences)

#%% Get a list of all the items belonging to those categories
items.rewind()
all_items_list = epe.get_all_items_list(items, final_cats_to_use)
#%%
# We convert our list into a dataframe with price at the end
data_transposed = zip(all_items_list)     
df_cols = [key for key in final_cats_to_use]
df_cols.append('price')

_df = pd.DataFrame.from_records(all_items_list, columns=df_cols)
_df.head()

#%% We now combine with the NLP topic probabilities

corpus = epe.get_category_corpus(category_id)


#%%
nlp_cats = 6
model_fname = 'lda_'+ str(nlp_cats) + '.model'
model =  gensim.models.LdaModel.load(model_fname)
vecs = model[corpus]
prob_list = []
index_list = []
# We need to careful here because LDA model from gensim does not output all indexes if they are 0 <sigh>
for v in vecs:
    row = [0 for i in range(0,nlp_cats)]
    for prob in v[0]:
        row[prob[0]] = prob[1]
    prob_list.append(row)

lda_df = pd.DataFrame.from_records(prob_list, columns=['nlp_' + str(cat_num) for cat_num in range(0,nlp_cats,1)])

#%%

_df_w_lda = _df.join(lda_df,how='right')

#%%
# We convert out dataframe to numbers where appropriate
# importlib.reload(epe)

df_mixed = _df_w_lda.copy()
#df_mixed = _df.copy()
df_mixed = epe.convert_df_to_number(df_mixed)
#### we clean the low numbers classes as they have no use for us
# If we set 30%, we remove everything that has more than 30% NaN
max_null_pct = 50
pruned_df = epe.remove_cols_below_null_pct(df_mixed,max_null_pct)
# Now we drop all rows with no not nulls...
# We will make this better in the future,
# pruned_df = pruned_df.dropna()
#%%


#%% Let's plot the distribution of brands
# Careful, slow(!)
importlib.reload(epe)
epe.plot_and_save_category_counts(pruned_df,15)

#%%
 
# Minimum amount of counts in category
# FIX THIS SO IT ONLY USES CATEGORICAL DATA !
min_occurrence_pct = 1 # what's the minimum percentag
count_threshold = item_count*(min_occurrence_pct/100)

def assign_other_cat_value_below_threshold(pruned_df,count_threshold):
    _temp_df = pruned_df.copy()
    for _cat in _temp_df.columns:

        # if it's not a number
        if not np.issubdtype(_temp_df[_cat].dtype, np.number):
            print('cat is ' + _cat)
            
            _counts = _temp_df[_cat].value_counts()
    
            def assign_other_threshold(x):        
                if x is np.nan:
                    # print('is np.nan')
                    return x
                if np.isnan(_counts[x]):
                    # print('if np.nan counts')
                    return x
                if _counts[x] < count_threshold:
                    # print('we set other')
                    return 'other'
                else:
                    return x

            _temp_df[_cat] = _temp_df[_cat].apply(assign_other_threshold)
    return _temp_df

_temp_df = assign_other_cat_value_below_threshold(pruned_df,count_threshold)


#%%
# Right now we drop the NaN
###### Not using other
_temp_df = pruned_df
_temp_df = _temp_df.fillna('other')
# _temp_df = _temp_df.dropna()

#%%

# def preprocess_df(pruned_df):
#     my_df = pruned_df.copy()
#     cols = my_df.columns
#     num_cols = my_df._get_numeric_data().columns
#     cat_cols = list(set(cols) - set(num_cols))
#     print('cat cols')
#     print(cat_cols)
#     # We do not want to pre process our final price column
#     dataset = my_df
#     X = dataset.drop('price', axis=1)
#     # I check if I have more numerical entries besides the price   
#     if len(X.drop(cat_cols, axis=1).columns) > 0: 
#         X_scaled = X.drop(cat_cols, axis=1)
#         X_scaled = pd.DataFrame(sc.fit_transform(X_scaled), columns=X_scaled.columns.values)
#         #Add back in string columns
#         X_scaled[cat_cols] = X[cat_cols]
#     else:
#         print('here I am!')
#         X_scaled = X
#         print(X_scaled)

#     X_scaled = pd.get_dummies(X_scaled, columns=cat_cols, drop_first=True)
#     return X_scaled


# importlib.reload(epe)
# We split into X and y
dataset = _temp_df
y = np.log(dataset['price'])
# y = dataset['price']
X = dataset.drop('price', axis=1)

X_scaled = epe.preprocess_df(_temp_df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

#%%

param_grid = {"n_estimators": [200, 500],
    "max_depth": [3, None],
    # "max_features": [1, 3, 5, int(np.ceil(float(len(X_train.columns))/3))],
    "max_features": [1, 3, 5, len(X_train.columns)],
    # 'max_features': ['sqrt', 'log2'],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [True, False]
    }

model = RandomForestRegressor(random_state=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose = 5)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
# %%

print('best score is ' + str(grid.best_score_))
regressor = RandomForestRegressor(**grid.best_estimator_.get_params())
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)
y_pred = regressor.predict(X_test)

#%%
 # Plot 
plt.scatter(x=y_test, y=y_pred,c='b')
plt.plot([1.5,9],[1.5,9])
best_score = str(round(grid.best_score_,2))
plt_title = 'cutoff nan > ' + str(max_null_pct) + '(R^2=' + best_score + ')'
plt.title(plt_title)
plt.ylabel('prediced price (log)')
plt.xlabel('price (log)')
plt_fname = 'bike_model_using_other_drop_nan_cutoff_' + str(max_null_pct) + '.png'
plt_fname = 'bike_model_using_other_drop_instead_of_nan_cutoff_' + str(max_null_pct) + '.png'

plt.savefig(plt_fname, dpi=200)

plt.show()


#%%
from sklearn.externals import joblib
import pickle
from datetime import datetime

# we save everything, sharing the same date tag
today = datetime.now()
_date = today.strftime('%Y%m%d%H%M%S')
_df.to_pickle('datasets/df_raw_' + _date + '.pkl')
X.to_pickle('datasets/df_x_' + _date + '.pkl')
y.to_pickle('datasets/df_y_' + _date + '.pkl')
X_train.to_pickle('datasets/xtrain_' + _date + '.pkl')
pruned_df.to_pickle('datasets/pruned_' + _date + '.pkl')

with open('datasets/cats_to_use_' + _date + '.pkl', 'wb') as handle:
    pickle.dump(final_cats_to_use, handle, protocol=pickle.HIGHEST_PROTOCOL)
model_fname = 'datasets/rfreg_nlp_' + _date + '.pkl'
joblib.dump(regressor, model_fname)


#%%
