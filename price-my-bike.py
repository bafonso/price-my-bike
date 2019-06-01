#%%
import importlib
from eBayPriceEstimator import ebaypriceestimator as epe 
from eBayPriceEstimator import mongodb as mdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



importlib.reload(epe)
importlib.reload(mdb)

#%%
category_id = 177831
number_of_bikes = 0 # 0 is get them all
item_specific_details = epe.extract_all_features(category_id,number_of_bikes)
#%%

item_count = mdb.get_ebay_item_category(category_id,number_of_bikes).count()
min_occurrence_pct = 1 # what's the minimum percentag
min_occurrences = item_count*(min_occurrence_pct/100)
final_cats_to_use = {}

final_cats_to_use = epe.prune_categories(item_specific_details,min_occurrences)


#%% Get all the items
items = mdb.get_ebay_item_category(category_id,number_of_bikes)
#%% Get a list of all the items belonging to those categories
all_items_list = epe.get_all_items_list(items, final_cats_to_use)
#%%
# We convert our list into a dataframe with price at the end
data_transposed = zip(all_items_list)     
df_cols = [key for key in final_cats_to_use]
df_cols.append('price')
_df = pd.DataFrame.from_records(all_items_list, columns=df_cols)
_df.head()

#%%
# We convert out dataframe to numbers where appropriate
_df = epe.convert_df_to_number(_df)
#### we clean the low numbers classes as they have no use for us
pruned_df = epe.remove_cols_below_null_pct(_df,50)
# Now we drop all rows with no not nulls...
# We will make this better in the future,
pruned_df = pruned_df.dropna()

#%%

# We split into X and y
dataset = pruned_df
y = np.log(dataset['price'])
# y = dataset['price']
X = dataset.drop('price', axis=1)

X_scaled = epe.preprocess_df(pruned_df)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)


#%%

param_grid = {"n_estimators": [200, 500],
    "max_depth": [3, None],
    "max_features": [1, 3, 5, len(X_train.columns)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [True, False]}

model = RandomForestRegressor(random_state=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
# %%

print('best score is ' + str(grid.best_score_))
regressor = RandomForestRegressor(**grid.best_estimator_.get_params())

regressor.fit(X_train, y_train)

regressor.score(X_test, y_test)
y_pred = regressor.predict(X_test)

