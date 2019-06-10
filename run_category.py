#%%
import importlib
from eBayPriceEstimator import ebaypriceestimator as epe 
from eBayPriceEstimator import mongodb as mdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import gensim

from xgboost import XGBRegressor
from sklearn.externals import joblib
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

importlib.reload(epe)
importlib.reload(mdb)
from sklearn.metrics import mean_squared_error

#%%
use_topic_modeling = True
regressor_type = 'scikit_random_forest'
# regressor_type = 'linear_regression'
# regressor_type = 'xgboost'


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

# corpus, id2word = create_corpus(data_words, stop_words)

#%% We now combine with the NLP topic probabilities

if use_topic_modeling:
    # Load corpus from file
    corpus_fname = 'datasets/corpus_' + str(category_id) + '.mm'
    corpus = gensim.corpora.MmCorpus(corpus_fname)
    nlp_cats = 6
    model_fname = 'lda_'+ str(nlp_cats) + '.model'
    lda_model =  gensim.models.LdaModel.load(model_fname)
    vecs = lda_model[corpus]
    prob_list = []
    index_list = []
    # We need to careful here because LDA model from gensim does not output all indexes if they are 0 <sigh>
    for v in vecs:
        row = [0 for i in range(0,nlp_cats)]
        for prob in v:
            row[prob[0]] = prob[1]
        prob_list.append(row)

    lda_df = pd.DataFrame.from_records(prob_list, columns=['nlp_' + str(cat_num) for cat_num in range(0,nlp_cats,1)])

    _df_w_lda = _df.join(lda_df,how='right')
    df_mixed = _df_w_lda.copy()

else:
    df_mixed = _df.copy()
#%%
# importlib.reload(epe)
epe.print_topics_to_file(lda_model,'topicos.txt')
#%%

# # 1. Wordcloud of Top N words in each topic
# from matplotlib import pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.colors as mcolors

# cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

# cloud = WordCloud(
#     # stopwords=stop_words,
#                   background_color='white',
#                   width=2500,
#                   height=1800,
#                   max_words=10,
#                   colormap='tab10',
#                   color_func=lambda *args, **kwargs: cols[i],
#                   prefer_horizontal=1.0)

# topics = lda_model.show_topics(formatted=False)

# fig, axes = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)

# for i, ax in enumerate(axes.flatten()):
#     fig.add_subplot(ax)
#     topic_words = dict(topics[i][1])
#     cloud.generate_from_frequencies(topic_words, max_font_size=300)
#     plt.gca().imshow(cloud)
#     plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
#     plt.gca().axis('off')


# plt.subplots_adjust(wspace=0, hspace=0)
# plt.axis('off')
# plt.margins(x=0, y=0)
# plt.tight_layout()
# plt.savefig('lda_topics.png', dpi=200)
# plt.show()



#%%


#df_mixed = _df.copy()
df_mixed = epe.convert_df_to_number(df_mixed)
#### we clean the low numbers classes as they have no use for us
# If we set 30%, we remove everything that has more than 30% NaN
max_null_pct = 50
pruned_df = epe.remove_cols_below_null_pct(df_mixed,max_null_pct)
# Now we drop all rows with no not nulls...
# We will make this better in the future,
# pruned_df = pruned_df.dropna()


#%% Let's plot the distribution of brands
# Careful, slow(!)
# importlib.reload(epe)
# epe.plot_and_save_category_counts(pruned_df,15)

#%%

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

filltype='nan_other'

_temp_df = pruned_df.copy()

if filltype == 'nan_other':
    _temp_df = _temp_df.fillna('other')
elif filltype == 'other_and_drop':
    min_occurrence_pct = 1 # what's the minimum percentag
    count_threshold = item_count*(min_occurrence_pct/100)
    _temp_df = assign_other_cat_value_below_threshold(pruned_df,count_threshold)
    _temp_df = _temp_df.dropna()
elif filltype == 'drop_nan':
    _temp_df = _temp_df.dropna()

#%%

def preprocess_df(pruned_df):
    my_df = pruned_df.copy()
    cols = my_df.columns
    num_cols = my_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    nlp_cols = []
    for col in _temp_df.columns:
        if 'nlp' in col:
            nlp_cols.append(col)
    # print(nlp_cols)
    # print('cat cols')
    # print(cat_cols)
    # We do not want to pre process our final price column
    dataset = my_df
    X = dataset.drop('price', axis=1)
    # I check if I have more numerical entries besides the price
    protected_cols = cat_cols+nlp_cols
    if len(X.drop(protected_cols, axis=1).columns) > 0:
        # Then we also make sure NLP columns are not processed(!)
        X_scaled = X.drop(protected_cols, axis=1)     
        # sc.fit_transform(X_scaled)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_scaled), columns=X_scaled.columns.values)
        #Add back in string columns
        X_scaled[protected_cols] = X[protected_cols]
    else:
        # print('here I am!')
        X_scaled = X
        # print(X_scaled)

    X_scaled = pd.get_dummies(X_scaled, columns=cat_cols, drop_first=True)
    return X_scaled

# importlib.reload(epe)
# We split into X and y
dataset = _temp_df.copy()

def signed_log10(x):
    if abs(x) < 1:
        return 0
    else:
        return np.sign(x)*np.log10(abs(x))


y = np.log(dataset['price'])
# y = dataset['price']
X = dataset.drop('price', axis=1)

X_scaled = preprocess_df(_temp_df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 10)

#%%

regressor_type = 'lasso_regression'

if regressor_type == 'scikit_random_forest':

    param_grid = {"n_estimators": [10, 20,50],
        "max_depth": [3, None],
        "max_features": [1, 3, 5, int(np.ceil(float(len(X_train.columns))/3))],
        # "max_features": [1, 3, 5, len(X_train.columns)/3],
        # 'max_features': ['sqrt', 'log2'],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 10],
        "bootstrap": [True, False]
        }

    model = RandomForestRegressor(random_state=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose = 5)
    grid.fit(X_train, y_train)

    print('best score is ' + str(grid.best_score_))
    regressor = RandomForestRegressor(**grid.best_estimator_.get_params())
    regressor.fit(X_train, y_train)
    regressor.score(X_test, y_test)
    # y_pred = regressor.predict(X_test)
    # y_pred_train = regressor.predict(X_train)
    y_pred_train = grid.best_estimator_.predict(X_train)
    y_pred_test = grid.best_estimator_.predict(X_test)
    mean_sq_err = mean_squared_error(y_test, y_pred_test)
    r_sqr = r2_score(y_test, y_pred_test)

elif regressor_type == 'xgboost':
    
    model = XGBRegressor()
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose = True)
    y_pred_train = model.predict(X_train)
    y_test = model.predict(X_test)
    r_sqr = model.score(X_test,y_test)

elif regressor_type == 'linear_regression':
    lin_regressor = LinearRegression().fit(X_train, y_train)
    y_pred_train = lin_regressor.predict(X_train)
    y_pred_test = lin_regressor.predict(X_test)
    mean_sq_err = mean_squared_error(y_test, y_pred_test)
    r_sqr = r2_score(y_test, y_test)

elif regressor_type == 'ridge_regression':
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    r_sqr = r2_score(y_test, y_pred_test)

elif regressor_type == 'lasso_regression':
    lasso = linear_model.Lasso(random_state=0, max_iter=10000)
    alphas = np.logspace(-4, -0.5, 30)
    tuned_parameters = [{'alpha': alphas}]
    n_folds = 5
    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds)
    clf.fit(X_train, y_train)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)
    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(n_folds)

    plt.semilogx(alphas, scores + std_error, 'b--')
    plt.semilogx(alphas, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

    plt.ylabel('CV score +/- std error')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([alphas[0], alphas[-1]])
    lasso = linear_model.Lasso(**clf.best_estimator_.get_params())
    clf.best_estimator_.fit(X_train, y_train)
    y_pred_train = clf.best_estimator_.predict(X_train)
    y_pred_test = clf.predict(X_test)
    r_sqr = r2_score(y_test, y_pred_test)


# elif regressor_type == 'logistic_regression':



#%%
 # Plot 
SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(figsize=[12,10])
ax1 = plt.scatter(x=y_train, y=y_pred_train,c='b', label='Train')
ax2 = plt.scatter(x=y_test, y=y_pred_test,c='r', label='Test')
plt.plot([1.5,9],[1.5,9],'k--', lw=4)
# best_score = str(round(grid.best_score_,2))
# plt_title = 'cutoff nan > ' + str(max_null_pct) + '(R^2=' + best_score + ')'
plt.ylabel('predicted price (log)')
plt.xlabel('price (log)')
if use_topic_modeling:
    plt_title = regressor_type + ' model with NLP, $R^{2}$= ' + str(round(r_sqr,3))
    plt_fname = 'bike_model_using_' + regressor_type + '_with_nlp.png'
else:
    plt_title = regressor_type + ' model without NLP, $R^{2}$= ' + str(round(r_sqr,3))
    plt_fname = 'bike_model_using_' + regressor_type + '_no_nlp.png'

plt.title(plt_title)
plt.legend(loc='upper left')
axes = plt.gca()
xlim = axes.get_xlim()
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

axes.set_xlim([0,xlim[1]])
axes.set_ylim([0,xlim[1]])

plt.savefig(plt_fname, dpi=200)
plt.show()


#%%

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
    
if use_topic_modeling:
    model_fname = 'datasets/' + regressor_type + '_nlp_' + _date + '.pkl'
else:
    model_fname = 'datasets/' + regressor_type + '_' + _date + '.pkl'
joblib.dump(regressor, model_fname)



#%%
