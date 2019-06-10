#%%
from eBayPriceEstimator import ebaypriceestimator as epe
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import gensim
from gensim import corpora
import spacy

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

# corpus = epe.get_category_corpus(category_id)
# corpus_fname = 'datasets/corpus_' + str(category_id) + '.mm'
# gensim.corpora.MmCorpus.serialize(corpus_fname, corpus)

# Bikes
category_id = 177831
num_of_topics = 6

# texts = epe.get_texts_from_ebay_category(category_id)

#%%
# import importlib
# importlib.reload(epe)
# texts = epe.get_texts_from_ebay_category(category_id)
# data_words, stop_words = epe.initial_text_clean_up(texts)
# corpus, id2word = epe.create_corpus(data_words, stop_words)

#%%
texts = epe.get_texts_from_ebay_category(category_id)
#%%
data_words, stop_words = epe.initial_text_clean_up(texts)
#%%

# corpus, id2word = epe.create_corpus(data_words, stop_words)
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_words_nostops = epe.remove_stopwords(data_words, stop_words)
data_words_bigrams = epe.make_bigrams(data_words_nostops,bigram_mod)
nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = epe.lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(data_lemmatized)
corpus = [id2word.doc2bow(text) for text in data_lemmatized]

#%%

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_of_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

#%%
# importlib.reload(epe)
new_text = 'broken ultegra shimano blue and green'
new_dict = lda_model.id2word

data_words, stop_words = epe.initial_text_clean_up(texts)
print(data_words)
print(stop_words)
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_words_nostops = remove_stopwords(data_words, stop_words)
data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)
nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# new_doc = epe.prep_text_for_lda(new_text)
# new_doc = " ".join(new_doc[0])
print(new_doc)
#%%
new_corp = [new_dict.doc2bow(text) for text in new_doc]
topic_prob = lda_model[new_corp]
topic_prob = lda_model.get_document_topics(new_corp)


# vecs = lda_model.get_document_topics(new_corp)
# for v in vecs:
#     print(v)
row = [0 for i in range(0,nlp_cats)]
i=0
for topic in topic_prob[0]:
    row[topic[0]] = topic[1]
    i += 1
print(row)
# print(i)
#%%

opa = [(0, 0.10170505), (1, 0.022586359), (2, 0.061251238), (3, 0.5947561), (4, 0.16218098), (5, 0.057520278)]
len(opa)
for o in opa:
    print(o)
#%%

new_topics = lda_model[new_corp]

i=0
for topic in new_topics:
    i += 1

print(i)

#%%
model_fname = 'lda_' + str(num_of_topics) + '.model'
lda_model.save(model_fname)
print("--- %s seconds ---" % (time.time() - start_time))




#%%
import re
from gensim.utils import simple_preprocess

data = [['shimano blue red dragon']]
def sent_to_words(sentences):
    for sent in sentences:
        # sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        # sent = re.sub('\s+', ' ', sent)  # remove newline chars
        # sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


data_words = list(sent_to_words(data))
print(data_words[:1])

data_ready = process_words(data_words)  # processed Text Data!
print(data_ready)
id2word = lda_model.id2word
new_corpus = [id2word.doc2bow(text) for text in data_ready]
print(new_corpus)
#%%
