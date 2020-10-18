import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pickle


def clean_text(docs):
    lemmatizer = WordNetLemmatizer()
    lemm_docs = []

    for doc in (docs):
        words = word_tokenize(doc)
        lemm_docs.append(' '.join([lemmatizer.lemmatize(x) for x in words
                                   if x not in punctuation and not x.isdigit()]))
    return lemm_docs


wiki = pd.read_csv('archive/city_wiki.csv')
wiki = wiki[['City', 'city_wiki_text']]

location_stopwords = wiki['City'].str.lower().tolist() + ['city', 'louis', 'los', 'las', 'vegas', 'angeles', 'saint',
                                                          'petersburg', 'san', 'antonio', 'new', 'york', 'orleans',
                                                          'diego', 'francisco']

stopwords = list(STOPWORDS) + location_stopwords
docs = list(wiki['city_wiki_text'])
lemm_docs = pickle.load(open('lemmatized_docs1.pkl', 'rb'))

tfidf_vectorizer = TfidfVectorizer(max_features=20000,  # only top 5k by freq
                                   lowercase=True,  # drop capitalization
                                   ngram_range=(1, 2),
                                   min_df=2,  # note: absolute count of doc
                                   max_df=0.95,  # note: % of docs
                                   token_pattern=r'\b[a-z]{3,12}\b',  # remove short, non-word-like terms
                                   stop_words=stopwords)  # default English stopwords

tfidf_documents = tfidf_vectorizer.fit_transform(lemm_docs)
tfidf_documents_array = tfidf_documents.toarray()
feature_names = tfidf_vectorizer.get_feature_names()

results = []
words_to_remove = []

for counter, doc in enumerate(tfidf_documents_array):
    tf_idf_tuples = list(zip(feature_names, doc))
    one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples,
                                              columns=['term', 'score']).sort_values(by='score',
                                                                                     ascending=False).reset_index(
        drop=True)
    # get the first 100 important words
    results.append(one_doc_as_df.iloc[:100])
    # save all the rest just in case
    words_to_remove += (one_doc_as_df.iloc[100:]['term'].tolist())

wordcloud = WordCloud(width=800, height=800, background_color='white',
                      stopwords=stopwords, margin=4, font_step=1)

for i in range(len(results)):
    term_freq_dict = results[i].set_index('term').T.to_dict('record')[0]
    if 0 in term_freq_dict.values():
        term_freq_dict = dict([(k, v) for k, v in term_freq_dict.items() if v != 0])
    wordcloud.generate_from_frequencies(term_freq_dict)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(wiki['City'][i])
    plt.savefig('wordcloud_cities2/' + wiki['City'][i].replace('.', '') + '2.jpg')
    plt.close()
