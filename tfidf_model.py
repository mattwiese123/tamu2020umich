import pandas as pd
from word_cloud import stopwords, lemm_docs
import re
from gensim import corpora, models, similarities
import pickle
from nltk import word_tokenize


cities_country = pd.read_csv('archive/cities.csv')
cities_country = cities_country.drop_duplicates()
cities_country['City'] = cities_country['City'].apply(
    lambda x: x.split(',')[0].strip().replace('.', '').replace('á', 'a'))
city_country_dict = cities_country.set_index('City').T.to_dict('record')[0]


def clean_text2(text):
    text = text.lower()
    index_pat = '\(sx \d+\.\d\-\d+[\(\da-z\)]*\)'
    text = re.sub(index_pat, ' <index> ', text)
    month_full = 'january|february|march|april|may|june|july|august|september|october|november|december'
    month_abbr = 'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|sept'

    web = 'http[s]*:\/\/[\S]+'
    text = re.sub(web, '<url>', text)

    num_pat = '[\s][\d\.\,\- \;\(\)]{2,}[\s]'
    text = re.sub(num_pat, ' <num> ', text)

    mon1 = r'[ ]*(?:%s) <num>[ ]*' % month_full
    mon2 = r'[ ]*(?:%s)[ .]* <num>[ ]*' % month_abbr
    text = re.sub(mon1, ' <date> ', text)
    text = re.sub(mon2, ' <date> ', text)

    text = re.sub('\$[ ]*[\d\,\.\<num\>]+', '<money>', text)
    text = re.sub('\s{2,}', ' ', text)

    text_token = word_tokenize(text)
    text = ' '.join([x for x in text_token if x not in stopwords])

    return text


# lemm_docs2=[clean_text2(x) for x in lemm_docs]
lemm_docs2 = pickle.load(open('lemmatized_docs2.pkl', 'rb'))
list_of_cities_tokens = [x.split() for x in lemm_docs2]

wiki = pd.read_csv('archive/city_wiki.csv')
wiki = wiki[['City', 'city_wiki_text']]
cities = wiki['City'].tolist()


def train_tfidf(list_of_cities_cleaned=lemm_docs2):
    dictionary = corpora.Dictionary([x.split() for x in list_of_cities_cleaned])
    dictionary.filter_n_most_frequent(30)

    corpus = [dictionary.doc2bow(x.split()) for x in list_of_cities_cleaned]
    tfidf = models.TfidfModel(corpus)

    pickle.dump(tfidf, open('tfidf_gensim.pkl', 'wb'))
    pickle.dump(dictionary, open('dictionary_gensim.pkl', 'wb'))
    pickle.dump(corpus, open('corpus_gensim.pkl', 'wb'))


def calc_document_similarity_tfidf(list_of_words, tfidf=pickle.load(open('tfidf_gensim.pkl', 'rb'))):
    """

    :param list_of_words: the words that user choose to be important
    :param tfidf: load the trained model, keep as default
    :return: return ranked cities based on similarity
    """

    dictionary = pickle.load(open('dictionary_gensim.pkl', 'rb'))
    corpus = pickle.load(open('corpus_gensim.pkl', 'rb'))
    feature_cnt = len(dictionary.token2id)

    list_of_words2 = []
    for word in list_of_words:
        if ' ' in word:
            list_of_words2 += word.split()
        else:
            list_of_words2.append(word)

    kw_vector = dictionary.doc2bow(list_of_words2)

    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[kw_vector]]

    results = pd.DataFrame([cities, sim]).T
    results.columns = ['city', 'similarity_score']
    results = results.sort_values('similarity_score', ascending=False)

    # return top 5 cities with similarity scores
    results_df = results.reset_index().drop('index', axis=1).iloc[:5]

    top_cities = results_df['city'].tolist()

    # check if any city in US
    city_in_us = False
    for city in top_cities:
        if city_country_dict[city.replace('_', ' ').replace('.', '')] == 'United States':
            city_in_us = True
            break
        else:
            continue

    if not city_in_us:
        most_sim_foreign_city = top_cities[0]
        foreign_to_us = pd.read_csv('city_similarity.csv')
        most_sim_us_city = foreign_to_us[foreign_to_us['city'] == most_sim_foreign_city]['most_similar_us_city']
        results_df.iloc[0] = [most_sim_us_city, 1]

    results_df['country'] = results_df['city'].apply(lambda x: city_country_dict[x.replace('_', ' ').replace('.', '')])

    return results_df


def calc_pairwise_city_similarities():
    try:
        return pd.read_csv('city_similarity.csv')
    except:
        pairwise_sims = {}

        for city, tokens in zip(cities, list_of_cities_tokens):
            temp = calc_document_similarity_tfidf(tokens)
            for x in temp['city']:
                # just in case there is special characters that are not cleaned
                try:
                    if city_country_dict[x.replace('_', ' ').replace('.', '')] == 'United States':
                        most_similar_US_city = x
                        break
                except:
                    continue
            most_similar_cities = list(temp['city'][1:6])
            pairwise_sims[city] = [most_similar_US_city] + most_similar_cities

        pairwise_sims_df = pd.DataFrame(pairwise_sims).T.reset_index()
        pairwise_sims_df.columns = ['city', 'most_similar_us_city', '1', '2', '3', '4', '5']
        pairwise_sims_df.to_csv('city_similarity.csv')


if __name__ == '__main__':
    train_tfidf()
