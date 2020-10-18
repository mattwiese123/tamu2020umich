import streamlit as st
import SessionState
import pandas as pd
import numpy as np

from tfidf_model import calc_document_similarity_tfidf
from random_forest_model import random_forest_pred

st.beta_set_page_config(page_title='Which City?', layout='wide')

# BEGIN TIM'S BLOCK

def normalize_data(s):
    x, y = s.min(), s.max()
    return (s - x) / (y - x)

def gen_city_df():
    df = pd.read_csv('movehubqualityoflife.csv').merge(pd.read_csv('movehubcostofliving.csv'))
    # higher better
    df['Movehub Rating'] = normalize_data(df['Movehub Rating'])
    df['Purchase Power'] = normalize_data(df['Purchase Power'])
    df['Health Care'] = normalize_data(df['Health Care'])
    df['Quality of Life'] = normalize_data(df['Quality of Life'])
    df['Avg Disposable Income'] = normalize_data(df['Avg Disposable Income'])
    # lower better
    df['Pollution'] = normalize_data(df['Pollution'] * -1)
    df['Crime Rating'] = normalize_data(df['Crime Rating'] * -1)
    df['Cappuccino'] = normalize_data(df['Cappuccino'] * -1)
    df['Cinema'] = normalize_data(df['Cinema'] * -1)
    df['Wine'] = normalize_data(df['Wine'] * -1)
    df['Gasoline'] = normalize_data(df['Gasoline'] * -1)
    df['Avg Rent'] = normalize_data(df['Avg Rent'] * -1)
    return df


wordcloud_values = list(pd.read_csv('keywords_freq.csv').keyword)

session_state = SessionState.get(df = gen_city_df(), last_cities = None, total_clicks = 0, features_importance = None)

st.sidebar.header('Pick From Two Cities')
df = session_state.df
features = df.columns[1:]
if session_state.features_importance is None:
    session_state.features_importance = pd.Series([0]*len(features), index=features)
twocities = df.sample(2)


col1, col2 = st.sidebar.beta_columns([2,2])
with col1:
    st.sidebar.subheader(twocities.iloc[0].City)
    cname = twocities.iloc[0].City.replace(" ", "+")
    button1 = st.sidebar.button(f'Pick this city', key='button1')
    st.sidebar.markdown(f'![{cname}](https://source.unsplash.com/random/140x100?{cname},landmark) ![{cname}](https://source.unsplash.com/random/140x100/?{cname},food) ![{cname}](https://source.unsplash.com/random/140x100/?{cname},famous)')
with col2:
    st.sidebar.subheader(twocities.iloc[1].City)
    button2 = st.sidebar.button(f'Pick this city', key='button2')
    cname = twocities.iloc[1].City.replace(" ", "+")
    st.sidebar.markdown(f'![{cname}](https://source.unsplash.com/random/140x100?{cname},landmark) ![{cname}](https://source.unsplash.com/random/140x100/?{cname},food) ![{cname}](https://source.unsplash.com/random/140x100/?{cname},famous)')

wordcloud_selected = st.sidebar.multiselect('Select words for word cloud', wordcloud_values)

if button1:
    pick = 0
    unpick = 1
elif button2:
    pick = 1
    unpick = 0
else:
    pick = unpick = -1

if session_state.last_cities is not None and pick is not -1:
    st.write(f'Picked: {session_state.last_cities.iloc[pick].City}')

if button1 or button2:
    t = twocities.drop(columns=['City']).T
    s = (t.iloc[:,pick] > t.iloc[:,unpick]).astype(int)
    session_state.features_importance = session_state.features_importance + s
    session_state.total_clicks += 1

feature_weights = session_state.features_importance / session_state.total_clicks

session_state.last_cities = twocities

# END TIM'S BLOCK

#access the 2 variables
st.markdown(f'Feature weights:')
st.dataframe(feature_weights)
st.write(random_forest_pred(feature_weights[1:].values.reshape(1, -1)))
st.markdown(f'Selected words:')
st.markdown(wordcloud_selected)
st.write(calc_document_similarity_tfidf(wordcloud_selected))