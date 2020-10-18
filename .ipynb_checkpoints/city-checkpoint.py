import streamlit as st
import altair as alt

import SessionState
import pandas as pd

import numpy as np

from tfidf_model import calc_document_similarity_tfidf
from random_forest_model import random_forest_pred

st.beta_set_page_config(page_title='Which City?', layout='wide')

# BEGIN TIM'S BLOCK

imgdim = "140x90"
ddf = pd.read_csv('city_wiki.csv')
ddf.rename(columns={'lng':'lon'}, inplace=True) 

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

session_state = SessionState.get(df = gen_city_df(), last_cities = None, total_clicks = 0, features_importance = None, twocities = None)
# session_state.twocities = None
st.sidebar.header('Pick From Two Cities')

wordcloud_selected = st.sidebar.multiselect('Select words for word cloud', wordcloud_values)


df = session_state.df
features = df.columns[2:]
if session_state.features_importance is None:
    session_state.features_importance = pd.Series([0]*len(features), index=features)
twocities = session_state.twocities


col1, col2 = st.sidebar.beta_columns(2)
with col1:
    c1sh = st.sidebar.empty()
    button1 = st.sidebar.button(f'Pick this city', key='button1')
    c1img = st.sidebar.empty()
with col2:
    c2sh = st.sidebar.empty()
    button2 = st.sidebar.button(f'Pick this city', key='button2')
    c2img = st.sidebar.empty()

if button1 or button2 or twocities is None:
    twocities = session_state.twocities = df.sample(2)
    
c1sh.subheader(twocities.iloc[0].City)
c1name = twocities.iloc[0].City.replace(" ", "+")
c1img.markdown(f"""
![{c1name}](https://source.unsplash.com/random/{imgdim}/?{c1name},landmark)
![{c1name}](https://source.unsplash.com/random/{imgdim}/?{c1name},food)

![{c1name}](https://source.unsplash.com/random/{imgdim}/?{c1name},famous)
![{c1name}](https://source.unsplash.com/random/{imgdim}/?{c1name})
""")
c2sh.subheader(twocities.iloc[1].City)
c2name = twocities.iloc[1].City.replace(" ", "+")
c2img.markdown(f"""
![{c2name}](https://source.unsplash.com/random/{imgdim}/?{c2name},landmark)
![{c2name}](https://source.unsplash.com/random/{imgdim}/?{c2name},food)

![{c2name}](https://source.unsplash.com/random/{imgdim}/?{c2name},famous)
![{c2name}](https://source.unsplash.com/random/{imgdim}/?{c2name})
""")

reset_button = st.sidebar.button(f'Reset model')
if reset_button:
    session_state.features_importance = pd.Series([0]*len(features), index=features)
    session_state.total_clicks = 0
    

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
    t = twocities.drop(columns=['City','Movehub Rating']).T
    s = (t.iloc[:,pick] > t.iloc[:,unpick]).astype(int)
    session_state.features_importance = session_state.features_importance + s
    session_state.total_clicks += 1

feature_weights = session_state.features_importance / session_state.total_clicks

session_state.last_cities = twocities

# END TIM'S BLOCK


#access the 2 variables
col1, col2 = st.beta_columns([2,4])
if (len(wordcloud_selected) > 0):
    wf_output = calc_document_similarity_tfidf(wordcloud_selected)
if (session_state.total_clicks > 0):
    rf_output = random_forest_pred(feature_weights.values.reshape(1, -1))

with col1:
    show_input = st.checkbox("Show model input?")
    if (session_state.total_clicks > 0) and (show_input):
        rf_output = random_forest_pred(feature_weights.values.reshape(1, -1))
        st.header('Random Forest')
        st.subheader('Predicted City')
        st.write(rf_output)
        st.subheader('Feature weights:')
        st.dataframe(feature_weights)
    if (len(wordcloud_selected) > 0) and (show_input):
        st.header('Word Cloud Selection')
        st.subheader('Selected words:')
        st.markdown(wordcloud_selected)
        # wf_output = calc_document_similarity_tfidf(wordcloud_selected)
        st.write(wf_output)
with col2:
    st.header('Map')
    if (len(wordcloud_selected) > 0):
        city_forest = rf_output.split(',')[0]
        # df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
        city_forest_df = pd.DataFrame([[city_forest, 0, 0]], columns=['city', 'similarity_score', 'country'])
        temp_df = wf_output.append(city_forest_df)
        st.map((ddf[ddf['City'].isin(temp_df['city'])][['City', 'lat', 'lon']]))
        for city in wf_output['city']:
            st.subheader(city)
            st.image('wordcloud_cities2/'+city+'2.jpg')
        st.subheader(city_forest)
        st.image('wordcloud_cities2/'+city_forest+'2.jpg')
    else:
        st.map(ddf[['City', 'lat', 'lon']])