from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

raw_df = pd.read_csv('merged_movinghub.csv')
raw_df = raw_df.drop(['City', 'Country'], 1)
X = raw_df.iloc[:, 1:]
y = raw_df['Movehub Rating']


# Linear Models
def random_forest_train(X, y):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X)
    X = scaler.fit_transform(X)
    random_forest_regressor = RandomForestRegressor(max_depth=3, random_state=0)
    random_forest_regressor.fit(X, y)

    pickle.dump(random_forest_regressor, open('random_forest_regressor.pkl', 'wb'))


def random_forest_pred(X):
    """
    Calculate city closeness based on movehub data
    :param X:
    :return:
    """
    raw_df = pd.read_csv('merged_movinghub.csv')
    score_map = raw_df[['City', 'Country', 'Movehub Rating']]
    score_map['location'] = score_map['City'] + ', ' + score_map['Country']

    model = pickle.load(open('random_forest_regressor.pkl', 'rb'))
    result = model.predict(X)

    city_scores = score_map['Movehub Rating'].tolist()

    min_diff = 100
    for ind, score in enumerate(city_scores):
        diff = abs(result - score)
        if diff <= min_diff:
            min_diff = diff
            close_city = score_map['location'][ind]

    return close_city
