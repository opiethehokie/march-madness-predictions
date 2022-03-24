import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer

from db.cache import features_exist, read_features, write_features
from ratings.markov import markov_stats
from ratings.off_def import adjust_stats
from ml.aggregators import (custom_ratings, descriptive_stats, elo,
                            modified_rpi, statistics, time_series_stats)

TOURNEY_START_DAY = 134

# data sampling

def filter_out_of_window_games(data, features, sday, syear, eyear):
    in_window_data = (data.pipe(lambda df: df[df.Daynum >= sday])
                      .pipe(lambda df: df[df.Season >= syear])
                      .pipe(lambda df: df[df.Season <= eyear]))
    in_window_features = features.query('Season >= @syear and Season <= @eyear')
    return in_window_data, in_window_features

def sample_tourney_like_games(data, features, k=10):
    features = features.copy()
    tourney_features = features.query('Daynum >= @TOURNEY_START_DAY')
    reg_features = features.query('Daynum < @TOURNEY_START_DAY')
    nn = NearestNeighbors(n_neighbors=k).fit(reg_features)
    _, indices = nn.kneighbors(tourney_features)
    indices = indices.reshape(indices.shape[0]*indices.shape[1])
    reg_features = features.iloc[indices]
    sample_features = pd.concat([reg_features, tourney_features], axis=0)
    sample_data = pd.concat([data.iloc[indices], data[(data.Daynum >= TOURNEY_START_DAY)]], axis=0)
    return sample_data, sample_features

# model selection

def custom_train_test_split(data, features, predict_year):
    train_data = data[(data.Season != predict_year) | (data.Daynum < TOURNEY_START_DAY)]
    train_features = features.query('Season != @predict_year or Daynum < @TOURNEY_START_DAY')
    test_data = data[(data.Season == predict_year) & (data.Daynum >= TOURNEY_START_DAY) & (data.Daynum != 999)]
    test_features = features.query('Season == @predict_year and Daynum >= @TOURNEY_START_DAY and Daynum != 999')
    predict_features = features.query('Season == @predict_year and Daynum == 999')
    train_results = train_data[['Wteam', 'Lteam', 'Wscore', 'Lscore']].apply(_win, axis=1)
    test_results = test_data[['Wteam', 'Lteam', 'Wscore', 'Lscore']].apply(_win, axis=1)
    cv = _custom_cv(train_features)
    return (train_features.values.astype('float64'), test_features.values.astype('float64'), predict_features.values.astype('float64'),
            train_results.values, test_results.values, cv)

def _custom_cv(X):
    season_idx = X.index.get_level_values('Season')
    seasons = np.sort(season_idx.unique())
    day_idx = X.index.get_level_values('Daynum')
    # sort of walk-forward cross-validation
    # https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9
    return [(np.where((season_idx == season) & (day_idx < TOURNEY_START_DAY))[0],
             np.where((season_idx == season) & (day_idx >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]

def _win(df):
    return int(df.Wteam < df.Lteam)

# feature extraction

def _construct_sos(data, start_day, bust_cache=False):
    if features_exist('sos') and not bust_cache:
        return read_features('sos')
    rpi1 = pd.DataFrame(modified_rpi(data, start_day, weights=(.15, .15, .7)), columns=['rpi1'])
    rpi2 = pd.DataFrame(modified_rpi(data, start_day, weights=(.25, .25, .5)), columns=['rpi2'])
    rpi3 = pd.DataFrame(modified_rpi(data, start_day, weights=(.25, .5, .25)), columns=['rpi3'])
    sos = pd.concat([rpi1, rpi2, rpi3], axis=1)
    write_features(sos, 'sos')
    return sos

def _construct_stats(data, start_day, bust_cache=False):
    if features_exist('stats') and not bust_cache:
        return read_features('stats')
    stats1 = pd.DataFrame(statistics(data, start_day, descriptive_stats, frequency_domain=False))
    stats1.columns = ['desc-stat%s' % i for i in range(1, np.size(stats1, 1) + 1)]
    stats2 = pd.DataFrame(statistics(data, start_day, time_series_stats, frequency_domain=False))
    stats2.columns = ['time-series-stat%s' % i for i in range(1, np.size(stats2, 1) + 1)]
    stats3 = pd.DataFrame(statistics(data, start_day, descriptive_stats, frequency_domain=True))
    stats3.columns = ['fft-desc-stat%s' % i for i in range(1, np.size(stats3, 1) + 1)]
    stats4 = pd.DataFrame(statistics(data, start_day, time_series_stats, frequency_domain=True))
    stats4.columns = ['fft-time-series-stat%s' % i for i in range(1, np.size(stats4, 1) + 1)]
    stats = pd.concat([stats1, stats2, stats3, stats4], axis=1)
    write_features(stats, 'stats')
    return stats

def _construct_ratings(data, start_day, bust_cache=False):
    if features_exist('ratings') and not bust_cache:
        return read_features('ratings')
    rating1 = pd.DataFrame(custom_ratings(data, start_day, adjust_stats), columns=['off', 'def'])
    rating2 = pd.DataFrame(custom_ratings(data, start_day, markov_stats), columns=['markov'])
    rating3 = pd.DataFrame(elo(data, start_day), columns=['elo'])
    ratings = pd.concat([rating1, rating2, rating3], axis=1)
    write_features(ratings, 'ratings')
    return ratings

def _construct_other(data, start_day):
    data = data[(data.Daynum >= start_day)]
    alocation = pd.DataFrame(np.where((data.Wteam < data.Lteam) & (data.Wloc == 'H'), 1, 0), columns=['location1'])
    blocation = pd.DataFrame(np.where((data.Wteam > data.Lteam) & (data.Wloc == 'H'), 1, 0), columns=['location2'])
    #TODO should try some coach features here and save in sqlite
    location = pd.concat([alocation, blocation], axis=1)
    return location

def extract_features(data, start_day):
    other = _construct_other(data, start_day)
    sos = _construct_sos(data, start_day)
    ratings = _construct_ratings(data, start_day)
    stats = _construct_stats(data, start_day)
    features = pd.concat([other.reset_index(drop=True),
                          sos.reset_index(drop=True),
                          ratings.reset_index(drop=True),
                          stats.reset_index(drop=True)
                         ], axis=1)
    feature_data = data[(data.Daynum >= start_day)]
    features.index = pd.MultiIndex.from_arrays(feature_data[['Season', 'Daynum']].values.T, names=['Season', 'Daynum'])
    return features

# data pipeline

def prepare_data(games, future_games, start_day, start_year, predict_year, num_features=50):
    print('Starting with %d games' % games.shape[0])
    games = pd.concat([games, future_games], axis=0, sort=False, ignore_index=True)
    games = games.fillna(0)
    features = extract_features(games, start_day)
    games, features = filter_out_of_window_games(games, features, start_day, start_year, predict_year)
    games, features = sample_tourney_like_games(games, features, k=10)
    print('Using %d games' % games.shape[0])
    X_train, X_test, X_predict, y_train, y_test, cv = custom_train_test_split(games, features, predict_year)

    rf = RandomForestClassifier(n_estimators=num_features)
    selection = SelectFromModel(rf, threshold=-np.inf, max_features=num_features)
    X_train = selection.fit_transform(X_train, y_train)
    X_test = selection.transform(X_test)
    X_predict = selection.transform(X_predict)

    preprocessor = PowerTransformer(standardize=True)
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)
    X_predict = preprocessor.transform(X_predict)

    selected_features = features.columns[selection._get_support_mask()]
    print('Feature list:', ['%i:%s' % (i, selected_features[i]) for i in range(0, len(selected_features))])
    assert games.shape[0] == features.shape[0]

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    return X_train, X_test, X_predict, y_train, y_test, cv
