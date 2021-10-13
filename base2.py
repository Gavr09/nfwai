import os
import pickle
import cfgrib
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostClassifier
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass

# модули из репозитория https://github.com/sberbank-ai/no_fire_with_ai_aij2021
import helpers, preprocessing, features_generation, prepare_train
from solution import FEATURES

import warnings
warnings.simplefilter("ignore")
plt.rcParams["figure.figsize"] = (16,8)


ds = cfgrib.open_datasets('input/ERA5_data/temp_2018.grib')



times, latitudes, longitudes = preprocessing.parse_dims(ds)


lat_min = round(latitudes.min(), 1)
lat_max = round(latitudes.max(), 1)

lon_min = round(longitudes.min(), 1)
lon_max = round(longitudes.max(), 1)

lat_min, lat_max, lon_min, lon_max


step = 0.2
array_of_lats = np.arange(lat_min, lat_max, step).round(1)
array_of_lons = np.arange(lon_min, lon_max, step).round(1)
print(len(array_of_lats), len(array_of_lons))


train = prepare_train.make_train('input/train_raw.csv',
                                  array_of_lons, array_of_lats, step,
                                  start_date='2012-01-01')
print(train.shape)
if train.shape[1] == 19:
    print('фичи в порядке!')
else:
    print('фичи не в порядке!')
train.head()


train.dt.min(), train.dt.max()


sample_test = pd.read_csv('input/sample_test.csv', parse_dates=['dt'])
print(sample_test.shape)
sample_test.head()


cities_df = gpd.read_file('input/city_town_village.geojson')
cities_df = cities_df[['admin_level', 'name', 'population', 'population:date', 'place', 'geometry']]
cities_df = cities_df[cities_df.place != 'city_block'].reset_index(drop=True)
cities_df['lon'] = cities_df['geometry'].x
cities_df['lat'] = cities_df['geometry'].y

cities_df.loc[cities_df.lon < 0, 'lon'] += 360
cities_df.loc[cities_df.population.notna(), 'population'] = cities_df[cities_df.population.notna()]                            .population.apply(helpers.split_string).str.replace(" ", "").astype(int)
cities_df.head()


# ### Переводим значения city_lon и city_lat к нашей сетке

# Координаты центра каждого поселения переводим в рамки наших границ ячеек, как в таргетах

# In[19]:


cities_df = helpers.add_edges_polygon(cities_df)
cities_df = cities_df[(cities_df.lon_max <= lon_max) &                      (cities_df.lon_min >= lon_min) &                      (cities_df.lat_min >= lat_min) &                      (cities_df.lat_max <= lat_max)].reset_index(drop=True)


# ### Добавляем `grid_index` для городов

# После того как мы добавили границы для ячейки размера 0.2 x 0.2 широты - долготы, можем добавить `grid_index`

# In[20]:


cities_df = helpers.get_grid_index(cities_df, array_of_lons, array_of_lats)
cities_df.rename(columns={'lon': 'city_lon',
                          'lat': 'city_lat'}, inplace=True)

PATH_TO_ADD_DATA = 'additional_data/'

grid_list_full = [el.split('.')[0] for el in os.listdir("input/ERA5_data")]
grid_list_old = [el.split('.')[0] for el in os.listdir("input/ERA5_data")\
             if el.startswith(("temp", "wind",
                               "evaporation1", "evaporation2",
                               "heat1", "heat2", "vegetation")) and el.endswith(('2020.grib', '2021.grib'))]
grid_list = [grid for grid in grid_list_full if grid not in grid_list_old]
for file_name in grid_list:
    preprocessing.make_pool_features("input/ERA5_data",
                                     file_name, PATH_TO_ADD_DATA)

train = features_generation.add_pooling_features(train, PATH_TO_ADD_DATA, count_lag=3)
#val = features_generation.add_pooling_features(val, PATH_TO_ADD_DATA, count_lag=3)

train = features_generation.add_cat_date_features(train)
#val = features_generation.add_pooling_features(val, PATH_TO_ADD_DATA, count_lag=3)

print('Final train shape = ' + str(train.shape))

train.to_parquet('ds.parquet')

'''
cat_features = ['month', 'day', 'weekofyear', 'dayofweek', 'place']
cat_features = train[FEATURES].columns.intersection(cat_features)
cat_features = [train[FEATURES].columns.get_loc(feat) for feat in cat_features]
cat_features


# <a id='part3.1'></a>
# <h2 align="center">3.1 Многоклассовая классификация</h2>

# #### Создадим таргет - через какое количество дней начнется пожар

# In[ ]:


def get_multiclass_target(df):
    df = df.copy()
    for i in range(8, 0, -1):
        df.loc[df[f'infire_day_{i}'] == 1, 'multiclass'] = i
    df.fillna(0, inplace=True)
    return df.multiclass


# In[ ]:


train_targets = train.iloc[:,11:11+8]
val_targets = val.iloc[:,11:11+8]

train_target_mc = get_multiclass_target(train_targets)
val_target_mc = get_multiclass_target(val_targets)


# In[ ]:


train_dataset_mc = Pool(data=train[FEATURES],
                    label=train_target_mc,
                    cat_features=cat_features)

eval_dataset_mc = Pool(data=val[FEATURES],
                    label=val_target_mc,
                    cat_features=cat_features)
model_mc = CatBoostClassifier(iterations=100, random_seed=8,
                              eval_metric='MultiClass', auto_class_weights="Balanced")
model_mc.fit(train_dataset_mc,
          eval_set=eval_dataset_mc,
          verbose=False)


# <a id='part3.2'></a>
# <h2 align="center">3.2 Модель для каждого таргета</h2>

# #### Изменим таргеты в соответствии с метрикой

# In[ ]:


train_targets = (
    train_targets.replace(0, np.nan).fillna(axis=1, method="ffill").fillna(0).astype(int)
)

val_targets = (
    val_targets.replace(0, np.nan).fillna(axis=1, method="ffill").fillna(0).astype(int)
)


# In[ ]:


models = []
for i in range(8):
    train_dataset = Pool(data=train[FEATURES],
                        label=train_targets.iloc[:,i],
                        cat_features=cat_features)

    eval_dataset = Pool(data=val[FEATURES],
                        label=val_targets.iloc[:,i],
                        cat_features=cat_features)
    model = CatBoostClassifier(iterations=100, random_seed=i+1, eval_metric='F1', auto_class_weights="Balanced")
    model.fit(train_dataset,
              eval_set=eval_dataset,
              verbose=False)
    models.append(model)


# In[ ]:


if not os.path.exists("models2/"):
    os.mkdir("models2/")
for idx, model in enumerate(models):
    path_to_model = f"models2/model_{idx+1}_day.pkl"

    with open(path_to_model, 'wb') as f:  
        pickle.dump(model, f)
        
with open("models2/model_mc.pkl", 'wb') as f:  
    pickle.dump(model_mc, f)
'''