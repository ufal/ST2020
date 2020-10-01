import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle 
from collections import defaultdict 
import sklearn

class Dataset():
    def __init__(self, clusters):
        self.train_x = pd.read_csv('../../data/train_x.csv')
        self.train_y = pd.read_csv('../../data/train_y.csv')
        self.dev_x = pd.read_csv('../../data/dev_x.csv')
        self.dev_y = pd.read_csv('../../data/dev_y.csv')
        
        self.test_x = pd.read_csv('../../data/test_x.csv')

        self.kmeans = self.create_kmeans(self.train_x['latitude'].to_numpy(), self.train_x['longitude'].to_numpy(), clusters)

        self.train_x = self.preprocess(self.train_x)
        self.train_y = self.preprocess(self.train_y)
        self.dev_x = self.preprocess(self.dev_x)
        self.dev_y = self.preprocess(self.dev_y)

        self.test_x = self.preprocess(self.test_x)


        self.lang_to_int = {}
        self.int_to_lang = {}

        self.feature_maps = [{} for i in range(self.train_x.shape[1])]
        self.feature_maps_int = [{} for i in range(self.train_x.shape[1])]

        self.feature_id_to_column_id = {}
        self.all_features = []
        
        self.global_feature_id = 0
        self.train_dataset = self.create_dataset(pd.concat([self.train_y, self.dev_x]))
        # self.train_dataset = self.create_dataset(pd.concat([self.train_y, self.test_x]))
        
        self.all_features = np.array(self.all_features)
        self.class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(self.all_features), self.all_features)
        print(self.class_weights)



    def create_kmeans(self, latitude, longitude, clusters=100):
        # 0.7374686716791979 100
        print(latitude, longitude)
        return KMeans(n_clusters=clusters).fit(np.hstack([latitude.reshape(-1, 1), longitude.reshape(-1, 1)]))


    def create_dataset(self, dataset):
        dataset = dataset.to_numpy()

        new_dataset = []
        for line in dataset:
            new_line = []
            self.add_lang(line[0])
            new_line.append(self.lang_to_int[line[0]])

            for feature, column_id in zip(line[1:], range(1, line.shape[0])):
                if not pd.isnull(feature) and feature != '?':
                    self.add_feature_value(column_id,feature)
                    new_line.append((column_id, self.feature_maps[column_id][feature]))
            
            new_dataset.append(np.array(new_line))
        
        return np.array(new_dataset)
                
    def add_feature_value(self, column_id, feature_value):
        if feature_value not in self.feature_maps[column_id]:
            self.feature_maps[column_id][feature_value] = self.global_feature_id
            self.feature_maps_int[column_id][self.global_feature_id] = feature_value
            self.feature_id_to_column_id[self.global_feature_id] = column_id
            self.global_feature_id += 1

        self.all_features.append(self.feature_maps[column_id][feature_value])


    def add_lang(self, lang_name):
        if lang_name not in self.lang_to_int:
            self.lang_to_int[lang_name] = len(self.lang_to_int)
            self.int_to_lang[len(self.int_to_lang)] = lang_name

    def preprocess(self, dataset):
        dataset['cluster'] = self.kmeans.predict(np.hstack([dataset['latitude'].to_numpy().reshape(-1, 1), dataset['longitude'].to_numpy().reshape(-1, 1)]))
        return dataset.drop(columns=['Unnamed: 0', 'wals_code', 'latitude', 'longitude', 'countrycodes'])

    def batch_generator(self, batch_size=512):
        while True:
            idxs = np.random.randint(0, self.train_dataset.shape[0], size=batch_size)
            batch = []
            for idx in idxs:
                if np.random.uniform() < 0.5:
                    while True:
                        feature_id = np.random.randint(1, self.global_feature_id)
                        column_id = self.feature_id_to_column_id[feature_id]
                        if (column_id, feature_id) not in self.train_dataset[idx]:
                            break

                    label = 0
                    # if np.random.uniform() < 0.05:
                    #     label = 1
                    batch.append((self.train_dataset[idx][0], column_id, feature_id, label, self.class_weights[feature_id]))
                else:
                    feature_id = np.random.randint(1, len(self.train_dataset[idx]))
                    column_id, feature_id = self.train_dataset[idx][feature_id]
                    label = 1
                    # if np.random.uniform() < 0.05:
                    #     label = 0
                    batch.append((self.train_dataset[idx][0], column_id, feature_id, label, self.class_weights[feature_id]))
            batch = np.array(batch)
            yield (batch[:, 0], batch[:, 2]), batch[:, 3] # ignoring column_id
