import pandas as pd
import numpy as np

class Dataset():
    def __init__(self):
        self.train_x = pd.read_csv('../../data/train_x.csv')
        self.train_y = pd.read_csv('../../data/train_y.csv')
        self.dev_x = pd.read_csv('../../data/dev_x.csv')
        self.dev_y = pd.read_csv('../../data/dev_y.csv')

        self.train_x = self.preprocess(self.train_x)
        self.train_y = self.preprocess(self.train_y)
        self.dev_x = self.preprocess(self.dev_x)
        self.dev_y = self.preprocess(self.dev_y)

        self.lang_to_int = {}
        self.int_to_lang = {}

        self.feature_maps = [{} for i in range(self.train_x.shape[1])]
        self.feature_maps_int = [{} for i in range(self.train_x.shape[1])]

        self.feature_id_to_column_id = {}

        self.global_feature_id = 0
        self.train_dataset = self.create_dataset(pd.concat([self.train_y, self.dev_x]))

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

    def add_lang(self, lang_name):
        if lang_name not in self.lang_to_int:
            self.lang_to_int[lang_name] = len(self.lang_to_int)
            self.int_to_lang[len(self.int_to_lang)] = lang_name

    def preprocess(self, dataset):
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
                    batch.append((self.train_dataset[idx][0], column_id, feature_id, label))
                else:
                    feature_id = np.random.randint(1, len(self.train_dataset[idx]))
                    column_id, feature_id = self.train_dataset[idx][feature_id]
                    label = 1
                    # if np.random.uniform() < 0.05:
                    #     label = 0
                    batch.append((self.train_dataset[idx][0], column_id, feature_id, label))
            batch = np.array(batch)
            yield (batch[:, 0], batch[:, 2]), batch[:, 3] # ignoring column_id
