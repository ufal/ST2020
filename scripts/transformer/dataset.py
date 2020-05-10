import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

class Dataset:
    def __init__(self):
        self.train_x = pd.read_csv('../../data/train_x.csv')
        self.train_y = pd.read_csv('../../data/train_y.csv')
        self.dev_x = pd.read_csv('../../data/dev_x.csv')
        self.dev_y = pd.read_csv('../../data/dev_y.csv')

        self.train_x = self.preprocess_dataframe(self.train_x).to_numpy()
        self.train_y = self.preprocess_dataframe(self.train_y).to_numpy()
        self.dev_x = self.preprocess_dataframe(self.dev_x).to_numpy()
        self.dev_y = self.preprocess_dataframe(self.dev_y).to_numpy()

        self.dev_x_mask = self.dev_x == '?' 

        self.label_encoders, self.class_weights = self.create_encoders()
        self.label_encoders_lens = [len(self.label_encoders[i].classes_) for i in range(len(self.label_encoders))]

        self.train_x = self.encode_data(self.train_x)
        self.train_y = self.encode_data(self.train_y)
        self.dev_x = self.encode_data(self.dev_x)
        self.dev_y = self.encode_data(self.dev_y)

    def get_num_of_features(self):
        return self.train_x.shape[1]

    def create_encoders(self):
        label_encoders = []
        class_weights = []
        for i in range(self.train_x.shape[1]):
            concat_data = np.hstack([['?'], self.train_x[:,i], self.train_y[:,i], self.dev_x[:,i], self.dev_y[:,i]])
            encoder = LabelEncoder()
            encoder = encoder.fit(concat_data)
            label_encoders.append(encoder)
            class_weights.append(compute_class_weight('balanced', np.unique(concat_data), concat_data))
        return label_encoders, class_weights

    def preprocess_dataframe(self, dataframe):
        return dataframe.drop(columns=['Unnamed: 0', 'name', 'wals_code', 'latitude', 'longitude', 'countrycodes']).fillna('nan')

    def encode_data(self, dataframe):
        for column, encoder in zip(range(dataframe.shape[1]), self.label_encoders):
            dataframe[:, column] = encoder.transform(dataframe[:, column])
        return dataframe
        
    def random_mask(self, x):
        mask = tf.random.uniform([x.shape[0]])
        y = tf.identity(x)
        tmp = []
        for i in range(len(mask)):
            tmp.append(self.label_encoders[i].transform(['?'])[0])
        x = tf.where(mask < 0.5, x, tmp)

        return x, y

    def map_to_output(self, x, y):
        return x, y
        inputs = {}
        outputs = {}
        for i in range(1, x.shape[0]+1):
            inputs['input_{}'.format(i)] = x[i-1]
        
        for i in range(y.shape[0]):
            outputs['output_{}'.format(i)] = y[i]


        return inputs, outputs

    def batch_generator(self, dataset_x, dataset_y=None, batch_size=256):
        mask_symbols = []
        for i in range(dataset_x.shape[1]):
            mask_symbols.append(self.label_encoders[i].transform(['?'])[0])
        while True:
            if dataset_y is None:
                idxs = np.random.randint(0, dataset_x.shape[0], size=batch_size)
                batch = []
                ys = []
                masks = []
                for idx in idxs:
                    mask = np.zeros(dataset_x[idx].shape[0])
                    for column_id in range(dataset_x[idx].shape[0]):
                        if dataset_x[idx][column_id] == self.label_encoders_lens[column_id]-1:
                            mask[column_id] = 1.0
                    
                    mask[mask == 0] = np.random.uniform((mask==0).shape)

                    x = np.array(dataset_x[idx], copy=True)
                    y = np.array(dataset_x[idx], copy=True)
                    ys.append(y)

                    random_feature_values = []
                    for i in range(len(mask)):
                        random_value = np.random.choice(self.label_encoders[i].classes_)
                        random_feature_values.append(self.label_encoders[i].transform([random_value])[0])
                    
                    mask2 = np.zeros(dataset_x[idx].shape[0])
                    # slow it could use random choice from argwhere
                    for i in np.argsort(mask)[:2]:
                        x[i] = mask_symbols[i]
                        mask2[i] = 1.0

                    # x = np.where(mask < limit, x, mask_symbols)
                    # x = np.where(mask < 0.05, x, y)
                    # x = np.where(mask < 0.01, x, random_feature_values)
                    masks.append(mask2)
                    batch.append(x)

                ys = np.array(ys)
                masks = np.array(masks)
                batch = np.array(batch)
                mask_dict = {}
                for i in range(y.shape[0]):
                    mask_dict['output_{}'.format(i)] = masks[:,i]

                yield [list(batch[:,i]) for i in range(dataset_x.shape[1])], [list(ys[:,i]) for i in range(dataset_x.shape[1])], mask_dict
            else:
                yield [list(dataset_x[:,i]) for i in range(dataset_x.shape[1])], [list(dataset_y[:,i]) for i in range(dataset_x.shape[1])]


    def get_train_and_dev(self):
        # train = tf.data.Dataset.from_tensor_slices(np.asarray(self.train_y, dtype=np.int32)).shuffle(self.train_y.shape[0]).map(self.random_mask).map(self.map_to_output).cache()
        # dev = tf.data.Dataset.from_tensor_slices((np.asarray(self.dev_x, dtype=np.int32), np.asarray(self.dev_y, dtype=np.int32))).map(self.map_to_output).cache() 

        return self.batch_generator(self.train_y), self.batch_generator(self.dev_x, self.dev_y)


        
        