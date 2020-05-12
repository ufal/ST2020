import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import evaluate

class MostCommon:
    def __init__(self, data):
        self.global_counters = [Counter() for _ in range(data.shape[1])]
        for lang in data:
            for feature_idx in range(data.shape[1]):
                if type(lang[feature_idx]) is str and lang[feature_idx] != '?':
                    self.global_counters[feature_idx][lang[feature_idx]] += 1
        
    def __call__(self, x, idx):
        return self.global_counters[idx].most_common(1)[0][0]

        
class KNN(tf.keras.callbacks.Callback):
    def __init__(self, neighbours):
        self.neighbours = neighbours   
        self.x = pd.read_csv('../../data/train_x.csv').drop(columns=['Unnamed: 0', 'wals_code', 'latitude', 'longitude', 'countrycodes']).to_numpy()
        self.x_to_predict = pd.read_csv('../../data/dev_x.csv').drop(columns=['Unnamed: 0', 'wals_code', 'latitude', 'longitude', 'countrycodes']).to_numpy()
        self.golden = pd.read_csv('../../data/dev_y.csv').drop(columns=['Unnamed: 0', 'wals_code', 'latitude', 'longitude', 'countrycodes']).to_numpy()
        self.most_common = MostCommon(self.x)     

    def on_epoch_end(self, epoch, logs=None):
        results = {}
        for n in self.neighbours:
            train = self.model.get_layer('langs').get_weights()[0][:1125]
            dev = self.model.get_layer('langs').get_weights()[0][1125:]
            dist_matrix = cosine_distances(dev, train)

            self.closest = []
            for i in dist_matrix:
                self.closest.append(np.argsort(i)[1:])

            x_to_predict = np.array(self.x_to_predict, copy=True)
            results[n] = self.knn(n, self.x, x_to_predict, self.golden, self.closest, self.most_common)
        
        print()
        for n, acc in np.array(list(results.items()))[np.argsort(list(results.values()))[-10:]]:
            print("{}: {}".format(n, acc))
        print()

    def knn(self, k, x, x_to_predict, golden, distance_matrix_sorted, fall_back):
        tmp = np.array(x_to_predict, copy=True)
        for i in range(len(x_to_predict)):
            closest_langs = distance_matrix_sorted[i][:k]
            should_fill = x_to_predict[i] == '?'
            known = x[closest_langs][:,should_fill]
            counters = [Counter() for _ in range(len(known[0]))]
            for j in known:
                for l in range(len(j)):
                    if type(j[l]) is str and j[l] != '?':
                        counters[l][j[l]] += 1
        
            for counter, idx in zip(counters, np.argwhere(should_fill==True)):
                idx=idx[0]
                prediction = counter.most_common(1)
                if len(prediction) == 0:
                    prediction = fall_back(x_to_predict[i], idx)
                else:
                    prediction = prediction[0][0]
                    
                x_to_predict[i][idx] = prediction          

        acc = evaluate.evaluate(tmp, x_to_predict, golden)
        return acc

class Filler(tf.keras.callbacks.Callback):
    def __init__(self, feature_maps, feature_maps_int):
        self.x_to_predict = pd.read_csv('../../data/dev_x.csv').drop(columns=['Unnamed: 0', 'wals_code', 'latitude', 'longitude', 'countrycodes']).to_numpy()
        self.golden = pd.read_csv('../../data/dev_y.csv').drop(columns=['Unnamed: 0', 'wals_code', 'latitude', 'longitude', 'countrycodes']).to_numpy()
        self.feature_maps = feature_maps
        self.feature_maps_int = feature_maps_int

        self.best = 0

        self.form = pd.read_csv('../../data/dev_x.csv')
        self.columns = self.form.columns

    def on_epoch_end(self, epoch, logs=None):
        print()
        print('Filler: {}, best: {}'.format(self.fill(np.array(self.x_to_predict, copy=True), self.golden), self.best))
        print()
        
    def write_results(self, predicted):
        form_copy = np.array(self.form.to_numpy(), copy=True)
        result = np.concatenate([form_copy[:,:5], predicted], axis=1)
        
        result = pd.DataFrame(data=result, columns=self.columns)
        result = result.fillna('nan')
        result.to_csv('../../outputs/lang_embedding.csv', index=False)


    def fill(self, x_to_predict, golden):
        tmp = np.array(x_to_predict, copy=True)
        cnt = 0
        for line in x_to_predict:
            should_fill = np.argwhere(line == '?').flatten()
            for j in should_fill:
                possible_values = np.array(list(self.feature_maps[j].values()))
                lang_ids = np.array([cnt+1125]*len(possible_values))
                # column_ids = np.array([j]*len(possible_values))
                prob = self.model.predict_on_batch((lang_ids, possible_values))
                prediction = np.argmax(prob)
                predicted_feature = possible_values[prediction]
                x_to_predict[cnt][j] = self.feature_maps_int[j][predicted_feature]
            
            cnt += 1

        acc = evaluate.evaluate(tmp, x_to_predict, golden)
        if acc >= self.best:
            self.model.save_weights('best.h5')
            self.best = acc
            self.write_results(x_to_predict)
        return acc
