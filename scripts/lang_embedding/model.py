import tensorflow as tf
from lang_embedding.callbacks import KNN, Filler
import lang_embedding.models as models
import tensorflow_addons as tfa

class Model():
    def __init__(self, langs_num, feature_val_num, dropout, embedding_size):


        self.model = models.get_model_embedding(langs_num, feature_val_num, dropout, embedding_size)
        # self.model.load_weights('best.0.7443609022556391.h5')
        self.model.summary()
        optimizer = tf.optimizers.Adam()
        # 512
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

    def train(self, generator, epochs, steps_per_epoch, feature_maps, feature_maps_int):

        knn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,\
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450, 500]

        callbacks = [Filler(feature_maps, feature_maps_int), KNN(knn)]
        self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
