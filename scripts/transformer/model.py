import tensorflow as tf
import numpy as np
from transformer.layers import Encoder
from transformer.layers import CustomSchedule

import evaluate
import os

class Model():
    def __init__(self, num_of_features, label_encoders, embedding_size, dev_x_mask, dev_x, dev_y):
        self.num_of_features = num_of_features
        self.label_encoders = label_encoders
        # self.model = self.create_model(2048)
        self.model = self.create_attention_model(256)
        self.dev_x_mask = dev_x_mask
        self.dev_x = dev_x
        self.dev_y = dev_y
        
        
        learning_rate = CustomSchedule(512)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        self.model.compile(
            optimizer=self.optimizer,
            loss=[ tf.keras.losses.SparseCategoricalCrossentropy() for i in range(num_of_features)],
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy() for i in range(num_of_features)],
        )
        self.model.summary()
        with open('report.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


    def create_attention_model(self, d_model): #
        outs = []
        inputs = []
        for i in range(self.num_of_features):
            inputs.append(tf.keras.layers.Input(shape=1))
            x = tf.keras.layers.Embedding(len(self.label_encoders[i].classes_), d_model)(inputs[-1])
            outs.append(x)

        outs = tf.reshape(outs, [-1, self.num_of_features, d_model])
        encoder = Encoder(num_layers=8, d_model=d_model, num_heads=1, dff=256)
        outputs = encoder(outs, training=True, mask=None)
        outpus = []
        for i in range(self.num_of_features):
            x = tf.keras.layers.Dense(64, tf.nn.relu)(outputs[:, i, :])
            outpus.append(tf.keras.layers.Dense(len(self.label_encoders[i].classes_), activation=tf.nn.softmax, name='output_{}'.format(i))(x))
        
        return tf.keras.Model(inputs=inputs, outputs=outpus)
        

    def calculate_acc_on_eval(self, predictions):
        predictions = np.array(predictions)
        predictions = np.transpose(predictions)
        tmp = np.array(self.dev_x, copy=True)
        tmp[self.dev_x_mask] = '?'
        dev_x = np.array(self.dev_x, copy=True)
        dev_x[self.dev_x_mask] = predictions[self.dev_x_mask]

        return evaluate.evaluate(tmp, dev_x, self.dev_y)

    @tf.function
    def train_on_batch(self, x, y, masks, class_weights):
        with tf.GradientTape() as tape:
            probs = self.model.predict_on_batch(x)
            losses = []
            for i in range(len(probs)):
                losses.append(self.loss(y[i], probs[i], sample_weight=masks[i], class_weight=class_weights[i]))
                self.accuracy(y[i], probs[i], masks[i])
                    
        gradients = tape.gradient(losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return losses

                
    def train(self, train, dev, batch_size, epochs, class_weights):
        # train = train.batch(batch_size)
        # dev = dev.batch(batch_size)
        for epoch in range(epochs):
            self.accuracy.reset_states()
            print('Starting {} epoch'.format(epoch))
            cnt = 0
            for batch in train:
                losses = self.model.train_on_batch(batch[0], batch[1], class_weight=class_weights, reset_metrics=False, sample_weight=batch[2])
                # losses = self.train_on_batch(batch[0], batch[1], np.array(list(batch[2].values())), class_weights)
                print("Loss {}".format(np.sum(losses)))
                print("Accuracy {}".format(np.mean([i.result() for i in self.model.metrics])))
                for i in self.model.metrics:
                    i.reset_states()

                cnt += 1
                if cnt == 20: 
                    break
                                
            self.accuracy.reset_states()
            for x, y in dev:
                probs = self.model.predict_on_batch(x)
                for i in range(len(probs)):
                    self.accuracy(y[i], probs[i])
                break

            print("Dev accuracy {}".format(self.accuracy.result()))
            self.accuracy.reset_states()
            print("Accuracy via eval: {}".format(self.calculate_acc_on_eval([np.argmax(i, axis=-1) for i in probs])))

            








            