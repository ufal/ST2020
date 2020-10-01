import tensorflow as tf


def get_model_embedding(langs_num, feature_val_num, dropout=0.1, embedding_size=1024):
        lang_inp = tf.keras.layers.Input(shape=1)
        lang_embedding = tf.keras.layers.Embedding(langs_num, embedding_size, name='langs')(lang_inp)
        if dropout is not None:
            lang_embedding = tf.keras.layers.Dropout(dropout)(lang_embedding)
        

        feature_inp = tf.keras.layers.Input(shape=1)
        feature_embedding = tf.keras.layers.Embedding(feature_val_num, embedding_size, name='feature_value')(feature_inp)
        if dropout is not None:
            feature_embedding = tf.keras.layers.Dropout(dropout)(feature_embedding)

        merged = tf.keras.layers.Dot(name = 'dot_product', normalize = True, 
                 axes = 2)([lang_embedding, feature_embedding])
    
        x = tf.keras.layers.Reshape(target_shape = [1])(merged)

        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
        return tf.keras.Model(inputs=[lang_inp, feature_inp], outputs=output)

