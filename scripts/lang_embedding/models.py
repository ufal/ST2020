import tensorflow as tf


def get_model_embedding(langs_num, feature_val_num, dropout=0.3, hidden=None):
        lang_inp = tf.keras.layers.Input(shape=1)
        lang_embedding = tf.keras.layers.Embedding(langs_num, 1024, name='langs')(lang_inp)
        if dropout is not None:
            lang_embedding = tf.keras.layers.Dropout(dropout)(lang_embedding)
        
        if hidden is not None:
            lang_embedding = tf.keras.layers.Dense(hidden, tf.nn.relu)(lang_embedding)

        if dropout is not None:
            lang_embedding = tf.keras.layers.Dropout(dropout)(lang_embedding)

        feature_inp = tf.keras.layers.Input(shape=1)
        feature_embedding = tf.keras.layers.Embedding(feature_val_num, 1024, name='feature_value')(feature_inp)
        if dropout is not None:
            feature_embedding = tf.keras.layers.Dropout(dropout)(feature_embedding)

        if hidden is not None:
            feature_embedding = tf.keras.layers.Dense(hidden, tf.nn.relu)(feature_embedding)

        if dropout is not None:
            feature_embedding = tf.keras.layers.Dropout(dropout)(feature_embedding)

        merged = tf.keras.layers.Dot(name = 'dot_product', normalize = True, 
                 axes = 2)([lang_embedding, feature_embedding])
    
        x = tf.keras.layers.Reshape(target_shape = [1])(merged)

        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
        return tf.keras.Model(inputs=[lang_inp, feature_inp], outputs=output)


# based on https://arxiv.org/pdf/1708.05031.pdf
def get_NeuMF(num_users, num_items, mf_dim=512, layers=[256, 128, 64, 32, 16], dropout=0.5):
    num_layer = len(layers) #Number of layers in the MLP
    
    user_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input =  tf.keras.layers.Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User =  tf.keras.layers.Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user')
    MF_Embedding_Item =  tf.keras.layers.Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item')   

    MLP_Embedding_User =  tf.keras.layers.Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user")
    MLP_Embedding_Item =  tf.keras.layers.Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item')   
    
    # MF part
    mf_user_latent =  tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
    mf_user_latent = tf.keras.layers.Dropout(dropout)(mf_user_latent)

    mf_item_latent =  tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))
    mf_item_latent = tf.keras.layers.Dropout(dropout)(mf_item_latent)

    mf_vector =  tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part 
    mlp_user_latent = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
    mlp_user_latent = tf.keras.layers.Dropout(dropout)(mlp_user_latent)

    mlp_item_latent = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))
    mlp_item_latent = tf.keras.layers.Dropout(dropout)(mlp_item_latent)


    mlp_vector = tf.keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = tf.keras.layers.Dense(layers[idx], activation='relu', name="layer%d" %idx)(mlp_vector)
        layer = tf.keras.layers.Dropout(dropout)(layer)
        mlp_vector = layer

    # Concatenate MF and MLP parts
    predict_vector = tf.keras.layers.Concatenate()([mf_vector, mlp_vector])
    predict_vector = tf.keras.layers.Dropout(dropout)(predict_vector)
    
    # Final prediction layer
    prediction = tf.keras.layers.Dense(1, activation='sigmoid', name = "prediction")(predict_vector)
    
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model