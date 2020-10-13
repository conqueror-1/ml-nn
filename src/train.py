def train_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

if __name__ == '__main__':
    print ("Need to call as a module")
else :
     
    import tensorflow.keras as keras
    import tensorflow as tf
    train_model()
