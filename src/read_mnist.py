def read():
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train,y_train),(x_test,y_test)

if __name__ == '__main__':
    print ("Need to call as a module")
else:
    import tensorflow as tf
