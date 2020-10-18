import sys
import numpy as np
import logging
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from keras import Input, Model
from keras import backend as K
from keras.layers import Input, Dense, Layer, Dropout
from keras.models import Model
from keras.initializers import glorot_normal
from keras.optimizers import Adam

LOG_FILENAME = 'training.log'
logging.basicConfig(format = '%(message)s', filename=LOG_FILENAME,level=logging.INFO)


def print_log(*a, stdout=True):
    string = ' '.join([str(x) for x in a])
    logging.info(string)
    if stdout:
        print(string)
        sys.stdout.flush()    


# Random forest builds an ensemble of decision trees by training each tree on a subset of features and
# bootstrapped data
#
# The variance is found as the variance of the predictions from each decision tree in the ensemble

def random_forest(X_train, X_val, X_test, y_train, n_estimators=1000, n_jobs=12):

    print_log('Starting training')

    # Build and train model
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
    rf.fit(X_train, y_train)

    # Testing and validation set predictions
    test_predictions = rf.predict(X_test)
    val_predictions = rf.predict(X_val)
    
    # Make predictions on each individual weak learner and find variance
    estimator_predictions = np.array([pred.predict(X_test) for pred in rf.estimators_])
    variance = np.var(estimator_predictions, axis=0)
    
    print_log('Done random forest training and predictions')
    
    return test_predictions, val_predictions, variance
    
    
    
# Gradient boosting iteratively builds decision trees to improve on the previous one, the prediction 
# it produces is the average prediction from all trees produced in this process, making it an ensemble method
#
# The variance is found as the variance of the predictions from each decision tree in the ensemble

def gradient_boosting(X_train, X_val, X_test, y_train, n_estimators=1000):

    print_log('Starting training')

    # Build and train model
    gb = GradientBoostingRegressor(n_estimators = 1000)
    gb.fit(X_train, y_train)

    # Testing and validation set predictions
    test_predictions = gb.predict(X_test)
    val_predictions = gb.predict(X_val)
    
    # Make predictions on each individual weak learner, find variance
    estimator_predictions = np.array(list(gb.staged_predict(X_test)))
    variance = np.var(estimator_predictions, axis=0)
    
    print_log('Done gradient boosting training and predictions')
    
    return test_predictions, val_predictions, variance
    
    

# Neural network which uses a technique called Monte Carlo dropout
# Dropout is performed at testing time and predictions are repeated a large number of times
# This is meant to simulate training a large number of networks
# Each prediction will be slightly different depending on which connections are dropped when predicting
#
# Variance is found as the variance of all predictions

def neural_network_dropout(X_train, X_val, X_test, y_train):

    # Transform data for neural network
    X_train = StandardScaler().fit_transform(X_train)
    X_val = StandardScaler().fit_transform(X_val)
    X_test = StandardScaler().fit_transform(X_test)
    
    # Build training model, one hidden layer with 1000 neurons (adding more layers did not improve performance)
    input_layer = Input(shape=(X_train.shape[1],))

    x = Dense(1000, activation='tanh')(input_layer)
    x = Dropout(rate=0.2)(x)
    x = Dense(1)(x)

    train_model = Model(input_layer, x)
    train_model.compile(loss='mean_squared_error', optimizer='adam')

    # Implements early stopping
    callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=100)]

    # Trains model
    train_model.fit(X_train, y_train, epochs=500, callbacks = callback)
    
    # Save weights
    temp_weights = [layer.get_weights() for layer in train_model.layers]


    # Build testing model with a higher dropout rate
    input_layer = Input(shape=(X_train.shape[1],))

    x = Dense(1000, activation='tanh')(input_layer)
    x = Dropout(rate=0.4)(x, training=True)
    x = Dense(1)(x)

    test_model = Model(input_layer, x)
    test_model.compile(loss='mean_squared_error', optimizer='adam')

    # Transfer weights from training model to testing
    for i in range(len(temp_weights)):
        test_model.layers[i].set_weights(temp_weights[i])


    # Make a large number of predictions
    estimator_predictions = []
    test_predictions = []
    count = 300

    for i in range(count):
        estimator_predictions.append(test_model.predict(X_test).ravel())
    
    print_log('Finished predictions')
    estimator_predictions = np.array(estimator_predictions).transpose()

    # Point prediction is the average of all predictions made
    for x in estimator_predictions:
        test_predictions.append(x.sum() / count)

    estimator_predictions = np.swapaxes(estimator_predictions, 0, 1)
    
    # Find variance
    variance = np.var(estimator_predictions, axis=0)
    
    # Validation set predictions for kNN ICP
    val_predictions = train_model.predict(X_val).ravel()
    
    print_log('Done neural network training and predictions')
    
    return np.array(test_predictions), val_predictions, variance



# I found the proper scoring layer implemented here:
# https://medium.com/@albertoarrigoni/paper-review-code-deep-ensembles-nips-2017-c5859070b8ce
#
# While this article cites 'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles' (NIPS 2017)
# this network was originally proposed in 'Estimating the Mean and Variance of the Target Probability Distribution' (1994)

# Negative log-likelihood criterion loss function
def custom_loss(sigma):
    def log_likelihood_loss(y_true, y_pred):
        return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma)) + 1e-6
    return log_likelihood_loss

# Neural network that predicts mean and variance
class ProperScoringLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ProperScoringLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1', 
                                        shape=(1000, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
                                      
        self.kernel_2 = self.add_weight(name='kernel_2', 
                                        shape=(1000, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
                                      
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim, ),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim, ),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(ProperScoringLayer, self).build(input_shape) 
        
    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        
        x_max = K.max(output_sig)
        output_sig_pos = K.log(K.sum(K.exp(output_sig - x_max))) + x_max
        
        return [output_mu, output_sig_pos]
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]



# This is an implementation of the paper 'Estimating the Mean and Variance of the Target Probability Distribution'
# (Did not implement this one completely myself - code was found online in a Medium article (cited above the class), 
# I only had to do a bit of debugging to get it to converge)
#
# The idea is that the model itself learns to predict its own variance

def proper_scoring_method(X_train, X_val, X_test, y_train):

    # Transform data for neural network
    X_train = StandardScaler().fit_transform(X_train)
    X_val = StandardScaler().fit_transform(X_val)
    X_test = StandardScaler().fit_transform(X_test)
    
    # Build training model, one hidden layer with 1000 neurons
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(1000, activation='tanh')(inputs)
    mu, sigma = ProperScoringLayer(1, name='main_output')(x)
    model = Model(inputs, mu)
    model.compile(loss=custom_loss(sigma), optimizer=Adam(lr=1e-3))

    # Implements early stopping
    callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.04)]

    # Train model
    model.fit(X_train, y_train, epochs=500, callbacks=callback, batch_size=150)
    
    # Allows us to make predictions with our model
    get_intermediate = K.function(inputs=[model.input], outputs=model.get_layer('main_output').output)

    # Make predictions for mean and variance
    test_predictions, sigmas = [], []
    for record in X_test:
        mu, sigma = get_intermediate(np.array([record]))
        test_predictions.append(mu.reshape(1,)[0])
        sigmas.append(sigma.reshape(1,)[0])

    variance = np.abs(np.array(sigmas))
    
    # Validation set predictions for kNN ICP
    val_predictions, _ = get_intermediate(X_val)
        
    print_log('Done proper scoring method training and predictions')
    
    return np.array(test_predictions), val_predictions.ravel(), variance