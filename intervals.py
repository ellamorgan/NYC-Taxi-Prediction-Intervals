import sys
import numpy as np
import scipy.stats as st

from sklearn.neighbors import NearestNeighbors


# An implementation of the paper 'Deep and Confident Prediction for Time Series at Uber'
# We have model uncertainty (variance) from the models, now we must find inherent noise, which measures error in the validation
# set predictions
#
# Our contribution: Applying this method to ensemble methods (random forest and gradient boosting) instead of only neural networks,
#                   and testing using a neural network which predicts its own variance (proper scoring method) instead of only
#                   using the dropout method
#
# Inputs:           test_predictions - test predictions, val_predictions - validation set predictions,
#                   y_val - validation set targets, cp_target - capture percentage target,
#                   variance - variance from prediction model used
#
# Outputs:          y_lower - a lower bound for the prediction, y_upper - an upper bound for the prediction

def variance_and_inherent_noise(test_predictions, val_predictions, y_val, cp_target, variance):

    # We will split the validation set data into d intervals grouped by size
    d = int(len(y_val) / 1000)

    # Validation predictions are sorted by the size of the prediction
    intervals = [[] for Null in range(d)]
    int_size = int(len(val_predictions) / d) + 1

    val_pred_target = np.vstack((y_val, val_predictions)).T
    val_pred_target = val_pred_target[val_pred_target[:,1].argsort()]

    # Absolute error is added to each predictions corresponding interval
    for i, val in enumerate(val_pred_target):
        intervals[int(i / int_size)].append(abs(val[1]-val[0]))

    # Ranges for the intervals are found (max value in each interval)
    ranges = []
    for i in range(1,d):
        ranges.append(val_pred_target[i * int_size,1])
    ranges.append(8000)

    prediction_intervals = []

    # Compute the squared error for each interval
    for i in intervals:
        i = (np.array(i)**2).sum() / len(i)
        prediction_intervals.append(i)

    # Assign each prediction a measurement of inherent noise based on the range it falls in
    test_intervals = []
    for pred in test_predictions:
        for i, r in enumerate(ranges):
            if pred <= r:
                test_intervals.append(prediction_intervals[i])
                break
  
  
    # Combine model uncertainty and inherent noise
    test_intervals = np.sqrt(variance + test_intervals) * st.norm.ppf(1 - (1 - cp_target) / 2)

    # Calculate intervals
    y_lower = test_predictions - test_intervals
    y_upper = test_predictions + test_intervals
    
    print('Finished variance and inherent noise')
    sys.stdout.flush() 
    
    return y_lower, y_upper
    

# An implementation of the paper 'Accelerating Difficulty Estimation for Conformal Regression Forests'
# Uses a measure of similarity (kNN) to determine how well a prediction can be made, based on previous predictions made
# The benefit of this method is its independence from the model (it also performed the best on this dataset)
#
# Inputs:  test_predictions - test predictions, X_test - testing data, val_predictions - validation set predictions,
#          X_val - validation data, y_val - validation set targets, cp_target - capture percentage target, 
#          n_neighbors - number of neighbors to compare to + 1
#
# Outputs: y_lower - a lower bound for the prediction, y_upper - an upper bound for the prediction

def knn_inductive_conformal(test_predictions, X_test, val_predictions, X_val, y_val, cp_target, n_neighbors=51):

    abs_error = np.abs(val_predictions - y_val)

    # Train nearest neighbours algorithm on the validation set
    nearest_neighbors = NearestNeighbors(n_neighbors=51)
    nearest_neighbors.fit(X_val)
    
    # Find the nearest neighbours of each record in validation set
    # (Exclude that record itself)
    distances, neighbors = nearest_neighbors.kneighbors(X_val)
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]
    
    # Calculates a measurement of difficulty for each record in the validation set
    mu = []
    for d, n in zip(distances, neighbors):
        mu.append((abs_error[n] / (d + 0.01)).sum() / (1 / (d + 0.01)).sum())
    errors = np.sort(abs_error / mu)
    
    # This is alpha_delta, after ordering the nonconformity measures the value in the
    # (size validation set) * cp_target position is selected and will get multiplied
    # with each difficulty measurement for the test set
    multiplier = errors[int(len(errors) * cp_target)]
    
    # Find nearest neighbours in validation set to the records in the test set
    distances, neighbors = nearest_neighbors.kneighbors(X_test)
    
    # Find measurement of difficulty for each record in the test set
    test_intervals = []
    for d, n in zip(distances, neighbors):
        test_intervals.append((abs_error[n] / (d + 0.01)).sum() / (1 / (d + 0.01)).sum())
        
    # Calculate intervals
    test_intervals = np.array(test_intervals) * multiplier
    y_lower = test_predictions - test_intervals
    y_upper = test_predictions + test_intervals
    
    print('Finished kNN inductive conformal prediction')
    sys.stdout.flush() 

    return y_lower, y_upper