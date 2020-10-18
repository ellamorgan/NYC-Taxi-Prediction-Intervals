import sys
import numpy as np
import pandas as pd
import argparse
import models
import intervals

from datafile import get_taxi_data

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model names: rf, gb, nn, psm', required=True)
    parser.add_argument('--data', type=str, help='Path to data, default nyc_taxi.csv', default='nyc_taxi.csv')
    parser.add_argument('--cp_target', type=float, help='Capture percentage target as a decimal. Ex. 0.9 for 90%% capture percentage target', default=0.95)
    args = parser.parse_args()
    
    
    # Read data, separate features and target
    X_train, y_train, X_val, y_val, X_test, y_test = get_taxi_data(args.data)
    models.print_log('Train size: %d, validation size: %d, test size: %d' % (len(y_train), len(y_val), len(y_test)))
    
    
    # Train model
    # Random forest
    if args.model == 'rf':
        test_predictions, val_predictions, variance = models.random_forest(X_train, X_val, X_test, y_train, n_estimators=1000, n_jobs=12)
    
    # Gradient boosting
    elif args.model == 'gb':
        test_predictions, val_predictions, variance = models.gradient_boosting(X_train, X_val, X_test, y_train, n_estimators=1000)
    
    # Neural network (with MC dropout)
    elif args.model == 'nn':
        test_predictions, val_predictions, variance = models.neural_network_dropout(X_train, X_val, X_test, y_train)
    
    # Neural network with proper scoring method
    elif args.model == 'psm':
        test_predictions, val_predictions, variance = models.proper_scoring_method(X_train, X_val, X_test, y_train)


    # Prediction intervals using the variance and inherent noise method
    y_lower_var, y_upper_var = intervals.variance_and_inherent_noise(test_predictions, val_predictions, y_val, args.cp_target, variance)
    
    # Prediction intervals using the kNN inductive conformal prediction method
    y_lower_icp, y_upper_icp = intervals.knn_inductive_conformal(test_predictions, X_test, val_predictions, X_val, y_val, args.cp_target, n_neighbors=51)
    

    # Count number of targets which fall inside their prediction interval
    count_var = len([1 for a, b, c in zip(y_lower_var, y_test, y_upper_var) if b > a and b < c])
    count_icp = len([1 for a, b, c in zip(y_lower_icp, y_test, y_upper_icp) if b > a and b < c])
    
    models.print_log('Variance and Inherent Noise Method: %.2f%% of records within 95%% interval' % (100 * count_var / len(test_predictions)))
    models.print_log('kNN Inductive Conformal Prediction Method: %.2f%% of records within 95%% interval' % (100 * count_icp / len(test_predictions)))