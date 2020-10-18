# NYC-Taxi-Prediction-Intervals
This is the code for the paper 'Prediction Intervals of Machine Learning Models for Taxi Trip Length', accepted to be published in the conference proceedings of AMMCS 2019.

To run the random forest model with a target capture percentage of 95%:
```
python main.py --model rf --data nyc_taxi.csv --cp_target 0.95
```

Options for model are:  
&emsp;rf - random forest  
&emsp;gb - gradient boosting  
&emsp;nn - neural network with MC dropout  
&emsp;psm - neural network using a proper scoring method

The parameter cp_target represents the target capture percentage, i.e. what percent of predictions should ideally be within the interval.   
Should be a real value between 0 and 1; default is set to 0.95.
