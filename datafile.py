import sys
import pandas as pd
import numpy as np
import datetime as dt
import argparse

from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans


def get_taxi_data(datafile):

    df = pd.read_csv(datafile)
    
    y = np.array(df.loc[:,'trip_time_in_secs'])
    df = df.drop('trip_time_in_secs', 1)
    X = np.array(df)
    feats = np.array(df.columns)
    
    # Split into training, validation, and testing sets with a 60/20/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Processes dataset
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to datafile to be processed', required=True)
    parser.add_argument('--days', type=str, help='Days of month as ints separated by commas. Ex. 2,4,5 for tues, thurs, fri', default='-1')
    parser.add_argument('--hours', type=str, help='Hours of day as ints separated by commas. Ex. 11,12,13 for 11AM, 12PM, 1PM', default='-1')
    args = parser.parse_args()

    df = pd.read_csv(args.file, low_memory=False)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    l1 = len(df)
    print('%d total records' % (l1))
    sys.stdout.flush() 

    # Create hour and day of month columns
    # Only keep records on days in 'days' and hours in 'hours'
    pickup = pd.to_datetime(df['tpep_pickup_datetime'])

    df['pickup_day'] = pickup.apply(lambda e: e.day)
    if args.days != '-1':
        days = [int(x) for x in args.days.split(',')]
        df = df[df['pickup_day'].isin(days)]
    else:
        days = range(1,32)

    df['pickup_hour'] = pickup.apply(lambda e: e.hour)
    if args.hours != '-1':
        hours = [int(x) for x in args.hours.split(',')]
        df = df[df['pickup_hour'].isin(hours)]
    else:
        hours = range(24)

    l2 = len(df)
    print('%d records within the specified days and hours' % (l2))
    sys.stdout.flush() 

    # Find trip time in seconds
    df['trip_time_in_secs'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds()

    # Drop records with a trip distance of 0
    df = df.drop(df['trip_distance'].index[df['trip_distance'] == 0.0])
    l3 = len(df)
    print('%d records removed which had a trip distance of 0' % (l2-l3))
    sys.stdout.flush() 

    # Drop records < 30 seconds or > 2 hours
    df = df.drop(df['trip_time_in_secs'].index[df['trip_time_in_secs'] < 30])
    df = df.drop(df['trip_time_in_secs'].index[df['trip_time_in_secs'] > 7200])
    l4 = len(df)
    print('%d records removed which had a trip length less than 30 seconds or more than 2 hours' % (l3-l4))
    sys.stdout.flush() 


    # Drop records which started or ended outside of NYC
    df = df.drop(df['pickup_longitude'].index[df['pickup_longitude'] < -74.03])
    df = df.drop(df['pickup_longitude'].index[df['pickup_longitude'] > -73.75])
    df = df.drop(df['pickup_latitude'].index[df['pickup_latitude'] < 40.63])
    df = df.drop(df['pickup_latitude'].index[df['pickup_latitude'] > 40.83])
    df = df.drop(df['dropoff_longitude'].index[df['dropoff_longitude'] < -74.03])
    df = df.drop(df['dropoff_longitude'].index[df['dropoff_longitude'] > -73.75])
    df = df.drop(df['dropoff_latitude'].index[df['dropoff_latitude'] < 40.63])
    df = df.drop(df['dropoff_latitude'].index[df['dropoff_latitude'] > 40.83])
    l5 = len(df)
    print('%d records removed which started or ended outside of NYC' % (l4-l5))
    sys.stdout.flush() 


    # Find average trip speeds, drop records with average speeds > 100km/h (likely an error if driving in NYC)
    length = list(df.loc[:,'trip_time_in_secs'])
    distance = df['trip_distance']
    df['trip_speed'] = 3600 * distance / length
    df = df.drop(df['trip_speed'].index[df['trip_speed'] > 100])
    l6 = len(df)
    print('%d records removed with average speed > 100km/h' % (l5-l6))

    print()
    print('Done removing records')
    print('%d total records removed' % (l1-l6))
    print('%d records remaining' % (l6))
    print('Adding average speed and trips per hour to dataset next')
    sys.stdout.flush() 


    # Find average speeds and number of trips each hour
    data = np.array(df)
    total_speeds = np.zeros((len(days),len(hours)))
    count = np.zeros((len(days),len(hours)))

    h_index = list(df.columns).index('pickup_hour')
    d_index = list(df.columns).index('pickup_day')
    s_index = list(df.columns).index('trip_speed')

    for d in data:
        day = days.index(d[d_index])
        hour = hours.index(d[h_index])
        total_speeds[day,hour] += d[s_index]
        count[day,hour] += 1
    average_speeds = total_speeds / count


    # For each record, find the average speed and number of trips in the previous hour
    avg_speeds = []
    trip_counts = []
    for d in data:
        day = days.index(d[d_index])
        hour = hours.index(d[h_index])
        
        if hour == 0:
            if day - 1 in days:
                avg_speeds.append(average_speeds[day-1,23])
                trip_counts.append(count[day-1,23])
            else:
                avg_speeds.append(0)
                trip_counts.append(1)
        else:
            avg_speeds.append(average_speeds[day,hour-1])
            trip_counts.append(count[day,hour-1])

    # Add to dataset
    df['avg_speed'] = avg_speeds
    df['trip_count'] = trip_counts
    
    
    print('Done. Clustering pickup and dropoff locations into neighbourhoods next')
    sys.stdout.flush() 

    # Cluster pickup/dropoff locations into neighborhoods
    coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                        df[['dropoff_latitude', 'dropoff_longitude']].values))
    kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10000).fit(coords)

    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
    
    
    print('Done. Finding bearing and checking if airport pickup or dropoff next')
    sys.stdout.flush()


    # Find bearing and check if airport pickup/dropoff
    # North = 0, North East = 1
    # East = 2, South East = 3
    # South = 4, South West = 5
    # West = 6, North West = 7

    bearing = []

    jfk_dropoff = []
    jfk_pickup = []
    laguardia_dropoff = []
    laguardia_pickup = []

    p_lat_index = list(df.columns).index('pickup_latitude')
    p_long_index = list(df.columns).index('pickup_longitude')
    d_lat_index = list(df.columns).index('dropoff_latitude')
    d_long_index = list(df.columns).index('dropoff_longitude')

    for d in data:
        d_lat = d[d_lat_index]
        p_lat = d[p_lat_index]
        d_long = d[d_long_index]
        p_long = d[p_long_index]
        
        del_lat = d_lat - p_lat
        del_long = d_long - p_long
        
        if (abs(del_lat) > (2.414 * abs(del_long))):
            bearing.append(0) if (del_lat > 0) else bearing.append(4)
        elif (abs(del_lat) < (0.414 * abs(del_long))):
            bearing.append(2) if (del_long > 0) else bearing.append(6)
        else:
            if (del_long > 0):
                bearing.append(1) if (del_lat > 0) else bearing.append(3)
            else:
                bearing.append(7) if (del_lat > 0) else bearing.append(5)
            
        if ((d_lat > 40.627 and d_lat < 40.655) and (d_long > 73.76 and d_long < 73.796)):
            jfk_dropoff.append(1)
        else:
            jfk_dropoff.append(0)
            
        if ((p_lat > 40.627 and p_lat < 40.655) and (p_long > 73.76 and p_long < 73.796)):
            jfk_pickup.append(1)
        else:
            jfk_pickup.append(0)
            
        if ((d_lat > 40.763 and d_lat < 40.791) and (d_long > 73.856 and d_long < 73.892)):
            laguardia_dropoff.append(1)
        else:
            laguardia_dropoff.append(0)
            
        if ((p_lat > 40.763 and p_lat < 40.791) and (p_long > 73.856 and p_long < 73.892)):
            laguardia_pickup.append(1)
        else:
            laguardia_pickup.append(0)

    df['bearing'] = bearing
    df['jfk_pickup'] = jfk_pickup
    df['jfk_dropoff'] = jfk_dropoff
    df['laguardia_pickup'] = laguardia_pickup
    df['laguardia_dropoff'] = laguardia_dropoff
    
    
    print('Done. One-hot encoding next')
    sys.stdout.flush()


    df_new = df[['pickup_hour', 'pickup_day', 'passenger_count', 'trip_distance', 'pickup_longitude', 
                 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'avg_speed', 'trip_count', 
                 'jfk_pickup', 'jfk_dropoff', 'laguardia_pickup', 'laguardia_dropoff']].copy()


    # One-hot-encode rate code, pickup cluster, dropoff cluster, and bearing (this dataset is very sparse)
    rate_code = list(df.loc[:,'RatecodeID'])
    for x in np.unique(rate_code):
        new_column = list(map(lambda y: 1 if x==y else 0, rate_code))
        df_new[('rate_code_' + str(x))] = new_column

    pickup_cluster = list(df.loc[:,'pickup_cluster'])
    for x in np.unique(pickup_cluster):
        new_column = list(map(lambda y: 1 if x==y else 0, pickup_cluster))
        df_new[('pickup_cluster_' + str(x))] = new_column
        
    dropoff_cluster = list(df.loc[:,'dropoff_cluster'])
    for x in np.unique(dropoff_cluster):
        new_column = list(map(lambda y: 1 if x==y else 0, dropoff_cluster))
        df_new[('dropoff_cluster_' + str(x))] = new_column
        
    bearing = list(df.loc[:,'bearing'])
    for x in np.unique(bearing):
        new_column = list(map(lambda y: 1 if x==y else 0, bearing))
        df_new[('bearing_' + str(x))] = new_column
        
    print('All finished. Saving file next')
    sys.stdout.flush()
    
    filename = 'processed_taxi_data.csv'
    df_new.to_csv(filename)
    print('Done! Saved as %s' % (filename))
