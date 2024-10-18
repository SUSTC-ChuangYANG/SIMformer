# The T-Drive dataset continuously records taxi movements over one week, with each trajectory containing over 1,400 points on average. 
# To reduce the high computational cost of calculating ground truth distances and improve training efficiency, 
# - We first segmented the original trajectories into trips according to stay points
# - Subsequently, we processed the two datasets following the same procedures as Geolife and Porto. 
# -- which means trajectories that are the too long, too short or too far from the central city area are removed.
# Author: Chuang Yang, 2024.10.18

import pandas as pd 
import skmob
from skmob.preprocessing import detection
import pickle as pk 



beijing_ranges = (117, 115.9, 40.7, 39.6) # lon_max, lon_min, lat_max, lat_min

def filter_traj_by_city_range(trajectories, ranges, city_name="beijing"):
    max_lon, min_lon, max_lat, min_lat = ranges
    print(f"[Data Preprocessing] Filtering trajectories by {city_name} city range...")
    print("[Data Preprocessing] Latitude range: ", min_lat, max_lat)
    print("[Data Preprocessing] Longitude range: ", min_lon, max_lon)
    
    target_trajs = []
    for i, traj in enumerate(trajectories):
        if i%1000 ==0: print(f"[Data Preprocessing] Processed {i} trajs")
        if(len(traj) > 2): # remove the trajectories with less than 2 points
            inrange = True
            new_traj = []
            for coord in traj:
                lon, lat = coord[0], coord[1]
                if((lat < min_lat) | (lat > max_lat) | (lon < min_lon) | (lon > max_lon)):
                    inrange = False
                new_traj.append((lon, lat))
            if inrange: target_trajs.append(new_traj)
    print("[Data Preprocessing] Trajectories in the target city: ", len(target_trajs))
    print("[Data Preprocessing] Sample city trajectory: ", target_trajs[0])
    # print the length of the trajectories
    lengths = [len(traj) for traj in target_trajs]
    print("[Data Preprocessing] Min length: ", min(lengths))
    print("[Data Preprocessing] Max length: ", max(lengths))
    return target_trajs 


def full_traj2trip(df,minutes_for_a_stop=5.0,spatial_radius_km=0.2):
    """
    input a full trajectory, output a list of trips, split by the stay points
    """
    # step 1. detect the stay points 
    trips = []
    tdf = skmob.TrajDataFrame(df, latitude='latitude',longitude="longitude",datetime='time', user_id='taxi_id')
    stdf = detection.stay_locations(tdf, stop_radius_factor=0.5, minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=spatial_radius_km, leaving_time=True)
    if len(stdf) ==0 : # zero means this traj does not have stay points, only one trip 
        total_time = tdf['datetime'].max() - tdf['datetime'].min()
        trips.append({"traj":tdf[['lng', 'lat']].values, "time":total_time.total_seconds()/60})
    else:
    # step 2. extract the start time and end time of each trip
        trip_start_time = tdf.datetime.min()
        trip_ranges = [] 
        for idx, row in stdf.iterrows():
            trip_end = row['datetime'] # stay start 
            trip_start = row['leaving_datetime'] # stay end 
            trip_ranges.append((trip_start_time, trip_end))
            trip_start_time = trip_start
        trip_ranges.append((trip_start_time, tdf.datetime.max()))
        trips = []
        for trip_range in trip_ranges:
            trip = tdf[(tdf.datetime >= trip_range[0]) & (tdf.datetime < trip_range[1])]
            if len(trip) > 0:
                trip = trip.sort_values(by='datetime')
                total_time = trip['datetime'].max() - trip['datetime'].min()
                trips.append({"traj":trip[['lng', 'lat']].values, "time":total_time.total_seconds()/60})
    # further filter, drop the trip with length less than 10 in advance
    trips = [trip for trip in trips if len(trip['traj'])>=10]
    return trips

def trip_extraction(minutes_for_a_stop=5.0, spatial_radius_km=0.1):
    all_trips = []
    for i in range(1, 10358):
        taxi_df = None
        try:
            taxi_df = pd.read_csv(f'taxi_log_2008_by_id/{i}.txt',header=None)
        except:
            print(f'error in {i}, empty file')
            continue
        taxi_df.columns = ['taxi_id', 'time', 'longitude', 'latitude']
        trips = full_traj2trip(taxi_df, minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=spatial_radius_km)
        all_trips.extend(trips)
        if i%100 == 0:
            print(f'processed {i} files')
            
    tlen = [ len(trip['traj']) for trip in all_trips]
    print(f'average trip length is {sum(tlen)/len(tlen)}')
    print(f'total trips {len(all_trips)}')
    return all_trips





if __name__ == "__main__":
    # step 1. extract the trips
    # extraction rules -> minutes_for_a_stop is 5 mins, spatial_radius is 100m
    all_trips = trip_extraction(minutes_for_a_stop=5.0, spatial_radius_km=0.1)
    trips = [trip["traj"] for trip in all_trips]
    # step 2. filter the trips by the city range 
    # [Preprocessing] Min Length: 10, Max Length: 200
    filtered_trips = filter_traj_by_city_range(trips, beijing_ranges)
    pk.dump(filtered_trips, open('tdrive_cleaned.pkl', 'wb'))
    print('all trips saved')