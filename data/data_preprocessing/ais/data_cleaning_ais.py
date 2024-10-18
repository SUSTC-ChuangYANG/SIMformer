import pandas as pd 
import numpy as np
import pickle as pk 
import argparse 
import os 


filter_range = {"min_lat": 20.62, "max_lat": 21.72, "min_lon": -158.51, "max_lon": -157.41}

def load_data(filename):
    df = pd.read_csv(filename)
    traj_df = df[["MMSI","BaseDateTime","LAT","LON","SOG"]]
    return traj_df

def get_cv(trip):
    lats = [x[0] for x in trip]
    lons = [x[1] for x in trip]
    cv_lat = np.std(lats) / np.mean(lats)
    cv_lon = np.std(lons) / np.mean(lons)
    return cv_lat, cv_lon

def extract_traj(traj, min_trip_length=10):
    # sort by time 
    traj = traj.sort_values('BaseDateTime')
    # get trip records
    all_records = traj[["LAT","LON","SOG","BaseDateTime"]].values
    acc_zero = 0 
    k = 0 # accumulative zero speed, if the speed is zero for more than k times, then we consider it as a new trajectory
    
    # drop the records with speed 0 at the beginning and end of the trajectory
    start_index = 0
    while all_records[start_index][2] == 0 and start_index < len(all_records) -1 :
        start_index += 1
    end_index = len(all_records) - 1
    while all_records[end_index][2] == 0 and end_index > 0:
        end_index -= 1
    all_records = all_records[start_index:end_index+1]
    if len(all_records) < min_trip_length:
        return [] 
    
    # record the start and end index of each trip 
    start_indices = [0]
    end_indices = [] 
    last_non_zero_index = 0
    for i, record in enumerate(all_records):
        current_speed = record[2]
        if current_speed != 0: 
            if acc_zero > k:  #  if the speed is zero for more than k times, then we consider it as a new trajectory
                # print("finalize one trajectory", last_non_zero_index)
                # print("find one new trajectory", i)
                start_indices.append(i)
                end_indices.append(last_non_zero_index)
            acc_zero = 0    # reset the number of continuous zeros
            last_non_zero_index = i 
            if i == len(all_records) - 1:
                end_indices.append(i)
        else:  # current_speed == 0
            acc_zero += 1
            if i == len(all_records) - 1: 
                end_indices.append(last_non_zero_index)
    if len(start_indices) == 0 or len(end_indices) == 0:
        return []
    else:
        # return the start and end index of each trip
        all_trips = [] 
        for start, end in zip(start_indices, end_indices):
            # print(start, end)
            trip = all_records[start:end+1]
            if len(trip) >= min_trip_length:  
                all_trips.append(trip)
        return all_trips
    
def filter_by_range(trips, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    in_range_trips = []
    for trip in trips:
        in_range = True
        for point in trip:
            if point[0] < min_lat or point[0] > max_lat or point[1] < min_lon or point[1] > max_lon:
                in_range = False 
                break
        if in_range:
            in_range_trips.append(trip)
    return in_range_trips


def dump_trajs(trajs, stat, filename):
    new_trajs = []
    for traj in trajs:
        new_traj = [ (float(p[1]), float(p[0])) for p in traj]
        new_trajs.append(new_traj)
    with open(filename, 'wb') as f:
        pk.dump({"trips": new_trajs, "stat":stat}, f)
        
def extract_trips(traj_df, min_trip_length=10):
    #group by MMSI, sort by time, process each group
    trips_df = traj_df.groupby('MMSI').apply(lambda x: extract_traj(x, min_trip_length)).reset_index()
    trips_df.columns = ['MMSI', 'trips']
    # drop the records without trips 
    trips_df["trip_count"] = trips_df["trips"].apply(lambda x: len(x))
    trips_df = trips_df[trips_df["trip_count"] != 0]
    # some trips are just stop over the sea, so the gps points  just show jittering, we need to drop them
    # drop the trips with cv (coefficient of variation) of lat and lon less than 1e-4
    trips_out = []
    for trips in trips_df["trips"]:
        for trip in trips:
            cv_lat, cv_lon = get_cv(trip)
            if cv_lat > 1e-4 or cv_lon > 1e-4:
                trips_out.append(trip)
    return trips_out


def get_monthly_data(month):
    total_trips_len = []
    monthly_trips = []
    if month in [1, 3, 5, 7, 8, 10, 12]:
        days = 31
    elif month in [4, 6, 9, 11]:
        days = 30
    else:
        days = 28
        
    month = f"{month:02d}"
    for day in range(1, days+1):
        day = f"{day:02d}"
        print(f"----- Start to process day {day} of month {month} ------")
        traj_df = load_data(f"/data/chuang/ais_raw_data/AIS_data_2021_{month}/AIS_2021_{month}_{day}.csv")
        trips = extract_trips(traj_df, min_trip_length=10)
        total_trips_len.append(len(trips))
        print("-> Total number of trajectories is ", len(trips))
        filtered_trips = filter_by_range(trips, min_lat=filter_range["min_lat"], max_lat=filter_range["max_lat"], min_lon=filter_range["min_lon"], max_lon=filter_range["max_lon"])
        if len(filtered_trips) != 0:    
            monthly_trips.extend(filtered_trips)
            print("-> Total number of trajectories in target region is ", len(filtered_trips))
            avg_len = np.mean([len(trip) for trip in filtered_trips])
            print("-> Average length of trajectories in target region is ", avg_len)
        else:
            print("-> No trajectory in target region")
    print(f"-> Total number of trajectories for month {month} is {np.sum(total_trips_len)}")
    print(f"-> Total number of trajectories in target region for month {month} is {len(monthly_trips)}")
    print(f"-> Average len of trajectories in target region for month {month} is {np.mean([len(trips) for trips in monthly_trips])}")
    stat = {"total_trip_count": np.sum(total_trips_len), "target_trip_count": len(monthly_trips), "avg_len": np.mean([len(trips) for trips in monthly_trips])}
    dump_trajs(monthly_trips, stat, f"AIS_2021_{month}_trips.pkl")
    print(f"----- Finish processing month {month} ------")
   
   
def merge_all():
    all_trips = [] 
    overall_trips = 0
    for month in range(1, 13):
        if os.path.exists(f"./AIS_2021_{month:02d}_trips.pkl"):
            with open(f"./AIS_2021_{month:02d}_trips.pkl", "rb") as f:
                data = pk.load(f)
                all_trips.extend(data["trips"])
                overall_trips += data["stat"]["total_trip_count"]
    print("Total trips: ", len(all_trips))
    print("Avg trip length: ", np.mean([len(trip) for trip in all_trips]))
    print("Overall trips: ", overall_trips) 
    pk.dump(all_trips, open("ais_cleaned.pkl", 'wb'))

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--start_month", type=int, default=1)
    argparse.add_argument("--end_month", type=int, default=12)
    args = argparse.parse_args()
    
    for month in range(args.start_month, args.end_month+1):
        get_monthly_data(month)
    merge_all()
    