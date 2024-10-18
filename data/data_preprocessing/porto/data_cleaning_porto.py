# Recording how data is extracted from the raw train.csv file.

import csv
import numpy as np
import pickle as pk
import argparse
from tqdm import tqdm

# from ICDE22 TMN
porto_ranges = (-7.9, -9.0, 41.8, 40.7)  # lon_max, lon_min, lat_max, lat_min
    
def get_file_handle(file_path):
    csvFile = open(file_path, 'r')
    reader = csv.reader(csvFile)
    return reader
    
def filter_traj_with_missing_records(traj_fh):
    traj_missing = []
    trajectories = []
    for item in tqdm(traj_fh):
        if(traj_fh.line_num == 1):
            continue
        if(item[7] == 'True'):
            traj_missing.append(item[8])
        if(item[7] == 'False'):
            trajectories.append(item[8][2:-2].split('],['))
    print("[Data Preprocessing] Trajectories with missing records: ", len(traj_missing))
    print("[Data Preprocessing] Trajectories without missing records: ", len(trajectories))
    print("[Data Preprocessing] Total trajectories: ", len(traj_missing) + len(trajectories))
    print("[Data Preprocessing] Sample trajectory currently: ", trajectories[0]) # show the trajectory after this step processing
    return trajectories

def filter_traj_by_city_range(trajectories, ranges):
    max_lon, min_lon, max_lat, min_lat = ranges
    print("[Data Preprocessing] Filtering trajectories by city range...")
    print("[Data Preprocessing] Latitude range: ", min_lat, max_lat)
    print("[Data Preprocessing] Longitude range: ", min_lon, max_lon)
    
    target_trajs = []
    for i, trajs in enumerate(trajectories):
        if i%10000 ==0: print(f"[Data Preprocessing] Processed {i} trajs")
        if(len(trajs) > 2): # remove the trajectories with less than 2 points
            Traj = []
            inrange = True
            for coord in trajs:
                tr = coord.split(',')
                if(tr[0] != '' and tr[1] != ''):
                    lon, lat  = float(tr[0]), float(tr[1])
                    if((lat < min_lat) | (lat > max_lat) | (lon < min_lon) | (lon > max_lon)):
                        inrange = False
                    traj_tup = (lon, lat)
                    Traj.append(traj_tup)
            if(inrange != False): target_trajs.append(Traj)
    print("[Data Preprocessing] Trajectories in the target city: ", len(target_trajs))
    print("[Data Preprocessing] Sample city trajectory: ", target_trajs[0])
    
    return target_trajs 


# [Data Preprocessing] Trajectories in the target city:  1665438 for Porto
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="./porto_raw.csv", help="path to the raw trajectory data")
    args = args.parse_args()
    
    trajectories = filter_traj_with_missing_records(get_file_handle(args.data_path))
    city_trajs = filter_traj_by_city_range(trajectories, porto_ranges)
        
    # save the processed data
    pk.dump(city_trajs, open(f"./porto_cleaned.pkl", "wb")) 
    
    
    
    