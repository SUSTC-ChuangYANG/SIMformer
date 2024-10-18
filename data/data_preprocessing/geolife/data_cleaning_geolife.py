import os
import pickle as pk
import pandas as pd

data_path = './Geolife Trajectories 1.3/Data'

# from ICDE22 TMN
beijing_ranges = (117, 115.9, 40.7, 39.6) # lon_max, lon_min, lat_max, lat_min

def get_all_trajs_path():
    traj_paths = []
    for i in range(0,182):
        user_data_path = data_path + '/' + str(i).zfill(3)
        if os.path.exists(user_data_path):
            traj_data_path = user_data_path + '/Trajectory'
            if os.path.exists(traj_data_path):
                trajs_name = os.listdir(traj_data_path)
                traj_paths.extend([traj_data_path + '/' + traj_name for traj_name in trajs_name])
    return traj_paths   

def read_traj(traj_path):
    df = pd.read_csv(traj_path, header=None, sep=',', skiprows=6, names=['lat', 'lon', 'zero', 'alt', 'days', 'date', 'time'])
    df["timestamp"] = df["date"] + ' ' + df["time"]
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()
    times = df["timestamp"].to_list()
    trajs = []
    for lat, lon, time in zip(lats, lons, times):
        record = tuple([lon, lat, time])
        trajs.append(record)
    return trajs

def batch_read_traj(traj_paths):
    all_trajs = []
    for i, traj_path in enumerate(traj_paths):
        traj = read_traj(traj_path)
        all_trajs.append(traj)
        if i % 100 == 0:
            print('read {} trajs'.format(i))
    print(f'{len(all_trajs )}done!')
    return all_trajs


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
    lengths = [len(traj) for traj in target_trajs]
    print("[Data Preprocessing] Min length: ", min(lengths))
    print("[Data Preprocessing] Max length: ", max(lengths))
    return target_trajs 
    
# [Data Preprocessing] Trajectories in the target city: 16854 for Geolife
if __name__ == "__main__":
    traj_paths = get_all_trajs_path()
    trajs = batch_read_traj(traj_paths)
    trajs = filter_traj_by_city_range(trajs, beijing_ranges, city_name="beijing")
    print(len(trajs))
    pk.dump(trajs, open('geolife_cleaned.pkl', 'wb'))
    print('done!')