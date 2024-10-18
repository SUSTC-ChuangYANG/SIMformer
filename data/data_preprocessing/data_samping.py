import pickle as pk 
import argparse
import numpy as np

def print_stats(trajs):
    lons = []
    lats = []
    for traj in trajs:
        for p in traj:
            lon, lat = p[0], p[1]
            lons.append(lon)
            lats.append(lat)
    lons = np.array(lons)
    lats = np.array(lats)
    mean_lon, mean_lat, std_lon, std_lat = np.mean(lons), np.mean(lats), np.std(lons), np.std(lats)    
    x = {"data_range": {"mean_lon": mean_lon, "mean_lat": mean_lat, "std_lon": std_lon, "std_lat": std_lat}}
    avg_traj_len = np.mean([len(traj) for traj in trajs])
    print(f"[Data Preprocessing] {x}")
    print(f"[Data Preprocessing] Average trajectory length: {avg_traj_len}")
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--k", type=int, default=10000, help="the number of sampled trajs for experiment")
    args.add_argument("--target_data", type=str, default="geolife", help="target city")
    args = args.parse_args()
    path1 = f"./{args.target_data}_coord.pkl"
    path2 = f"./{args.target_data}_grid.pkl"
    # The data generated in feature_construction.py has been shuffled, so we can directly take the first k items.
    coord_trajs = pk.load(open(path1, 'rb')) 
    grid_trajs = pk.load(open(path2, 'rb'))
    print("Before Sampling:")
    print_stats(coord_trajs)
    print("After Sampling:")
    print(f"[Data Preprocessing] Sampled {args.k} trajectories from {args.target_data} dataset")
    print_stats(coord_trajs[:args.k] )    
    pk.dump(coord_trajs[:args.k], open(f"../../dataset/{args.target_data}/{args.target_data}_coord_{args.k}.pkl", "wb"))
    pk.dump(grid_trajs[:args.k], open(f"../../dataset/{args.target_data}/{args.target_data}_grid_{args.k}.pkl", "wb"))