# This code is created based on the original code of Neutraj. 
# Source: https://github.com/yaodi833/NeuTraj/blob/master/tools/preprocess.py 
# Author: Chuang Yang, 2024.03.25

import random
import numpy as np
import pickle as pk 
import argparse


# beijing, used for geolife and tdrive 
beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117]
# porto 
porto_lon_range = [-9.0, -7.9]
porto_lat_range = [40.7, 41.8]
# ais 
ais_lat_range = [20.62, 21.72]
ais_lon_range = [-158.51, -157.41]

def shuffle_data(unshuffled_coord_trajs, unshuffled_grid_trajs):
    data = list(zip(unshuffled_coord_trajs, unshuffled_grid_trajs))
    np.random.shuffle(data)
    coord_trajs, grid_trajs = zip(*data)
    coord_trajs = list(coord_trajs)
    grid_trajs = list(grid_trajs)
    return coord_trajs, grid_trajs


class Preprocesser(object):
    def __init__(self, delta = 0.001, lat_range = [1,2], lon_range = [1,2], grid_filter = False):
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.grid_filter = grid_filter 
        self._init_grid_hash_function()
        

    def _init_grid_hash_function(self):
        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x  = self._frange(dXMin, dXMax, self.delta)  
        y  = self._frange(dYMin, dYMax, self.delta)  
        self.grid_size_x = len(x)
        self.grid_size_y = len(y) 
        print("[Preprocessing] Convert the given area into a grid map...")
        print("[Preprocessing] User-defined Longtitude Range: {}, Latitude Range: {}".format(self.lon_range, self.lat_range))
        print("[Preprocessing] Grid Delta: {}".format(self.delta))
        print("[Preprocessing] OUTPUT -> X Index Range: {}-{}, Y Index Range: {}-{}; ".format(0, self.grid_size_x-1, 0, self.grid_size_y-1), "Grid Size: {} * {}".format(self.grid_size_x, self.grid_size_y))
        print("[Preprocessing] OUTPUT -> Gird Index of Right Bottom : {}".format(self.get_grid_index((self.lon_range[1], self.lat_range[1]))))
        print("-"*50)

    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L

    def get_grid_index(self, tuple): 
        """
        Given a coordinate, return the index of the grid where this coordinate is located, 
        return [x,y] and the flattened index.
        Note: x represents grid id for longitude, y represents grid id for latitude.
        """
        eps = 1e-6
        test_tuple = tuple
        test_x,test_y = test_tuple[0]-eps,test_tuple[1]-eps
        x_grid = int ((test_x-self.lon_range[0])/self.delta)
        y_grid = int ((test_y-self.lat_range[0])/self.delta)
        index = y_grid*self.grid_size_x + x_grid 
        return x_grid, y_grid, index

    def traj2grid_seq(self, traj = []):
        """
        Convert a trajectory to a grid index sequence.
        """
        grid_traj = []
        for r in traj:
            lon, lat = r[0], r[1]
            x_grid, y_grid, index = self.get_grid_index((lon, lat))
            grid_traj.append((x_grid, y_grid, index))
        return grid_traj
    
    def batch_traj2grid_seq(self, trajs = []):
        """
        Convert multiple trajectories into grid index sequences, return the original data of the trajectories and the grid index data.
        """
        grid_trajs = []
        for i, traj in enumerate(trajs):
            if i%1000 == 0:
                print("[Preprocessing] Processed {} trajs".format(i))
            grid_traj = self.traj2grid_seq(traj)
            grid_trajs.append(grid_traj)
        return trajs, grid_trajs
    
    def squeeze_traj(self, raw_traj=[], grid_traj = []):
        """
        When multiple consecutive trajectory points fall into the same grid, only the last one is retained. 
        This processing is done in the codes of neutraj, t3s, and tmn, so we follow them. 
        params:
            raw_traj: original trajectory points
            grid_traj: grid index corresponding to the trajectory points
        Return:
            squeezed_traj: GPS trajectory data where only the last point is retained when multiple consecutive points fall into the same grid
            squeezed_grid_traj: Grid index data where only the last point is retained when multiple consecutive points fall into the same grid
        """
        privious = None
        squeezed_traj = []
        for i, (x, y, index) in enumerate(grid_traj):
            if privious==None:
                privious = index 
                squeezed_traj.append(raw_traj[i])
            else:
                if index != privious: 
                    squeezed_traj.append(raw_traj[i])
                    privious = index 
                    
        squeezed_grid_traj = [self.get_grid_index(p) for p in squeezed_traj] 
        return squeezed_traj, squeezed_grid_traj
    
    def batch_traj2squeezed_grid_seq(self, trajs = []):
        squeezed_grid_trajs = [] 
        squeezed_trajs = []
        for i, traj in enumerate(trajs):
            if i%10000 == 0:
                print("[Preprocessing] Processed {} trajs".format(i))
            if len(traj) <= 50: continue # neutraj's tricky, just copied from neutraj's code without modifation
            # set to 40 when using ais, otherwise the avaliable number of data is less than 10000
            grid_traj = self.traj2grid_seq(traj)
            squeezed_traj, squeezed_grid_traj = self.squeeze_traj(raw_traj=traj, grid_traj = grid_traj)
            squeezed_trajs.append(squeezed_traj)
            squeezed_grid_trajs.append(squeezed_grid_traj)    
        return squeezed_trajs, squeezed_grid_trajs     
            


def trajectory_feature_generation(path ='./data/toy_trajs',
                                  lat_range = None,
                                  lon_range = None,
                                  min_length=10, max_length=200, grid_filter = True, data_name= None):
    """
    source: https://github.com/yaodi833/NeuTraj/blob/a64300fcfad318227e314a9e60b48856da671d0a/tools/preprocess.py#L48C9-L48C22
    The `grid_filter` parameter determines whether the trajectory data should undergo a "compression" process. 
    When `grid_filter is set to True, the function will remove consecutive trajectory points that fall within the same grid cell, 
    retaining only the last point of each sequence. 
    If `grid_filter is False, the function will convert the trajectory into grid index sequences without any filtering, 
    preserving every point in the original trajectory, even if consecutive points fall within the same grid cell.
    Following the code of neutraj/t3s/tmn, the grid_filter is set to True by default.
    """
    
    #  read the raw trajectory data
    raw_trajs  =  pk.load(open(path, "rb"))
    print("[Preprocessing] Total Trajs:", len(raw_trajs))
    
    # initailize the grid 
    preprocessor = Preprocesser(delta = 0.001, lat_range = lat_range, lon_range = lon_range, grid_filter=grid_filter)
    
    # convert the trajectory into grid sequence
    trajs, grid_trajs = [], []
    if grid_filter:
        print("[Preprocessing] Convert the trajectory into a squeezed grid sequence...")
        trajs, grid_trajs = preprocessor.batch_traj2squeezed_grid_seq(raw_trajs)
    else:
        print("[Preprocessing] Convert the trajectory into a grid sequence...")
        trajs, grid_trajs = preprocessor.batch_traj2grid_seq(raw_trajs)
    print("[Preprocessing] Grid Mapping Completed!")
    
    assert len(trajs) == len(grid_trajs), "The length of the original trajectory and the grid trajectory should be the same!"
    
    
    # filter the trajectories by length 
    print("[Preprocessing] Start to filter the trajectories by length...") # this follwing the code of neutraj/t3s/tmn
    print("[Preprocessing] Min Length: {}, Max Length: {}".format(min_length, max_length))
    filterd_trajs = []
    filtered_grid_trajs = []
    for i, traj in enumerate(trajs):
        traj_len = len(traj)
        if (traj_len >= min_length) & (traj_len <= max_length):
            filterd_trajs.append(trajs[i])
            filtered_grid_trajs.append(grid_trajs[i])
    print("[Preprocessing] Trajs Num after Length Filtering: ", len(filterd_trajs))
    
    # shuffle the data
    filterd_trajs, filtered_grid_trajs = shuffle_data(filterd_trajs, filtered_grid_trajs)
    # save the processed data
    pk.dump(filterd_trajs, open(f'./{data_name}_coord.pkl', 'wb'))
    pk.dump(filtered_grid_trajs, open(f'./{data_name}_grid.pkl', 'wb'))
    
    avg_traj_len = np.mean([len(traj) for traj in filterd_trajs])
    print(f"[Preprocessing] Average trajectory length: {avg_traj_len}")
    print(f"[Preprocessing] Saved the processed data into ./{data_name}_coord.pkl and ./{data_name}_grid.pkl")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="./porto_cleaned.pkl", help="path to the raw trajectory data")
    args.add_argument("--target_data", type=str, default="porto", help="target city", choices=["porto", "geolife", "tdrive", "ais"])
    args = args.parse_args()
    if args.target_data == "porto":
        trajectory_feature_generation(path=args.data_path, 
                                  lat_range=porto_lat_range, lon_range=porto_lon_range, 
                                  min_length=10, max_length=200, grid_filter=True, data_name=args.target_data)
    if args.target_data == "geolife":
        trajectory_feature_generation(path=args.data_path, 
                                  lat_range=beijing_lat_range, lon_range=beijing_lon_range, 
                                  min_length=10, max_length=200, grid_filter=True, data_name=args.target_data)
    if args.target_data == "tdrive":
        trajectory_feature_generation(path=args.data_path, 
                                  lat_range=beijing_lat_range, lon_range=beijing_lon_range, 
                                  min_length=10, max_length=200, grid_filter=True, data_name=args.target_data)
    if args.target_data == "ais":
        trajectory_feature_generation(path=args.data_path, 
                                  lat_range=ais_lat_range, lon_range=ais_lon_range, 
                                  min_length=10, max_length=200, grid_filter=True, data_name=args.target_data)
