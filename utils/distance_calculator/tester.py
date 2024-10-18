import traj_dist.distance as tdist 
from utils import make_dir, load_data, save_data
import numpy as np 

def test(dis_matrix_path, raw_data_path, metric):
    dis_matrix = load_data(dis_matrix_path) # load the pre-computed distance matrix
    trajs = load_data(data_path=raw_data_path) # load the raw trajectory data
    trajs = [np.array(traj) for traj in trajs] # convert to numpy array
    
    traj_num = len(trajs) 
    k  = 0 
    while k < 100:
        i = np.random.randint(0, traj_num)
        j = np.random.randint(0, traj_num)
        if i == j:
            continue
        else:
            if metric == "discret_frechet":
                dis = tdist.discret_frechet(trajs[i], trajs[j])
            if metric == "hausdorff":
                dis = tdist.hausdorff(trajs[i], trajs[j])
            if metric == "dtw":
                dis = tdist.dtw(trajs[i], trajs[j])
            assert dis == dis_matrix[i][j], "Test failed! The {} distance between traj-{} and traj-{} is {}, but the distance in dis_matrix is {}".format(metric, i, j, dis, dis_matrix[i][j])
            print("Test {} passed, {}/100".format(k, k+1))
        k += 1
    print("All test passed!")
    
    