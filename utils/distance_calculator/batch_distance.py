import traj_dist.distance as tdist
import numpy as np
from utils import make_dir, load_data, save_data
from multiprocessing import Pool
import time 
import argparse
from tester import test
import os



def list_wise_distance(out_dir, i, pair_a, pair_b, measure):
    if i%100 == 0: 
        print("Traj-{} start!".format(i))
    dis = tdist.cdist(pair_a, pair_b, metric=measure) 
        
    out_path = "{}/{}.pkl".format(out_dir,i)
    save_data(dis, out_path =out_path)
    if i%100 == 0: 
        print("Finish Traj-{}: [{}, {}], Shape of dis is: {}".format(i, i+1, i+len(pair_b), dis.shape))
  


def batch_distance_computation(data_path, measure, out_dir, p_num):
    """_summary_

    Args:
        data_path (str): path of trajectory data 
        measure (str): , discrete_frechet, hausdorff, dtw, sspd, edr, lcss, erp 等 
        out_dir (str): output directory
        p_num (int): number of cpu cores used for parallel computing
    """
    
    data_name = data_path.split("/")[-1].split(".")[0] 
    out_dir = make_dir("{}_{}".format(data_name, measure), out_dir) # create output directory, name format: data_name_distance_measure
 
    print("Loading data...") 
    trajs = load_data(data_path=data_path)
    trajs = [np.array(traj) for traj in trajs]
    traj_num = len(trajs) 
    
    
    # batch distance computation and record time cost
    print("Start batch distance computation with {} processes...".format(p_num)) 
    start_time = time.time()
    pool = Pool(processes=p_num) 
    for i, traj in enumerate(trajs):
        if i == traj_num -1: 
            break
        pair_a = [trajs[i]] 
        pair_b = trajs[i+1:] 
        try:    
            pool.apply_async(list_wise_distance,(out_dir, i, pair_a, pair_b, measure,)) 
        except Exception as e:  
            print(e)
    pool.close()
    pool.join()
    end_time = time.time() 
    print("Finish batch distance computation, time cost: {}s".format(end_time-start_time))
    
    print("Start batch merging...")
    result_dis = np.zeros((traj_num, traj_num))
    for i in range(traj_num-1):
        file_name =  "{}/{}.pkl".format(out_dir,i)
        i_dis = load_data(file_name)
        result_dis[i, i+1:] = i_dis
        result_dis[i+1:, i] = i_dis
        
    print("Finish batch merging, save data to {}/distance.pkl".format(out_dir))
    
    
    # remove the intermediate files
    for i in range(traj_num-1):
        file_name =  "{}/{}.pkl".format(out_dir,i)
        os.remove(file_name) 
    
   
    save_data(result_dis, "{}/distance.pkl".format(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../../data/dataset/porto/porto_coord_10000.pkl")
    parser.add_argument('--measure', type=str, default="discret_frechet", choices=['dtw','hausdorff','discret_frechet'])
    parser.add_argument('--out_dir', type=str, default="./")
    parser.add_argument('--p_num', type=int, default=30) 
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--dis_matrix_path', type=str, default=None) # test only 
    parser.add_argument('--raw_data_path', type=str, default="../../data/dataset/porto/porto_coord_10000.pkl") # test only 
    
    
    args = parser.parse_args()
    
    if args.test:
        test(dis_matrix_path=args.dis_matrix_path, raw_data_path=args.raw_data_path, measure=args.measure)
    else:
        batch_distance_computation(args.data_path, args.measure, args.out_dir, args.p_num)
    



    



    