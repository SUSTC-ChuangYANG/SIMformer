# 🏋️ Preprocessing by Yourself

## ⚙️ Requirements
- Python 3.x
- Libraries: `numpy`, `pickle`, `argparse`, `traj_dist`, `scikit-mobility`
- Visualizaiton: `folium` (optional)

## 📈 Trajectory Preprocessing
###  🚕  Porto
You can download the raw porto dataset from [here](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data), put it into `data_preprocessing/porto`, then follow these steps to process the data.

```bash
cd data_preprocessing/porto
# step 1. data cleaning and filter by city range 
python data_cleaning_porto.py # you will get a file name `porto_cleaned.pkl`
# step 2. filter by length and map trajectory to grid  (required by baseline Neutraj and T3S)
python ../feature_construction.py --data_path ./porto_cleaned.pkl --target_data porto
# step 3. random sample trajectories for experiments 
python ../data_sampling.py --k 10000 --target_data porto
```

Above commands read and processe the raw porto taxi data, filtering out the trajectories located in the central part of Porto, Portugal as suggested by the [ICDE 2022 TMN paper](https://ieeexplore.ieee.org/abstract/document/9835456). It should be noted that SIMformer does not need to map trajectories to the grid. The reason for calculating the grid here is because the baselines ``Neutraj`` and ``T3S`` require it.

```json
"Porto": {
    "city_range": {
        "lon_max": -7.9, "lon_min": -9.0,
        "lat_max": 41.8, "lat_min": 40.7, 
    },
    "moving_object": "vehicle",
    "min_length": 10, 
    "max_length": 200,
    "grid_size": "1100*1100", // required by baseline Neutraj and T3S
    "data_length_after_cleaning":  1665438,
    "data_length_after_feature_construction": 599632,
    "num_of_trajs_used_for_exp": 10000
}
```

### 🏃 Geolife
You can download the raw geolife dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367), put it into `data_preprocessing/geolife`, then follow these steps to process the data.

```bash
cd data_preprocessing/geolife
# step 1. data cleaning and filter by city range 
python data_cleaning_geolife.py # you will get a file name `geolife_cleaned.pkl`
# step 2. filter by length and map trajectory to grid (required by baseline Neutraj and T3S)
python ../feature_construction.py --data_path ./geolife_cleaned.pkl --target_data  geolife
# step 3. random sample trajectories for experiments 
python ../data_sampling.py --k 10000 --target_data geolife
```

Above commands read and processe the raw Geolife data, filtering out the trajectories located in the central part of Beijing, China as suggested by the [ICDE 2022 TMN paper](https://ieeexplore.ieee.org/abstract/document/9835456). It should be noted that SIMformer does not need to map trajectories to the grid. The reason for calculating the grid here is because the baselines ``Neutraj`` and ``T3S`` require it.

```json
"Geolife": {
    "city_range": {  
        "lon_max": 117, "lon_min": 115.9,
        "lat_max": 40.7, "lat_min": 39.6, 
    },
    "moving_object": "human",
    "min_length": 10, 
    "max_length": 200,
    "grid_size": "1100*1100",  // required by baseline Neutraj and T3S
    "data_length_after_cleaning":  16854,
    "data_length_after_feature_construction": 11169,
    "num_of_trajs_used_for_exp": 10000  // random sampled 
}
```

### 🚕 T-Drive 
You can download the raw T-Drive dataset from [here](https://www.kaggle.com/datasets/arashnic/tdriver),put it into `data_preprocessing/tdrive`, then follow these steps to process the data.
This may cost **1-2 hours**. 
```bash
cd data_preprocessing/tdrive
# step 1. data cleaning and filter by city range 
python data_cleaning_tdrive.py # you will get a file name `tdrive_cleaned.pkl`
# step 2. filter by length and map trajectory to grid (required by baseline Neutraj and T3S)
python ../feature_construction.py --data_path ./tdrive_cleaned.pkl --target_data tdrive
# step 3. random sample trajectories for experiments 
python ../data_sampling.py --k 10000 --target_data tdrive
```

Above commands read and processe the raw T-Drive data, filtering out the trajectories located in the central part of Beijing, China as suggested by the [ICDE 2022 TMN paper](https://ieeexplore.ieee.org/abstract/document/9835456). It should be noted that SIMformer does not need to map trajectories to the grid. The reason for calculating the grid here is because the baselines ``Neutraj`` and ``T3S`` require it.

```json
"T-Drive": {
    "city_range": {  
        "lon_max": 117, "lon_min": 115.9,
        "lat_max": 40.7, "lat_min": 39.6, 
    },
    "moving_object": "vehicle",
    "min_length": 10, 
    "max_length": 200,
    "grid_size": "1100*1100",  // required by baseline Neutraj and T3S
    "data_length_after_cleaning":  142371,
    "data_length_after_feature_construction": 15314,
    "num_of_trajs_used_for_exp": 10000  // random sampled 

}
```

### 🚢 AIS 
Please download the AIS dataset of 2021 from [here](https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/index.html) first. This data is very large, so it is recommended to download it to a larger storage device. 

- We found that many ships' trajectories are tracked for 24 hours. Therefore, we segmented the original trajectories based on stop points (consecutive very close or identical location points), resulting in `6,434,330` trajectories.
- We further filtered the trajectories located within a small area of the Pacific waters in the United States, ultimately obtaining `25,570` trajectories.
- The code for these two steps is `data_cleaning_ais.py`.

```bash
cd data_preprocessing/ais
# step 1. data preprocessing and filter by range (this may cost a lot of time)
python data_cleaning_ais.py # you will get a file name `ais_cleaned.pkl`
# step 2. filter by length and map trajectory to grid (required by baseline Neutraj and T3S)
python ../feature_construction.py --data_path ./ais_cleaned.pkl --target_data ais
# step 3. random sample trajectories for experiments 
python ../data_sampling.py --k 10000 --target_data ais
```

```json
"AIS": {
    "city_range": {  
        "lon_max": -157.41, "lon_min": -158.51,
        "lat_max": 21.72, "lat_min": 20.62, 
    },
    "moving_object": "vessel",
    "min_length": 10, 
    "max_length": 200,
    "grid_size": "1100*1100",  // required by baseline Neutraj and T3S
    "data_length_after_cleaning":  25570,
    "data_length_after_feature_construction": 10700,
    "num_of_trajs_used_for_exp": 10000  // random sampled 

}
```

### 📝 Summary 
By following these steps, you will obtain four datasets. The grid file is only necessary for models that require grid information.

```json 
"porto": {
   "traj_data": "dataset/porto/porto_coord_10000.pkl",
   "grid_data": "dataset/porto/porto_grid_10000.pkl",
}, 
"geolife": {
   "traj_data": "dataset/geolife/geolife_coord_10000.pkl",
   "grid_data": "dataset/geolife/geolife_grid_10000.pkl",
},
"tdrive": {
   "traj_data": "dataset/tdrive/tdrive_coord_10000.pkl",
   "grid_data": "dataset/tdrive/tdrive_grid_10000.pkl",
},
"ais": {
   "traj_data": "dataset/ais/ais_coord_10000.pkl",
   "grid_data": "dataset/ais/ais_grid_10000.pkl",
}
```

---- 
## 🧮 Distance Matrix Calculation 

This article needs to calculate the distance matrix between trajectories as supervise information for the training and testing model. You can find the code at [utils/distance_calculator/](../../utils/distance_calculator/).

### ⚙️ Usage

```bash
python batch_distance.py --data_path "[your traj data path].pkl"  
                         --measure dtw # set based on the target distance measure 
                         --p_num 30 # The number of processes for parallel computing, used to accelerate calculations.  
                         --out_dir ./ #  output locations 
```

We also provide *Test Suite* to check the correctness of the calculation.

```bash
python batch_distance.py --test 1 
                         --measure dtw # set based on the target distance measure 
                         --dis_matrix_path "[measure]_distance_matrix.pkl" # The distance matrix calculated in last step.
                         --raw_data_path "[your traj data path].pkl"  
```


### 📝 Summary 
By following the above steps, you will obtain the distance matrices for each dataset. 

```json
"porto": {
        "dtw":  "dataset/porto/dis_matrix_dtw_10k.pkl",
        "haus": "dataset/porto/dis_matrix_haus_10k.pkl",
        "fret": "dataset/porto/dis_matrix_dfrec_10k.pkl",
},
"geolife": {
        "dtw":  "dataset/geolife/dis_matrix_dtw_10k.pkl",
        "haus": "dataset/geolife/dis_matrix_haus_10k.pkl",
        "fret": "dataset/geolife/dis_matrix_dfrec_10k.pkl",
},
"tdrive": {
        "dtw":  "dataset/tdrive/dis_matrix_dtw_10k.pkl",
        "haus": "dataset/tdrive/dis_matrix_haus_10k.pkl",
        "fret": "dataset/tdrive/dis_matrix_dfrec_10k.pkl",
},
"ais": {
        "dtw":  "dataset/ais/dis_matrix_dtw_10k.pkl",
        "haus": "dataset/ais/dis_matrix_haus_10k.pkl",
        "fret": "dataset/ais/dis_matrix_dfrec_10k.pkl",
}
```
