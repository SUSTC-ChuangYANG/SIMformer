base_path = "data/dataset"

DATASET =  {
    "porto": {
        "traj_data": f"{base_path}/porto/porto_coord_10000.pkl",
        'data_range': {'mean_lat': 41.16472526450768, 'mean_lon': -8.620352509402363, 
                       'std_lat': 0.03251428082044641, 'std_lon': 0.03428580177151251},
        "dis_matrix":{
            "dtw": f"{base_path}/porto/dis_matrix_dtw_10k.pkl",
            "haus": f"{base_path}/porto/dis_matrix_haus_10k.pkl",
            "fret": f"{base_path}/porto/dis_matrix_dfrec_10k.pkl",
        }
    },
    "geolife": {
        "traj_data": f"{base_path}/geolife/geolife_coord_10000.pkl",
        'data_range': {'mean_lon': 116.35826234985483, 'mean_lat': 39.9838625668318, 
                       'std_lon': 0.06344818227155774, 'std_lat': 0.05062172693897655},
        "dis_matrix":{
            "dtw":  f"{base_path}/geolife/dis_matrix_dtw_10k.pkl",
            "haus": f"{base_path}/geolife/dis_matrix_haus_10k.pkl",
            "fret": f"{base_path}/geolife/dis_matrix_dfrec_10k.pkl",
        }
    },
    "tdrive":{
        "traj_data": f"{base_path}/tdrive/tdrive_coord_10000.pkl",
        'data_range': {'mean_lon': 116.39907654383447, 'mean_lat': 39.928463888002206, 
                       'std_lon': 0.0919227421510489, 'std_lat': 0.0736084645534069},
        "dis_matrix":{
            "dtw":  f"{base_path}/tdrive/dis_matrix_dtw_10k.pkl",
            "haus": f"{base_path}/tdrive/dis_matrix_haus_10k.pkl",
            "fret": f"{base_path}/tdrive/dis_matrix_dfrec_10k.pkl",
            "erp": f"{base_path}/tdrive/dis_matrix_erp_10k.pkl",
        }
        
    },
    "ais": {
        "traj_data": f"{base_path}/ais/ais_coord_10000.pkl",
        'data_range': {'mean_lon': -157.937091426341,'mean_lat': 21.270019756643304,
                       'std_lon': 0.13135889150131033,'std_lat': 0.07963011966734229},
        "dis_matrix":{
            "dtw":  f"{base_path}/ais/dis_matrix_dtw_10k.pkl",
            "haus": f"{base_path}/ais/dis_matrix_haus_10k.pkl",
            "fret": f"{base_path}/ais/dis_matrix_dfrec_10k.pkl",
            "erp": f"{base_path}/ais/dis_matrix_erp_10k.pkl",
        }
    }
}