from jaxrl_m.dataset import Dataset
import os
import numpy as np
import tensorflow as tf
import random
import ml_collections

DATASETS = ["franka_microwave_ds", 
            "franka_hingecabinet_ds",
            "franka_slidecabinet_ds",
            "franka_microwaveopt_ds",
            "franka_microwaveopt_ds_v2",
            "nitish_microwave",
            "microwave_fewshot_reset",
            "microwave_custom_reset",
            "microwave-optonly-reset"]


def get_franka_config():
    return ml_collections.ConfigDict({
        'p_randomgoal': 0.1,
        'p_trajgoal': 0.7,
        'p_currgoal': 0.2,
        'reward_scale': 1.0,
        'reward_shift': -1.0,
        'p_samegoal': 0.5,
        'intent_sametraj': True,
        'terminal': True,
        'max_distance': ml_collections.config_dict.placeholder(int),
    })

def split_dict_contiguous(d, percent_to_keep, ratio=0.8, rews=False):
    terms = d['dones_float']
    stops = np.argwhere(terms).flatten().astype(int)
    partitions = np.split(np.arange(len(terms)), stops + 1)[:-1]
    random.shuffle(partitions) # now shuffled
    n_train = int(ratio * len(partitions))
    train_partitions = partitions[:n_train]
    val_partitions = partitions[n_train:]
    
    # Keep percent_to_keep of the data
    train_partitions = train_partitions[:int(percent_to_keep * len(train_partitions))]
    val_partitions = val_partitions[:int(percent_to_keep * len(val_partitions))]
    
    train_o_indxs = np.concatenate([part[:-1] for part in train_partitions])
    train_n_o_idxs = np.concatenate([part[1:] for part in train_partitions])
    val_o_indxs = np.concatenate([part[:-1] for part in val_partitions])
    val_n_o_idxs = np.concatenate([part[1:] for part in val_partitions])
    
    train_dict = dict()
    val_dict = dict()
    
    train_dict['observations'] = d['observations']['image'][train_o_indxs]
    train_dict['next_observations'] = d['observations']['image'][train_n_o_idxs]
    train_dict['dones_float'] = d['dones_float'][train_n_o_idxs]
    
    val_dict['observations'] = d['observations']['image'][val_o_indxs]
    val_dict['next_observations'] = d['observations']['image'][val_n_o_idxs]
    val_dict['dones_float'] = d['dones_float'][val_n_o_idxs]
    
    if rews:
        train_dict['rewards'] = d['rewards'][train_n_o_idxs]
        val_dict['rewards'] = d['rewards'][val_n_o_idxs]
    
    return train_dict, val_dict

def get_franka_dataset(datasets, percentages, v4=False):
    ROOT = "gs://rail-tpus-nitish-v3/franka/processed_data_resized"
    assert len(datasets) == len(percentages), "Invalid percentages provided"
    assert isinstance(datasets, list), "Invalid datasets provided"
    assert isinstance(percentages, list), "Invalid percentages provided"
    assert len(datasets) > 0, "No datasets provided"
    assert all([ds in DATASETS for ds in datasets]), "Invalid dataset provided"
    if v4:
        rootp = ROOT.replace('nitish-v3', 'nitish-v4')
    else:
        rootp = ROOT
    # Load up each set
    master_train_set = []
    master_val_set = []
    for i, dset in enumerate(datasets):
        fpath = os.path.join(rootp, dset + '.npy')
        with tf.io.gfile.GFile(fpath, 'rb') as file:
            datadict = np.load(file, allow_pickle=True).item()
            file.close()
        # split into train/val
        train_d, val_d = split_dict_contiguous(datadict, percentages[i])
        master_train_set.append(train_d)
        master_val_set.append(val_d)
        
    # join all the sets
    master_train_dict = {k: np.concatenate([d[k] for d in master_train_set], axis=0) for k in master_train_set[0].keys()}
    master_val_dict = {k: np.concatenate([d[k] for d in master_val_set], axis=0) for k in master_val_set[0].keys()}

    
    # create datasets
    master_train_ds = Dataset.create(
        observations=master_train_dict['observations'],
        next_observations=master_train_dict['next_observations'],
        dones_float=master_train_dict['dones_float'],
        actions=np.zeros_like(master_train_dict['dones_float']),
        rewards=np.zeros_like(master_train_dict['dones_float']),
        masks=np.ones_like(master_val_dict['dones_float'])
    )
    master_val_ds = Dataset.create(
        observations=master_val_dict['observations'],
        next_observations=master_val_dict['next_observations'],
        dones_float=master_val_dict['dones_float'],
        actions=np.zeros_like(master_val_dict['dones_float']),
        rewards=np.zeros_like(master_val_dict['dones_float']),
        masks=np.ones_like(master_val_dict['dones_float'])
    )
    return master_train_ds, master_val_ds


def get_franka_dataset_simple(datasets, percentages, v4=False):
    ROOT = "gs://rail-tpus-nitish-v3/franka/processed_data_resized"
    assert len(datasets) == len(percentages), "Invalid percentages provided"
    assert isinstance(datasets, list), "Invalid datasets provided"
    assert isinstance(percentages, list), "Invalid percentages provided"
    assert len(datasets) > 0, "No datasets provided"
    assert all([ds in DATASETS for ds in datasets]), "Invalid dataset provided"
    if v4:
        rootp = ROOT.replace('nitish-v3', 'nitish-v4')
    # Load up each set
    master_train_set = []
    master_val_set = []
    for i, dset in enumerate(datasets):
        fpath = os.path.join(rootp, dset + '.npy')
        with tf.io.gfile.GFile(fpath, 'rb') as file:
            datadict = np.load(file, allow_pickle=True).item()
            file.close()
        # split into train/val
        train_d, val_d = split_dict_contiguous(datadict, percentages[i], rews=True)
        master_train_set.append(train_d)
        master_val_set.append(val_d)
        
    # join all the sets
    master_train_dict = {k: np.concatenate([d[k] for d in master_train_set], axis=0) for k in master_train_set[0].keys()}
    master_val_dict = {k: np.concatenate([d[k] for d in master_val_set], axis=0) for k in master_val_set[0].keys()}
    # create datasets
    master_train_ds = Dataset.create(
        observations=master_train_dict['observations'],
        next_observations=master_train_dict['next_observations'],
        dones_float=master_train_dict['dones_float'],
        actions=np.zeros_like(master_train_dict['dones_float']),
        rewards=master_train_dict['rewards'] - 1,
        masks=np.ones_like(master_train_dict['dones_float'])
    )
    master_val_ds = Dataset.create(
        observations=master_val_dict['observations'],
        next_observations=master_val_dict['next_observations'],
        dones_float=master_val_dict['dones_float'],
        actions=np.zeros_like(master_val_dict['dones_float']),
        rewards=master_val_dict['rewards'] - 1,
        masks=np.ones_like(master_val_dict['dones_float'])
    )
    return master_train_ds, master_val_ds