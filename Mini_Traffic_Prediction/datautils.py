

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['all_speed_40','PeMS_road_speed_40','PeMS_speed_43','PeMS_road_speed_43','HuaNan_all_speed_30d', 'HuaNan_road_speed_72']

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    root_path = '../dataset/'
    size = [params.context_points, 0, params.target_points]
    dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
            'root_path': root_path,
            'data_path': 'PeMS_road_speed_40.csv',
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )


    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'PeMS_road_speed_40'
        context_points= 18
        target_points= 3
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
