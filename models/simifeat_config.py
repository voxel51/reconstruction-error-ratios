"""
Configuration for the simifeat model.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
| Adapted from https://github.com/UCSC-REAL/SimiFeat
"""

seed = 0
modality = 'image' # image, text, tabular
data_root = '.'
label_sel = 1 # which label/attribute we want to diagnose
train_label_sel = label_sel # 1 for noisy
test_label_sel = train_label_sel

feature_type = 'embedding' 

accuracy = dict(topk = 1, thresh = 0.5)
n_epoch = 10
details = False

train_cfg = dict(
    shuffle = True,
    batch_size = 128,
    num_workers = 1,
)

optimizer = dict(
    name = 'SGD',
    config = dict(
        lr = 0.1    
    )
)


hoc_cfg = dict(
    max_step = 1501, 
    T0 = None, 
    p0 = None, 
    lr = 0.1, 
    num_rounds = 50, 
    sample_size = 35000,
    already_2nn = False,
    device = 'cpu'
)


detect_cfg = dict(
    num_epoch = 21,
    sample_size = 35000,
    k = 10,
    name = 'simifeat',
    method = 'rank'
)
