
from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'FilterMGAE'
    MDT.MODEL.PARAM = {
        'seeds'         : [12,34,6,7], # 0.8367±0.0050
        "device"        : -1 ,
        "max_epoch"     : 200,             
        "warmup_steps"  : -1,
        "num_heads"     : 2 ,
        "num_out_heads" : 1,            
        "num_layers"    : 2,
        "num_hidden"    : 256,           
        "residual"      : False,
        "in_drop"       : 0.2,              
        "attn_drop"     : 0.1,   
        "norm"          : None ,
        "lr"            : 0.00015,
        "weight_decay"  : 0.0,                   
        "negative_slope": 0.2,                        
        "activation"    : "prelu",
        "mask_rate"     : 0.5,
        "drop_edge_rate": 0.0,
        "replace_rate"  : 0.0,
        "encoder"       : "gat",
        "decoder"       : "gat",
        "loss_fn"       : "sce",
        "alpha_l"       : 2,
        "optimizer"     : "adam", 
        "max_epoch_f"   : 0,
        "lr_f"          : 0.001,
        "weight_decay_f": 0.0,
        "linear_prob"   : False,
    
        "load_model"    : False,
        "save_model"    : False,
        "use_cfg"       : True,
        "logging"       : False,
        "scheduler"     : False,
        "concat_hidden" : False,

        # ssh add
        'scheme'           : 'fix',                         # fix or sample
        'use_feat_filter'  : False,
        'use_low'          : True,
        'mask_strategy'    : 'distribution',
        'dis_lambda '      : 1.0,
        'mix_type'         : 'degree-dis-csim-certifiedk', # 'mask_strategy'== mix
        'w_degree'         : 0.5,                          # 'mask_strategy'== mix
        'w_dis'            : 0.2,                          # 'mask_strategy'== mix
        'w_csim'           : 0.2,                          # 'mask_strategy'== mix
        'w_certifyk'       : 0.1,                          # 'mask_strategy'== mix
        'keep_scope_centerline'  : 0.05,
        'keep_scope_interval'    : 0.05,  
        'mask_scope_centerline'  : 0.95,
        'mask_scope_interval'    : 0.05,
        'keep_num_ratio'         : 1.0,
        'mask_num_ratio'         : 1.0,     


        # for graph classification
        "pooling"    : "mean",
        "deg4feat"   : False,
        "batch_size" :32,
        "batch_size_f" :256
    }

    return MDT