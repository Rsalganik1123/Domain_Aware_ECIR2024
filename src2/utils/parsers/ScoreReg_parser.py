from typing import NamedTuple
import glob
import os  
from itertools import product


class ScoreReg_Params(NamedTuple): 
    lr: float 
    dropout: float 
    batch_size: int 
    epochs: int 
    top_k: int 
    factor_num: int 
    num_layers: int 
    num_ng: int 
    