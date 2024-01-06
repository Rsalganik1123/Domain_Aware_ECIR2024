



from typing import NamedTuple

exp_path = 'FILL IN'
scratch_path = f"{exp_path}ablation_10000_100/Benchmark/MACR/"
class ArgsforScoreRegRuns(NamedTuple): 
    lr: float 
    dropout: float 
    batch_size: int 
    epochs: int
    top_k: int 
    factor_num: int 
    num_layers: int 
    num_ng: int 
    test_num_ng: int 
    out: bool 
    gpu: str
    dataset: str 
    sample: str 
    burnin: str 
    weight: float 
    model: str 
    reg: str 
    output_path: str 


macr_runs =  [
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=False, gpu="1", dataset="mus", burnin="no", model='NGCF', reg='no', output_path=scratch_path + "v1/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=False, gpu="1", dataset="mus", burnin="no", model='NGCF', reg='no', output_path=scratch_path + "v2/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=False, gpu="1", dataset="mus", burnin="no", model='NGCF', reg='no', output_path=scratch_path + "v3/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=False, gpu="1", dataset="mus", burnin="no", model='NGCF', reg='no', output_path=scratch_path + "v4/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=False, gpu="1", dataset="mus", burnin="no", model='NGCF', reg='no', output_path=scratch_path + "v5/",  weight=0.1) ] 

from main_graph import main
import os 

task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
print(task_id, type(task_id))
args_for_function = gamma_spread[int(task_id)]
print("LAUNCHING JOB:{} with params:{}".format(task_id, args_for_function))
