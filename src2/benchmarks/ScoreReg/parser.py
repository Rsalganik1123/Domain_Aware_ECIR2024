from typing import NamedTuple


scratch_path = "/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/LFM/subset/Score_Reg/"
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
    emb_size: int
    reg: str 
    output_path: str 


gamma_spread =  [
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_VANILLA/",  weight=0.0), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_VANILLA/",  weight=0.0), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_VANILLA/",  weight=0.0), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_VANILLA/",  weight=0.0), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_VANILLA/",  weight=0.0), 

    # ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_MACR/",  weight=0.0), 
    # ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_MACR/",  weight=0.0), 
    # ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_MACR/",  weight=0.0), 
    # ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_MACR/",  weight=0.0), 
    # ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_MACR/",  weight=0.0), 

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.1/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.1/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.1/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.1/",  weight=0.1), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.1/",  weight=0.1), 

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.2/",  weight=0.2), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.2/",  weight=0.2), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.2/",  weight=0.2), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.2/",  weight=0.2), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.2/",  weight=0.2), 

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.3/",  weight=0.3), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.3/",  weight=0.3), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.3/",  weight=0.3), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.3/",  weight=0.3), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.3/",  weight=0.3),

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.4/",  weight=0.4), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.4/",  weight=0.4), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.4/",  weight=0.4), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.4/",  weight=0.4), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.4/",  weight=0.4),

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.5/",  weight=0.5), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.5/",  weight=0.5), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.5/",  weight=0.5), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.5/",  weight=0.5), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.5/",  weight=0.5),

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.6/",  weight=0.6), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.6/",  weight=0.6), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.6/",  weight=0.6), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.6/",  weight=0.6), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.6/",  weight=0.6),

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.7/",  weight=0.7), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.7/",  weight=0.7), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.7/",  weight=0.7), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.7/",  weight=0.7), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.7/",  weight=0.7),

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.8/",  weight=0.8), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.8/",  weight=0.8), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.8/",  weight=0.8), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.8/",  weight=0.8), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.8/",  weight=0.8),

    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v1_W_0.9/",  weight=0.9), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v2_W_0.9/",  weight=0.9), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v3_W_0.9/",  weight=0.9), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v4_W_0.9/",  weight=0.9), 
    ArgsforScoreRegRuns(lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1", dataset="mus", burnin="no", emb_size = 10, model='NGCF', reg='no', output_path=scratch_path + "v5_W_0.9/",  weight=0.9),

] 