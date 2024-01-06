import torch 
import os 
import pickle 
import glob 
import re 

def load_ScoreReg_checkpoints(path): 
    checkpoints = glob.glob(path + 'checkpoints/*')
    checkpoints = [c for c in checkpoints if "model20" not in c]
    if len(checkpoints) == 0: 
        return None, -1 
    checkpoints.sort(key=lambda f: int(re.sub('\D', '', f)))
    latest = checkpoints[-1] #sorted(checkpoints)[-1]
    epoch = int(re.findall(r'\d+', latest)[-1])
    
    return latest, epoch 

def find_latest_checkpoint_clean(output_path, mode='u'): 
    checkpoints = glob.glob(output_path+"checkpoints/{}_*.pt".format(mode))
    checkpoints.sort(key=lambda f: int(re.sub('\D', '', f)))
    if len(checkpoints) == 0: 
        print("NO CHECKPOINTS FOUND")
        return None , -1 
    latest = checkpoints[-1] 
    epoch = int(re.findall(r'\d+', latest)[-1])
    return latest, epoch 


def load_checkpoints(model, optimizer, path, mode="u", val=None): 
    p, e = find_latest_checkpoint_clean(path, mode)
    chkpt = torch.load(p)
    try: 
        model.load_state_dict(chkpt['model_state'])
        optimizer.load_state_dict(chkpt['optimizer_state'])
        epoch = chkpt['epoch']
        global_steps = chkpt['global_steps']
        print("pretrained model loaded, epoch:{}, global_steps:{}".format(epoch, global_steps))
        return model,optimizer, epoch, global_steps
    except Exception as e: 
        print("difficulties loading checkpoint:", e)

def find_latest_checkpoint(output_path, mode='u', val=None):
    if val: 
        checkpoints = glob.glob(output_path+"checkpoints/{}_*{}.pt".format(mode, val))
    else: 
        checkpoints = glob.glob(output_path+"checkpoints/{}_*.pt".format(mode))
    checkpoints.sort(key=lambda f: int(re.sub('\D', '', f)))
    latest = checkpoints[-1] #sorted(checkpoints)[-1]
    epoch = int(re.findall(r'\d+', latest)[-1])
    return latest, epoch 

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)    
        
# def paths_from_args(dataset_name, output_path):
#     scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/' 
#     dataset_section = os.path.join(scratch_path, dataset_name)
#     checkpoint_path = os.path.join(dataset_section output_path)
#     embedding_path = ''
#     rec_path = ''

#     return 0 
 