# import torch 
import os 
import numpy as np 

# class EarlyStopper: 
#     def __init__(self, checkpoint_path, tolerance=5, delta=0.0, saving_checkpoints=False): 
#         self.tolerance = tolerance
#         self.delta = delta
#         self.counter = 0  
#         self.best_val = None  
#         self.checkpoint_path = checkpoint_path
#         self.saving_checkpoints = saving_checkpoints 
#     def stop_check(self, loss, model, optimizer): 
#         if loss > self.loss_records[-1]: 
#             self.counter += 1
#         if self.counter > self.tolerance: 
#             return True 
#         self.loss_records.append(loss)
#         return False 

#     def __call__(self, val_loss, global_steps, epoch, model, optimizer, loss_records = None, valid_loss_records=None): 
#         score = -val_loss
#         if self.best_val is None:
#             self.best_val = score
#             if self.saving_checkpoints: 
#                 self.save_checkpoint(val_loss,  epoch, global_steps, model, optimizer)
           
#         elif score < self.best_val + self.delta: 
#             self.counter += 1 
#             if self.counter >= self.tolerance: 
#                 return True 
#         else: 
#             self.best_val = score 
#             if self.saving_checkpoints: 
#                 self.save_checkpoint(val_loss,  epoch, global_steps, model, optimizer)
#             self.counter = 0 
#         return False 

#     def save_checkpoint(self, val_loss, epoch, global_steps, model, optimizer, loss_records = None, valid_loss_records=None): 
#         save_state = {
#             'global_steps': global_steps,
#             "epoch": epoch + 1,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#         }
#         backup_fpath = os.path.join(self.checkpoint_path, "model_bak_%06d.pt" % (epoch,))
#         print("saving checkpoint for epoch:{} to:{}".format(epoch, backup_fpath))
#         torch.save(save_state, backup_fpath)


class EarlyStopper: 
    def __init__(self, tolerance, delta): 
        self.tolerance = tolerance
        self.delta = delta
        self.counter = 0  
        self.lowest_loss = np.finfo(float).max
        
    
    def __reset__(self): 
        self.lowest_loss = np.finfo(float).max
        self.counter = 0  

    def __call__(self, val_loss, global_steps, epoch, model, optimizer, mode='utility'): 
        if mode == 'fairness': 
            val_loss = 1-val_loss 
        if val_loss > self.lowest_loss + self.delta:
            self.counter += 1 
            if self.counter >= self.tolerance: 
                return True 
        else:   
            self.lowest_loss = val_loss 
            return False 

    
