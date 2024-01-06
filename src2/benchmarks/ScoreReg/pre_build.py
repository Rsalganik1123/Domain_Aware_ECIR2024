import pickle
import pandas as pd 
from tqdm import tqdm
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import numpy as np 
import torch.utils.data as data
import os


def load_all_custom2(p, user_num, item_num):
	""" We load all the three file here to save time in each epoch. """
    
	'''train_data = pd.read_csv(
		'./data/train_df', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32}) '''
	#train_data = pd.read_csv('../data/train_df')    
	train_data = pd.read_csv(p)
	train_data = train_data[['uid', 'sid']]
	train_data['uid'] = train_data['uid'].apply(lambda x : int(x))
	train_data['sid'] = train_data['sid'].apply(lambda x : int(x))    
	train_data.columns = ['user', 'item']
    
	

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
        
	return train_mat

def create_adj_mat(user_num, item_num, R):
        from time import time
        t1 = time()
        adj_mat = sp.dok_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()

        adj_mat[:user_num, user_num:] = R
        adj_mat[user_num:, :user_num] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

def load_all_custom(mus_path, dataset=None):
	""" We load all the three file here to save time in each epoch. """
    
	total_data = pd.read_csv(mus_path + 'total_df')    
	total_data = total_data[['uid', 'sid']]    
	total_data['uid'] = total_data['uid'].apply(lambda x : int(x))
	total_data['sid'] = total_data['sid'].apply(lambda x : int(x))    
	user_num = total_data['uid'].max() + 1
	item_num = total_data['sid'].max() + 1
	del total_data    
    
	train_data = pd.read_csv(mus_path +'train_df')    
	train_data = train_data[['uid', 'sid']]
	train_data['uid'] = train_data['uid'].apply(lambda x : int(x))
	train_data['sid'] = train_data['sid'].apply(lambda x : int(x))    
	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = pd.read_csv(mus_path +'test_df')     
	test_data = test_data[['uid', 'sid']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))
	test_data.columns = ['user', 'item']    
	test_data = test_data.values.tolist()    
    
	val_data = pd.read_csv(mus_path +'val_df')     
	val_data = val_data[['uid', 'sid']]
	val_data['uid'] = val_data['uid'].apply(lambda x : int(x))
	val_data['sid'] = val_data['sid'].apply(lambda x : int(x))
	val_data.columns = ['user', 'item']    
	val_data = val_data.values.tolist()        

	neg_samples_data = pd.read_csv(mus_path +'neg_sample_df')     
	neg_samples_data = neg_samples_data[['uid', 'sid']]
	neg_samples_data['uid'] = neg_samples_data['uid'].apply(lambda x : int(x))
	neg_samples_data['sid'] = neg_samples_data['sid'].apply(lambda x : int(x))
	neg_samples_data.columns = ['user', 'item']    
	neg_samples_data = neg_samples_data.values.tolist()            

	'''
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in test_data:
		test_mat[x[0], x[1]] = 1.0
	'''
        
	total_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		total_mat[x[0], x[1]] = 1.0
	for x in test_data:
		total_mat[x[0], x[1]] = 1.0
	for x in val_data:
		total_mat[x[0], x[1]] = 1.0
	for x in neg_samples_data:
		total_mat[x[0], x[1]] = 1.0
        
	test_data = None # dummy code
	test_mat = None  # dummy code

	return train_data, test_data, user_num, item_num, train_mat, test_mat, total_mat

class BPRData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, total_mat=None, num_ng=0, is_training=None, sample_mode = None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.features2 = None
		self.num_item = num_item
		self.train_mat = train_mat
		self.total_mat = total_mat
		self.num_ng = num_ng
		self.is_training = is_training        
		self.sample_mode = sample_mode        
		# self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		### sample 2 pos, 2 neg        

		if True:
			assert self.is_training, 'no need to sampling when testing'
			self.features_fill = []
			### self.features is train [user, pos item] list
			tmp = pd.DataFrame(self.features)
			tmp.columns = ['uid', 'sid']
            
			### [user pos] -> [user pos1 pos2] 
			### by groupby uid, then shuffling sid
			tmp = tmp.sort_values('uid')
			tmp_list = list(range(tmp.shape[0]))
			rd.shuffle(tmp_list)
			tmp['rng'] = tmp_list
			sid2 = tmp.sort_values(['uid', 'rng']).sid
			tmp['sid2'] = sid2.reset_index().sid
			tmp = tmp[['uid', 'sid', 'sid2']]
			tmp = tmp.sort_index()
			self.features2 = tmp.values.tolist()   
            
			### add neg1, neg2
			### random sample until neg1, neg2 is not from total_mat            
			### note total_mat includes train, val, test, (test neg_samples)            
			for x in self.features2:
				u, pos1, pos2 = x[0], x[1], x[2]
				for t in range(self.num_ng):
					neg1, neg2 = np.random.randint(self.num_item, size = 2)
					while ((u, neg1) in self.total_mat) or ((u, neg2) in self.total_mat):
						neg1, neg2 = np.random.randint(self.num_item, size = 2)
					self.features_fill.append([u, pos1, pos2, neg1, neg2])
            

	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training \
					else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
					self.is_training else self.features
        
		if True:    
			user = features[idx][0]
			pos1 = features[idx][1]
			pos2 = features[idx][2]        
			neg1 = features[idx][3]                    
			neg2 = features[idx][4]                                
			return user, pos1, pos2, neg1, neg2

def part1(train, test, folder_path): 
    print("starting part 1")
    # train = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/complete_data_final_3way_with_emb_and_pos_contig.pkl'
    # test = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set1_clean.pkl'

    

    data = pickle.load(open(train, "rb"))

    train_idx = data['train_indices']
    val_idx = data['val_indices']
    interact = data['df_playlist'].sort_values(['pid', 'tid']).rename(columns={'pid':'uid', 'tid':'sid'})
    
    test_set = pickle.load(open(test, "rb"))
    test_set['pid_for_recs'] = test_set['pid']
    test_set = test_set.sort_values(['pid', 'tid']).rename(columns={'pid':'uid', 'tid':'sid'})

    # mus_real_train_df = interact.iloc[train_idx, :]
    # mus_val_df = interact.iloc[val_idx, :]
    mus_real_train_df = interact.loc[train_idx, :]
    mus_val_df = interact.loc[val_idx, :]
    mus_val_df['pid_for_recs'] = mus_val_df['uid']

    pid_test_dict = dict(zip(test_set.uid.unique(), 
                         list(range(max(interact.uid)+1, max(interact.uid)+1+ len(test_set.uid.unique()))))) 

    test_set['uid'] = test_set['uid'].apply(lambda x: pid_test_dict[x])

    mus_test_df = test_set
    mus_total_data = pd.concat([interact, test_set], ignore_index=True)

    gen_amount = test_set.sort_values(['uid','pos']).loc[test_set.pos < 10]
    mus_real_train_df = pd.concat([mus_real_train_df, gen_amount]) #[['old_pid', 'old_tid', 'sid', 'uid']]
 
    rd.seed(0)

    mus_n_user = len(mus_total_data.uid.unique())
    mus_n_item = len(mus_total_data.sid.unique())

    mus_item_set = set(list(range(mus_n_item)))
    mus_neg_sample_df = pd.DataFrame({'uid' : [], 'sid' : []})
    
    print("generating negative samples")
    for user in tqdm (list(range(mus_n_user))):
        mus_true_set = mus_total_data[mus_total_data['uid'] == user]['sid'].values
        mus_true_set = set(mus_true_set)
        mus_user_neg_samples = mus_item_set - mus_true_set
        mus_user_neg_samples = list(mus_user_neg_samples)
        
        mus_list_len = len(mus_user_neg_samples)
        mus_user_neg_samples = rd.sample(mus_user_neg_samples, 100)
        mus_tmp_neg_sample_df = pd.DataFrame({'uid' : [user]*100, 'sid' : mus_user_neg_samples})
        
        mus_neg_sample_df = pd.concat([mus_neg_sample_df, mus_tmp_neg_sample_df])
 
    mus_neg_sample_df['uid'] = mus_neg_sample_df['uid'].astype(int)
    mus_neg_sample_df['sid'] = mus_neg_sample_df['sid'].astype(int)
    mus_neg_sample_df['type'] = 'neg'

    mus_real_train_df['type'] = 'pos'
    mus_val_df['type'] = 'pos'
    mus_test_df['type'] = 'pos'

    mus_val_df_with_neg = pd.concat([mus_val_df, mus_neg_sample_df])

    mus_test_df_with_neg = pd.concat([mus_test_df, mus_neg_sample_df])


    mus_train_df = pd.concat([mus_real_train_df, mus_val_df]).drop(columns=['type'])

    mus_total_data = mus_total_data.reset_index()#[['uid', 'sid']]
    mus_train_df = mus_train_df.reset_index()#[['uid', 'sid']]
    mus_real_train_df = mus_real_train_df.reset_index()#[['uid', 'sid']]
    mus_val_df = mus_val_df.reset_index()#[['uid', 'sid', 'type']]
    mus_val_df_with_neg = mus_val_df_with_neg.reset_index()#[['uid', 'sid', 'type']]
    mus_test_df_with_neg = mus_test_df_with_neg.reset_index()#[['uid', 'sid', 'type']]

    # assert mus_total_data.shape[0] == mus_real_train_df.shape[0] + mus_val_df.shape[0] + mus_test_df.shape[0]

    # assert mus_val_df_with_neg.shape[0] == mus_val_df.shape[0] + mus_neg_sample_df.shape[0]

    # assert mus_test_df_with_neg.shape[0] == mus_test_df.shape[0] + mus_neg_sample_df.shape[0]
# 
    # folder_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/ScoreReg/from_contig/'
    print("saving")
    mus_total_data.to_csv(folder_path + 'total_df', index = False)
    mus_real_train_df.to_csv(folder_path +'train_df', index = False)
    mus_neg_sample_df.to_csv(folder_path +'neg_sample_df', index = False)

    mus_val_df.to_csv(folder_path +'val_df', index = False)
    mus_val_df_with_neg.to_csv(folder_path +'val_df_with_neg', index = False)

    mus_test_df.to_csv(folder_path +'test_df', index = False)
    mus_test_df_with_neg.to_csv(folder_path +'test_df_with_neg', index = False)

    mus_uid_pop_total = mus_total_data.uid.value_counts().reset_index()
    mus_uid_pop_total.columns = ['uid', 'total_counts']

    mus_sid_pop_total = mus_total_data.sid.value_counts().reset_index()
    mus_sid_pop_total.columns = ['sid', 'total_counts']

    mus_uid_pop_train = mus_train_df.uid.value_counts().reset_index()
    mus_uid_pop_train.columns = ['uid', 'train_counts']

    mus_sid_pop_train = mus_train_df.sid.value_counts().reset_index()
    mus_sid_pop_train.columns = ['sid', 'train_counts']


    mus_uid_pop_total.to_csv(folder_path +'uid_pop_total', index = False)
    mus_sid_pop_total.to_csv(folder_path +'sid_pop_total', index = False)

    mus_uid_pop_train.to_csv(folder_path +'uid_pop_train', index = False)
    mus_sid_pop_train.to_csv(folder_path +'sid_pop_train', index = False)

def part2(folder_path): 
    print("starting part 2")
    # folder_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/ScoreReg/from_contig/'
    p = folder_path + 'train_df'
    mus_train_df = pd.read_csv(p)

    p = folder_path +'total_df'
    mus_total_df = pd.read_csv(p)

    mus_u_num = len(mus_total_df.uid.unique())
    mus_s_num = len(mus_total_df.sid.unique())

    p = folder_path +'train_df'
    
    print("loading sparse mat")
    mus_R = load_all_custom2(p, mus_u_num, mus_s_num)
    print("loading dataset ")
    mus_adj_mat, mus_norm_adj_mat, mus_mean_adj_mat = create_adj_mat(mus_u_num, mus_s_num, mus_R)

    mus_path = folder_path
    print("saving")
    sp.save_npz(mus_path + 's_adj_mat.npz', mus_adj_mat)
    sp.save_npz(mus_path + 's_norm_adj_mat.npz', mus_norm_adj_mat)
    sp.save_npz(mus_path + 's_mean_adj_mat.npz', mus_mean_adj_mat)

def part3(folder_path): 
    rd.seed(0)
    print("starting part 3")
    # mus_path = '/home/mila/r/rebecca.salganik/Projects/Cloned_Repos/popbias/data/final_mus/'
    mus_path = folder_path 
    train_data, test_data, user_num, item_num, train_mat, test_mat, total_mat = load_all_custom(mus_path)
    train_dataset = BPRData(train_data, item_num, train_mat, total_mat, num_ng=1, is_training=True, sample_mode=None)
    train_dataset.ng_sample()
    negative_samples = train_dataset.features_fill
    total_epochs = 30
    num_ng = 3
    print("generating epochs prebuild")
    if not os.path.exists(mus_path+ 'train_samples/'): 
        os.mkdir(mus_path + 'train_samples/')
    for i in tqdm(range(total_epochs)):
        # print(i)
        train_list = []
        for j in range(num_ng):
            train_dataset.ng_sample()
            train_samples = train_dataset.features_fill
            train_list += train_samples
        with open(mus_path+f'train_samples/train_samples_{i}', 'wb') as fp:
            pickle.dump(train_list,fp)

def main(): 
    b = time()
    # folder_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/ScoreReg/from_contig/'
    # train = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/complete_data_final_3way_with_emb_and_pos_contig.pkl'
    # test = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set1_clean.pkl'

    # scratch_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_data_final/LFM_Subset/Full_Size/'

    # scratch_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_data_final/MPD/Filtered/'
    scratch_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data_Final2/LFM_Subset/'
    train= scratch_path + 'train_val.pkl' 
    test = scratch_path + 'test.pkl' 
    folder_path = scratch_path + 'ScoreRegWithGenAmount2/'
    part1(train, test, folder_path)
    part2(folder_path)
    part3(folder_path)
    a = time() 
    print("Built dataset for {} in {}".format(folder_path, (a-b)/60))

main() 