def split_by_pid(df, group_by_val, train_size=.8, val_size = .1, test_size=.1):
    print("***Splitting by Playlist***")
    train_pids, all_test_pids = train_test_split(df[group_by_val].unique(), test_size=test_size+val_size, random_state=1)
    all_test = df[df.pid.isin(all_test_pids)]
    val_pids, test_pids = train_test_split(all_test[group_by_val].unique(), test_size=val_size, random_state=1)
    train = df[df.pid.isin(train_pids)]
    val = df[df.pid.isin(val_pids)]
    test = df[df.pid.isin(test_pids)]
    
    print("***Current Set has {} pids in train, {} pids in val, {} pids in test".format(len(train_pids), len(val_pids), len(test_pids)))
    return list(train.index), list(val.index), list(test.index)