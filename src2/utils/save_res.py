import os 

fairness_header = "*** FAIRNESS AUDIT RESULTS ***"
performance_header = "*** PERFORMANCE METRIC RESULTS"
def save_results(folder, rec_path, mode, res): 
    if not os.path.exists(folder): 
        os.mkdir(folder)
    save_name = folder+'%s-%s.pkl' %(mode, rec_path)
    # if os.path.exists(save_name): 
    #     save_name = folder+'%s-%s-2.pkl' %(mode, rec_path)
    print("***Saving results to:{}***".format(save_name))
    with open(save_name, "w") as f: 
        if mode == 'perf':    
            f.write(performance_header + 'for file:{} \n'.format(rec_path))
        else: 
            f.write(fairness_header + 'for file:{} \n'.format(rec_path))
        for k in res.keys(): 
            f.write("{} : {}\n".format(k, res[k]))

def save_run_performance(folder, res): 
    if not os.path.exists(folder): 
        os.mkdir(folder)
    save_name = os.path.join(folder, 'run_performance.pkl')
    print("***Saving results to:{}***".format(save_name))
    with open(save_name, "w") as f: 
        for k in res.keys(): 
            f.write("{} : {}\n".format(k, res[k]))

def save_run_times(folder, res): 
    if not os.path.exists(folder): 
        os.mkdir(folder)
    save_name = os.path.join(folder, 'runtimes.pkl')
    print("***Saving runtime results to:{}***".format(save_name))
    with open(save_name, "w") as f: 
        for k in res.keys(): 
            f.write("{} : {}\n".format(k, res[k]))
    return 0 