# %matplotlib inline
import pickle, pandas, os, traceback
import numpy as np
from sklearn.preprocessing import normalize as uv
from opteval import benchmark_func as bf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams.update({'figure.max_open_warning': 0})
np.warnings.filterwarnings('ignore', r"Data with input dtype float128 was converted to float64 by the normalize function.")

n = 25; itr = 1000; x = 10; tt = 1; k = 0.5

args = [1,2,3]

mmz = True
viz = True

os.mkdir('n_{},itr_{},x_{},tt_{},k_{}_Run1'.format(n,itr,x,tt,k))
os.chdir('n_{},itr_{},x_{},tt_{},k_{}_Run1'.format(n,itr,x,tt,k))

def update_plot(i, data, scat, n):
    if len(data[i].T) == 1:
        scat.set_offsets(np.c_[data[i].T[0], np.arange(1,n+1)])
    else:
        scat.set_offsets(np.c_[data[i].T[0], data[i].T[1]])
    return scat
    
def WTFA(func, itr = itr, n = n, x = x, mmz = mmz, tt = tt, k = k, viz = viz):
    
    history = []
    
    z = -1 if mmz else 1
    
    #Auto Adjust n
    n = n*func.variable_num

    positions = np.random.uniform(low = func.min_search_range, 
                                  high = func.max_search_range,
                                  size = (n, func.variable_num))
    
    if viz and func.variable_num < 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([func.min_search_range[0], func.max_search_range[0]])
        if func.variable_num == 2:
            ax.set_ylim([func.min_search_range[1], func.max_search_range[1]])
            scat = ax.scatter(positions.T[0], positions.T[1])
        else:
            ax.set_ylim([0, n+1])
            scat = ax.scatter(positions.T[0], np.arange(1,n+1))
    
    vec = np.random.normal(size = (n, func.variable_num))
    mag = (vec**2).sum(axis=1) ** .5
    velocity = vec/mag[:,None]

    fitness = np.apply_along_axis(func.get_func_val, -1, positions)*z

    for ctr in range(itr):
        maxFit = max(fitness)
        minFit = min(fitness)
        if (maxFit-minFit) == 0:
            break
        newV = lambda i : x*(maxFit-(z*func.get_func_val(i)))/(maxFit-minFit)
        multiplier = np.apply_along_axis(newV, -1, positions)
        velocity = np.multiply(velocity, multiplier[:, np.newaxis], 
                               dtype=np.float128)
        m = ((velocity**2).sum(axis=1)<(k**2)) & ((velocity**2).sum(axis=1)>0)
        if len(velocity[m]) > 0: 
            velocity[m] = uv(velocity[m])*k
        
        positions = positions + (velocity*tt)
        m = (positions>func.max_search_range)|(positions<func.min_search_range)|(positions == np.nan)
        m = (np.any(m, axis=1))
        if len(positions[m]) > 0: 
            positions[m] = np.random.uniform(low = func.min_search_range, 
                                             high = func.max_search_range,
                                             size = (len(positions[m]), 
                                                     func.variable_num))
    
        fitness = np.apply_along_axis(func.get_func_val, -1, positions)*z

        history.append(positions)

        if maxFit < max(fitness):
            #print(-1*max(fitness), velocity[fitness == max(fitness)])
            vec = np.random.normal(size = (func.variable_num))
            mag = (vec**2).sum() ** .5
            velocity[fitness == maxFit] = (vec/mag)*k
        
    fitness = np.apply_along_axis(func.get_func_val, -1, positions)*z

    try:
        if viz and func.variable_num  < 3:
            ani = animation.FuncAnimation(fig, update_plot, frames=range(ctr),
                                          fargs=(history, scat, n))
            mywriter = animation.FFMpegWriter(bitrate=1000,fps=8)
            ani.save(func.func_name+'_'+str(func.variable_num)+'.mp4',writer=mywriter)
    except Exception as err:
        print('Exception in Visualization: '+func.func_name+'_'+str(func.variable_num))
        print(err)
        traceback.print_tb(err.__traceback__)
    
    return {'Computed Solution' : positions[fitness == max(fitness)],
            'Computed Value' : -1*max(fitness)}


params = {'n' : n, 'itr' : itr, 'x' : x , 'tt' : tt, 'k' : k}
results = {}
for num_args in args:
    for funcname in bf.__all__:
        if funcname in bf.__oneArgument__:
            func = eval('bf.{}()'.format(funcname))
        elif funcname in bf.__twoArgument__:
            func = eval('bf.{}({})'.format(funcname, num_args))
        elif funcname in bf.__threeArgument__:
            func = eval('bf.{}({},{})'.format(funcname, num_args, 1))
        else:
            print('Unknown Arguments: '+funcname)
            continue

        key = funcname+'_'+str(num_args)

        try:
            results[key] = WTFA(func)
        except Exception as err:
            print('Exception in Evaluation: '+key)
            print(err)
            traceback.print_tb(err.__traceback__)
            continue

        results[key]['Actual Solution'] = func.optimal_solution
        results[key]['Actual Value'] = func.global_optimum_solution
        results[key]['Name'] = func.func_name
        results[key]['Dimensions'] = func.variable_num

with open('results_dict.pkl', 'wb') as f:
    pickle.dump({'results': results, 'params': params} ,f)
with open('results_DF.pkl', 'wb') as f:
    pickle.dump({'results': pandas.DataFrame(results).T, 'params': params} ,f)

pandas.DataFrame(results).T.to_excel('Results.xlsx')
