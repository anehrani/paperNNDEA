#
#
#\

import numpy as np
import pandas as pd

from data_utils import read_data, process_data_uv
from simulaltor import simulator




if __name__ == '__main__':

    data = 'data/Sim-17DMUs-OS.xlsx' 
    save_dir = 'results/'

    data_size = 100 # number of simulations

    data_real, inputs, outputs, data_v, data_u = read_data(data)

    v_data = process_data_uv( data_v )
    u_data = process_data_uv( data_u )

    psimulator = simulator(data_size, data_real, inputs, outputs, v_data, u_data)

    sim_result = psimulator.run()   
   
    sim_res = np.array(sim_result).transpose(2,1,0)


    total_res = []

    total_sims = sim_res.shape[2]
    for i in range(sim_res.shape[0]): # iterate for dmus
        tmp_res = []
        for j in range(sim_res.shape[1]): # iterate over gamma
            tmp_res.append(sim_res[i,j].sum()/total_sims)

        total_res.append(tmp_res)


    final_res = np.array(total_res).transpose(1,0)
    pd.DataFrame(final_res).to_csv('./results/' + str(data_size) + '_newpaper_final.csv')
    average = final_res.sum(axis=1)/psimulator.num_dmus
    pd.DataFrame(average).to_csv('./results/' + str(data_size) + '_newpaper_average.csv')
