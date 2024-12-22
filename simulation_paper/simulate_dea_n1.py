#
#
#
import os
import numpy as np
import pandas as pd
import operator
import pickle

def read_data(data_dir, data_name):
    xls = pd.ExcelFile(data_dir + 'Sim-17DMUs-OS.xlsx')
    data_real = xls.parse(0)
    data_range = xls.parse(1)
    data_v = xls.parse(2)
    data_u = xls.parse(3)

    return data_real, data_range, data_v, data_u


def process_data_real(data_real ):
    real_data = []
    for i in range(data_real.shape[1]-1):
        real_data.append(data_real['DMU' + str(i+1)])
    data_real = np.array(real_data).transpose(1,0)
    return data_real



def process_data_u( data_u):
    gamma_u = data_u['Gamma']
    emu_u = data_u['DMU']
    u_u = np.array(data_u['u_1']) # .transpose(1,0) # , data_u['u_2']]


    ushape = u_u.shape[0]
    num_dmus = data_real.shape[1]

        
    u_1 = [u_u[i] for i in range(ushape) if i % 7 == 0]
    u_2 = [u_u[i] for i in range(ushape) if i % 7 == 1]
    u_3 = [u_u[i] for i in range(ushape) if i % 7 == 2]
    u_4 = [u_u[i] for i in range(ushape) if i % 7 == 3]
    u_5 = [u_u[i] for i in range(ushape) if i % 7 == 4]
    u_6 = [u_u[i] for i in range(ushape) if i % 7 == 5]
    u_7 = [u_u[i] for i in range(ushape) if i % 7 == 6]
    u_data = np.array([u_1, u_2, u_3, u_4, u_5, u_6, u_7]) # .transpose(1,0,2)

    return u_data


def process_data_v( data_v):

    gamma_v = data_v['Gamma']
    emu_v = data_v['DMU']
    v_v = np.array([data_v['v_1'], data_v['v_2']]).transpose(1,0)
    vshape = v_v.shape[0]
    # gamma_size = 201
    v_1 = [v_v[i,:] for i in range(vshape) if i % 7 == 0]
    v_2 = [v_v[i,:] for i in range(vshape) if i % 7 == 1]
    v_3 = [v_v[i,:] for i in range(vshape) if i % 7 == 2]
    v_4 = [v_v[i,:] for i in range(vshape) if i % 7 == 3]
    v_5 = [v_v[i,:] for i in range(vshape) if i % 7 == 4]
    v_6 = [v_v[i,:] for i in range(vshape) if i % 7 == 5]
    v_7 = [v_v[i,:] for i in range(vshape) if i % 7 == 6]
    v_data = np.array([v_1, v_2, v_3, v_4, v_5, v_6, v_7]) #.transpose(1,0,2)

    return v_data



def sort_dics(v1, v2):


    dv1 = {}
    dv2 = {}
    for i in range(v1.shape[0]):
        dv1.update({i+1:v1[i]})
        dv2.update({i+1:v2[i]})




    sorted_dv1 = sorted(dv1.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dv2 = sorted(dv2.items(), key=operator.itemgetter(1), reverse=True)


    return sorted_dv1, sorted_dv2


def check_reanking(produced, truth):


    produced = np.array(produced)
    truth = np.array(truth)


    gama_comp = []
    for j1 in range(produced.shape[0]): # this means for each gamma


        tmp_tr = truth[j1]
        tmp_pr = produced[j1]


        tmp_tr, tmp_pr = sort_dics(tmp_tr, tmp_pr)


        pr_dic={}
        tr_dic = {}
        pr_rank_list = []
        tr_rank_list = []
        pr_dmu_list = []
        tr_dmu_list = []


        for i1 in range(len(tmp_pr)):
            if i1 == 0:
                pr_dic.update({tmp_pr[i1][0]: (i1+1, tmp_pr[i1][1]) })
                pr_rank_list.append(i1+1)
                pr_dmu_list.append(tmp_pr[i1][0])


                tr_dic.update({tmp_tr[i1][0]: (i1+1, tmp_tr[i1][1]) })
                tr_rank_list.append(i1+1)
                tr_dmu_list.append(tmp_tr[i1][0])


            else:


                if pr_dic[pr_dmu_list[-1]][1] == tmp_pr[i1][1]:
                    pr_dic.update({tmp_pr[i1][0]: (pr_rank_list[-1], tmp_pr[i1][1]) })
                    pr_rank_list.append(pr_rank_list[-1])
                    pr_dmu_list.append(tmp_pr[i1][0])


                else:
                    pr_dic.update({tmp_pr[i1][0]: (pr_rank_list[-1]+1, tmp_pr[i1][1]) })
                    pr_rank_list.append(pr_rank_list[-1]+1)
                    pr_dmu_list.append(tmp_pr[i1][0])


                if tr_dic[tr_dmu_list[-1]][1] == tmp_tr[i1][1]:
                    tr_dic.update({tmp_tr[i1][0]: (tr_rank_list[-1], tmp_tr[i1][1]) })
                    tr_rank_list.append(tr_rank_list[-1])
                    tr_dmu_list.append(tmp_tr[i1][0])


                else:
                    tr_dic.update({tmp_tr[i1][0]: (tr_rank_list[-1] + 1, tmp_tr[i1][1]) })
                    tr_rank_list.append(tr_rank_list[-1]+1)
                    tr_dmu_list.append(tmp_tr[i1][0])


        # assuming there are only 5 dmus
        dmu_rank_compare = []
        for dmui in range(num_dmus):
            if pr_dic[dmui+1][0] == tr_dic[dmui+1][0]:
                dmu_rank_compare.append(1)
            else:
                dmu_rank_compare.append(0)


        gama_comp.append(dmu_rank_compare)


    return gama_comp




def tfunc(u, v, data_size, data_real):
    sim = []
    gamma_size = len(u[0]) -1 # u.shape[0]


    for k in range(data_size):


        x1 = np.array([np.random.uniform(564403, 621755, gamma_size), np.random.uniform(614371, 669665, gamma_size), \
                       np.random.uniform(762203, 798427, gamma_size), np.random.uniform(862016, 937044, gamma_size), \
                       np.random.uniform(1016898, 1082662, gamma_size), np.random.uniform(1164350, 1267970, gamma_size), \
                       np.random.uniform(1731916, 1816008, gamma_size)
                       ]).transpose(1,0)
        y1 = np.array([np.random.uniform(806549, 866063, gamma_size), np.random.uniform(917507, 985424, gamma_size), \
                       np.random.uniform(1117142, 1195562, gamma_size), np.random.uniform(1206179, 1261031, gamma_size), \
                       np.random.uniform(1381315, 1462543, gamma_size), np.random.uniform(1497679, 1652787, gamma_size), \
                       np.random.uniform(1702249, 1812655, gamma_size)
                       ]).transpose(1,0)
        x2 = np.array([np.random.uniform(674111, 743281, gamma_size), np.random.uniform(685943, 742345, gamma_size), \
                       np.random.uniform(762207, 805677, gamma_size), np.random.uniform(779894, 846496, gamma_size), \
                       np.random.uniform(799714, 877137, gamma_size), np.random.uniform(807172, 889416, gamma_size), \
                       np.random.uniform(818090, 895746, gamma_size)
                       ]).transpose(1,0)
        # y1 = np.array([np.random.uniform(157, 161, gamma_size), np.random.uniform(28, 40, gamma_size)]).transpose(1,0)
        # x2 = np.array([np.random.uniform(4, 12, gamma_size), np.random.uniform(0.16, 0.35, gamma_size)]).transpose(1,0)
        # y2 = np.array([np.random.uniform(157, 198, gamma_size), np.random.uniform(21, 29, gamma_size)]).transpose(1,0)
        # x3 = np.array([np.random.uniform(10, 17, gamma_size), np.random.uniform(.1, 0.7, gamma_size)]).transpose(1,0)
        # y3 = np.array([np.random.uniform(143, 159, gamma_size), np.random.uniform(28, 35, gamma_size)]).transpose(1,0)
        # x4 = np.array([np.random.uniform(12, 15, gamma_size), np.random.uniform(0.21, 0.48, gamma_size)]).transpose(1,0)
        # y4 = np.array([np.random.uniform(138, 144, gamma_size), np.random.uniform(21, 22, gamma_size)]).transpose(1,0)
        # x5 = np.array([np.random.uniform(19, 22, gamma_size), np.random.uniform(.12, 0.19, gamma_size)]).transpose(1,0)
        # y5 = np.array([np.random.uniform(158, 181, gamma_size), np.random.uniform(21, 25, gamma_size)]).transpose(1,0)
        # x = np.array([x1, x2, x3, x4, x5]).transpose(1,0,2)
        # y = np.array([y1, y2, y3, y4, y5]).transpose(1,0,2)
        x = np.array([x1, x2]).transpose(1,0,2)
        y = np.array([y1]).transpose(1,0,2)




        dmu = []
        for iter in range(gamma_size):
            dmus = []
            for jiter in range(len(u)):
                try:
                    uy = np.dot(u[ jiter][iter], y[iter, 0, jiter])
                    vx = np.dot(v[jiter][iter], x[iter, :, jiter])
                except:
                    print('here')
                target = 1. if uy/vx >= 1. else uy/vx
                dmus.append(target)
            dmu.append(dmus)


        dmu = np.array(dmu)
        gama_ranking_compare = check_reanking(dmu, data_real)


        sim.append(gama_ranking_compare)


    return sim


if __name__ == '__main__':


    data_size = 1000 # number of simulations


    sim_result = tfunc(u_data, v_data, data_size, data_real)


    sim_res = np.array(sim_result).transpose(2,1,0)


    total_res = []


    total_sims = sim_res.shape[2]
    for i in range(sim_res.shape[0]): # iterate for dmus
        tmp_res = []
        for j in range(sim_res.shape[1]): # iterate over gamma
            #        for k in range(sim_result.shape[2]):


            tmp_res.append(sim_res[i,j].sum()/total_sims)




        total_res.append(tmp_res)


    final_res = np.array(total_res).transpose(1,0)
    pd.DataFrame(final_res).to_csv('./results/' + str(data_size) + '_newpaper_final.csv')
    average = final_res.sum(axis=1)/num_dmus
    pd.DataFrame(average).to_csv('./results/' + str(data_size) + '_newpaper_average.csv')
    # with open('total_res.npy', 'wb') as f:
    #     pickle.dump(total_res, f)
