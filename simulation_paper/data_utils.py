
import numpy as np
import pandas as pd

def read_data(data_dir):
    xls = pd.ExcelFile(data_dir )
    data_real = xls.parse(0)
    inputs = xls.parse(1)
    outputs = xls.parse(2)
    data_v = xls.parse(3)
    data_u = xls.parse(4)

    return data_real, inputs, outputs, data_v, data_u


def process_data_real(data_real ):
    real_data = []
    for i in range(data_real.shape[1]-1):
        real_data.append(data_real['DMU' + str(i+1)])
    data_real = np.array(real_data).transpose(1,0)

    num_dmus = data_real.shape[1]
    return data_real, num_dmus


def process_data_uv( data_uv ):

    uv_data = {}
    for line in data_uv.iterrows():
        gamma = round(line[1][0],2)
        dmu = int(line[1][1])
        if not dmu in uv_data:
            uv_data[dmu] = {}
        uv_data[dmu][gamma] = line[1][2:].values    
    

    return uv_data
