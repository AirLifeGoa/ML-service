from sklearn.preprocessing import StandardScaler
from datetime import datetime,timedelta
import numpy
import pandas as pd
import torch
import numpy as np

#assuming currently only using PM10 as inout feature later will change it to all available data 

class LSTMData:

    def __init__(self,window_size=8):
        self.window_size = window_size
        self.scalar = StandardScaler()

    def prepare_data(self,data,output):

        df = data.copy()
        self.output = output
        data = df[[output]].copy()
        data = data.dropna()
        
        train_end = datetime(2019,12,30)
        test_end = datetime(2021,2,22)

        print(data.head() , data.shape) 
        train_data,self.test_data = self.split_train_test(data)
        print(train_data.shape, self.test_data.shape)
        train_data  = self.scale_data(train_data)

        self.train_data = self.create_inout_sequences(train_data)
        self.full_data = self.create_inout_sequences(self.scale_data(data))

        last_training_point = np.array(self.train_data[-1][0][1:].tolist() + self.train_data[-1][1].tolist()).reshape(-1)
        
        self.test_inputs = list(last_training_point)

        return self.train_data, self.test_data    

    def split_train_test(self,data):
        print(len(data))
        split_index = int(len(data)*0.85)
        train_end = data.index[split_index]

        train_data = data[:train_end]
        test_data = data[train_end + timedelta(days=1):]

        return train_data, test_data
    
    def scale_data(self, data):
        data_transformed = self.scalar.fit_transform(data).reshape(-1, 1)
        data_transformed = torch.FloatTensor(data_transformed)
        return data_transformed
    
    def rescale_data(self, data):
        data_transformed = self.scalar.inverse_transform(data).reshape(-1, 1)
        data_transformed = torch.FloatTensor(data_transformed)
        
        return data_transformed

    "complete code for fetching testinputs"
    def load_test_inputs(self):
        pass

    def create_inout_sequences(self, input_data):
        inout_seq = []
        L = len(input_data)
        for i in range(L-self.window_size):
            data_seq = input_data[i:i+self.window_size]
            data_label = input_data[i+self.window_size:i+self.window_size+1]
            inout_seq.append((data_seq ,data_label))

        return inout_seq