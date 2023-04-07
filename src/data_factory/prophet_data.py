from datetime import datetime,timedelta
import pandas as pd

class ProphetData:
     
    def __init__(self): 
        self.test_inputs = None

    def prepare_data(self,data,output):
        # Importing the data
        # df = pd.read_csv(r"./../data/combined revised data 2016-2021.csv", index_col=['Date'],
        #                 parse_dates=True, dayfirst=True)

        self.output = output
        
        df = data
        # TODO Add regressors to prophet model

        # Preprocessing the data
        data = df[[output]].copy()
        data = data.dropna()
        data.columns = ['y']
        data.index.names = ['ds']
        data.reset_index(inplace=True)

        # Train-Test Splitting
        # train_end = datetime(2020, 6, 29)
        # test_end = datetime(2021, 2, 22)

        data_temp = data.set_index('ds')
        split_index = int(len(data_temp)*0.85)
        print(data_temp.head ,split_index, data_temp.index[split_index])
        train_end = data_temp.index[split_index]

        train_data = data_temp[:train_end]

        test_data = data_temp[train_end + timedelta(days=1):]
        train_data.reset_index(inplace=True)
        test_data.reset_index(inplace=True)

        return train_data, test_data


    def holiday_changepoints_build(self):
        lockdown_dates = [datetime(2020, 3, 24),
                        datetime(2020, 4, 15),
                        datetime(2020, 5, 4),
                        datetime(2020, 5, 18),
                        datetime(2020, 6, 1)]
        holidays = pd.DataFrame({'holiday': ['lockdown 1', 'lockdown 2', 'lockdown 3', 'lockdown 4'],
                                'ds': [datetime(2020, 3, 25),
                                        datetime(2020, 4, 15),
                                        datetime(2020, 5, 4),
                                        datetime(2020, 5, 18)],
                                "lower_window": [0, 0, 0, 0],
                                'upper_window': [21, 19, 14, 14]})
                            
        return lockdown_dates, holidays
