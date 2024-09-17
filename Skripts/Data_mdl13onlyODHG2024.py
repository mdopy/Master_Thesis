import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize # TODO: replace sklearn libarys
import importlib
from BaseData20Fold import DataProcessor

'''
Method has to deliver labels for each single frame
'''



class DataProcessor_mdl13(DataProcessor):

    def __init__(self, config, **kwargs):
        self.reverse_train_valid = False
        super().__init__(config, **kwargs)
        self.Datasetname = 'ODHG2024'

    def load_handgestdata(self):
        path = (self.cwd / '..' / 'Data' / self.Datasetname / self.config['data']['ODHG2024_name']).resolve()
        self.handgestdata_angles = pd.read_pickle(path)

    def handangles2windows(self):
        if (self.quat_reference == 'first') and self.std_quats:
            print(f"\033[91m{'Referencing to the first quaternion in the frame is not working properly if quaternions are standardized'}\033[0m")
        # permutation not necessary

        pd.set_option('future.no_silent_downcasting', True)

        if self.n_classes == 14:
            handgestdata_angles = self.delete_nogesture(self.handgestdata_angles)
        elif self.n_classes == 15:
            handgestdata_angles = self.handgestdata_angles
        else:
            raise ValueError('n_classes must be 14 or 15')

        # ab hier brauch ich nicht 
        #handgestdata_angles['frame_idx'] = handgestdata_angles.groupby(['n_subject', 'n_gesture', 'n_finger', 'n_essai']).cumcount().astype('int32')

        self.max_n_frames_per_sequence = max(handgestdata_angles['frame_idx']) + 1


        if not self.reverse_train_valid:
            data_angles = {'train': handgestdata_angles.loc[handgestdata_angles['n_sequence'].isin(np.arange(1,6)), :], 
                'valid': handgestdata_angles.loc[handgestdata_angles['n_sequence'].isin(np.arange(6,11)), :]}
        else:
            data_angles = {'train': handgestdata_angles.loc[handgestdata_angles['n_sequence'].isin(np.arange(6,11)), :], 
                'valid': handgestdata_angles.loc[handgestdata_angles['n_sequence'].isin(np.arange(1,6)), :]}

        
        print('Make Windows and apply framreferences...')
        for setname, dataset in data_angles.items():

            windows = []
            data_idx = dataset['n_sequence'].unique()
            dataset.set_index(['n_sequence', 'frame_idx'], drop = False, inplace = True)
                
            for n_sequence in data_idx:
                # window_idx is set to zero for each new sequence
                # window_idx_global is set to zero for each new subject
                if n_sequence%5 == 1:
                    print(f'Processing subject {int((n_sequence-1)/5) + 1}')

                sequence = dataset.loc[n_sequence,:]

                if self.window_size >= self.max_n_frames_per_sequence:
                    # no cutting required
                    raise NotImplementedError("No cutting required functionality is not implemented.")
                else:
                    # cuting requiered
                    
                    if self.slide_data_into_window:
                        raise NotImplementedError("slide data into window functionality is not implemented.")

                        if (self.window_padding == 'zeros') or (self.window_padding == 'firstframe'): 
                            window_idx =  np.int32(0)
                            for i in range(0, len(sequence), self.window_stepsize):
                            # Get the current frame
                                if i < self.n_decisionframes:
                                    continue # decision arrea is 4 so at least 4 frames have to be present

                                startframe = np.max([0, i-self.window_size + 1])
                                #print(f'Frames: {startframe} to {i}')
                                window = sequence.loc[startframe:i, :].copy(deep = True)
                                window.loc[:, 'window_idx'] = window_idx
                                window.loc[:, 'window_idx_overall'] =  window_idx_overall
                                window.loc[:, 'frame_idx'] =  np.arange(len(window))
                                window = self.frameref(window)
                        
                                windows.append(window)
                                window_idx_overall += 1
                                window_idx += 1
                        else:
                            raise ValueError('if slide_data_into_window is True, window_padding must be "zeros" or "firstframe"')
                    else:
                        # start with full frame (equal to first models)
                        window_idx =  np.int32(0)
                        for i in range(0, len(sequence) - self.window_size + 1, self.window_stepsize):
                            # Get the current frame
                            #print(f'Frames: {i} to {i+self.windowsize}')
                            window = sequence.loc[i:i+self.window_size-1, :].copy(deep = True)
                            window.loc[:, 'window_idx_overall'] = window_idx
                            window = self.frameref(window)

                            windows.append(window)
                            window_idx += 1

            self.window_sets[setname] = pd.concat(windows, axis = 0)
            #self.window_sets[setname] = self.window_sets[setname].groupby(['n_subject', 'window_idx_overall']).apply(self.frameref)
            # self.window_sets[setname] = self.window_sets[setname].rename(columns={'n_sequence': 'n_subject'}) # TODO big mess todo, find bether solution


    def processwindows(self):
        pd.set_option('future.no_silent_downcasting', True)

        window_sets = self.window_sets

        print('Standardize and padding of data...')
        if self.standardize:
            if (self.mean is None) and (self.std is None):
                self.mean = window_sets['train'].loc[:, self.naming['X_names']].mean()
                self.std = window_sets['train'].loc[:, self.naming['X_names']].std()
                

            # mean = window_sets['train'].loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'].mean()
            # std = window_sets['train'].loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'].std()

        for setname, dataset in window_sets.items():

            if self.standardize:
                print('Standardize data...' + setname)
                # dataset.loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'] = (dataset.loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'] - mean) / std
                dataset.loc[:, self.naming['X_names']] = (dataset.loc[:, self.naming['X_names']] - self.mean) / self.std

            #dataset.sort_values(['window_idx_overall', 'frame_idx'], axis='index', ascending=True, inplace=True)
            # TODO check if timestamps are maching sortation

            #data_idx = dataset[['n_sequence', 'n_gesture', 'n_finger', 'n_essai', 'window_idx']].drop_duplicates().values
            dataset.set_index(['n_sequence', 'window_idx_overall'], inplace = True, drop = False)
            global_windowidx = dataset.loc[:, ['n_sequence', 'window_idx_overall']].drop_duplicates().values
            #aaaaaa
            n_windows = len(global_windowidx)
            x = np.zeros([n_windows, self.window_size, 27])
            Y_allframes = np.zeros([n_windows, self.window_size], dtype=int)

            Infolist = []
            info_dtypes = { 'n_sequence': 'uint8', 'n_gesture': 'uint8', 'n_finger': 'uint8', 'n_essai': 'uint8', 'window_idx': 'uint16',
                            'window_idx_overall': 'uint16', 'frame_idx': 'uint16', 'timestamp': 'float32',  'label': 'int8'}
            info_dtypes_odhg = { 'n_sequence': 'uint8', 'window_idx_overall': 'uint16', 'frame_idx': 'uint16', 'timestamp': 'float32',  'label': 'int8'}


            print("Write data into windows...")
            #for i, (subject, gesture, finger, essai, window_idx) in enumerate(data_idx):
            for i, (subject, window_idx) in enumerate(global_windowidx):

                window_dframe = dataset.loc[(subject, window_idx),:]
                window = window_dframe.loc[:, self.naming['X_names']].values


                if self.window_padding == 'firstframe':
                    x[i, :, :] = np.tile(window[0, :], (self.window_size, 1)) # firstframe / oldest frame is used as reference for whole window
                
                T = window.shape[0]
                x[i, -T:, :] = window

                Y_allframes[i,:] = window_dframe.loc[:, 'label'].values
                # if setname == 'ODHG2024':
                if True:
                    Info_single = window_dframe.loc[:, ['n_sequence', 'window_idx_overall', 'frame_idx', 'label', 'timestamp']].astype(info_dtypes_odhg)
                    Infolist.append(Info_single)
                else:
                    Info_single = window_dframe.loc[:, ['n_sequence', 'n_gesture', 'n_finger', 'n_essai', 'window_idx', 'window_idx_overall', 'frame_idx', 'label', 'timestamp']].astype(info_dtypes)
                    Infolist.append(Info_single)

            Y_allframes_oh = np.reshape(label_binarize(np.reshape(Y_allframes, n_windows* self.window_size), classes = np.arange(0, 15)), (n_windows, self.window_size, 15))
            self.window_sets_processed[setname] = {'X': x, 'Info': pd.concat(Infolist), 'Y_allframes': Y_allframes, 'Y_allframes_oh': Y_allframes_oh}

 

