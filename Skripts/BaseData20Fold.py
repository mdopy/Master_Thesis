import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import yaml
import pickle
from pathlib import Path
from sklearn.preprocessing import label_binarize # TODO: replace sklearn libarys

class DataProcessor:
    def __init__(self, config, **kwargs):

        self.load_config(config)

        self.fold = self.config['data']['fold']
        #self.testpersons = self.config['data']['testpersons']
        self.n_classes = self.config['data']['n_classes']
        self.point_reference = self.config['data']['point_reference']
        self.quat_reference = self.config['data']['quat_reference']
        self.standardize = self.config['data']['standardize']
        self.std_quats = self.config['data']['standardize_quaternions']
        self.window_size = self.config['data']['window_size']
        self.window_stepsize = self.config['data']['window_stepsize']
        self.window_padding = self.config['data']['window_padding']
        self.slide_data_into_window = self.config['data']['slide_data_into_window']
        self.concatgestures = self.config['data']['concatgestures']
        self.n_decisionframes = self.config['data']['n_decisionframes']
        self.Datasetname = 'DHG2016'

        self.seed = self.config['seed']

        self.mean = None
        self.std = None

        # overwrite evtl present
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.sequencesdf = None
        self.window_sets = dict()
        self.window_sets_list = dict()
        self.window_sets_processed = dict()

        self.max_n_frames_per_sequence = None
        self.persons_trval = None
        self.trainpersons = None
        self.validpersons = None


        self.cwd = Path.cwd()
        self.handgestdata_path = (self.cwd / '..' / 'Data' / 'DHG2016' / self.config['data']['handgestdata_name']).resolve()
        self.windowsname = 'Windows_ws' + str(self.window_size) + '_stepsize' + str(self.window_stepsize) + '_classes' + str(self.n_classes)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        with open('config_naming.yaml', 'r') as file:
            self.naming = yaml.safe_load(file)

    def load_handgestdata(self):
        handgestdata_angles = pd.read_pickle(self.handgestdata_path)
        columnames = handgestdata_angles.loc[:, 'timestamp':'Handpoint_Quaternion_s'].columns
        handgestdata_angles[columnames] = handgestdata_angles.loc[:, 'timestamp':'Handpoint_Quaternion_s'].astype('float32')
        handgestdata_angles[['label', 'n_subject', 'n_gesture', 'n_finger', 'n_essai']] = handgestdata_angles.loc[:, ('label', 'n_subject', 'n_gesture', 'n_finger', 'n_essai')].astype('uint8')
        handgestdata_angles['interpolated'] = handgestdata_angles.loc[:, 'interpolated'].astype('bool')
        self.handgestdata_angles = handgestdata_angles.sort_values(['n_subject', 'n_gesture', 'n_finger', 'n_essai', 'timestamp'], axis='index', ascending=True)

    def load_windows(self, folder):
        path = Path(self.cwd / '..' / 'Data' / self.Datasetname / folder / (self.windowsname + '.pkl'))
        with open(path, 'rb') as file:
            self.window_sets = pickle.load(file)
        self.persons_trval = self.window_sets['trval']['n_subject'].unique()

    def save_windowsets(self, folder):
        path = Path(self.cwd / '..' / 'Data' / self.Datasetname / folder)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / (self.windowsname + '.pkl'), 'wb') as file:
            pickle.dump(self.window_sets, file)
        print('Saved at:' + str(path / (self.windowsname + '.pkl')))

    def save_windowsets_processed(self, folder, name):
        path = Path(self.cwd / '..' / 'Data' / self.Datasetname / folder)
        path.mkdir(parents=True, exist_ok=True)
        with open((path / (name + '.pkl')).resolve(), 'wb') as file:
            pickle.dump(self.window_sets_processed, file)
        print('Saved at:' + str((path / (name + '.pkl')).resolve()))


    def save_config(self, folder):
        path = Path(self.cwd / '..' / 'Data' / self.Datasetname / folder)
        path.mkdir(parents=True, exist_ok=True)
        with open((path / 'config.yaml').resolve(), 'w') as file:
            yaml.dump(self.config, file)

    
    # def data2_tv_test(self, data_angles):
    #     # not used in this class
    #     n_subjects = self.handgestdata_angles['n_subject'].max()
    #     persons = list(range(1, n_subjects + 1))
    #     self.persons_trval = [person for person in persons if person not in self.testpersons]

    #     data_angles_test = data_angles.loc[data_angles['n_subject'].isin(self.testpersons), :]
    #     data_angles_trval = data_angles.loc[data_angles['n_subject'].isin(self.persons_trval), :]

    #     data_angles = {
    #         'test': data_angles_test,
    #         'trval': data_angles_trval
    #         }
    #     return data_angles

    def trval2split(self, data_angles):

        print(f'Fold: {self.fold}')
        # self.validpersons = randpersons_tv[self.fold*2:self.fold*2+2]
        self.validpersons = [self.fold,] # 20 fold xval
        self.trainpersons = [person for person in self.persons_trval if person not in self.validpersons]

        data_angles_train = data_angles.loc[data_angles['n_subject'].isin(self.trainpersons), :]
        data_angles_valid = data_angles.loc[data_angles['n_subject'].isin(self.validpersons), :]

        data_angles = {
            'train': data_angles_train,
            'valid': data_angles_valid,
        }
        return data_angles
    


    def resettimestamps(self, handgestangles):
        handgestangles.reset_index(inplace=True, drop=True)
        handgestangles.loc[:, 'timestamp'] = handgestangles.loc[:, 'timestamp'] - handgestangles.loc[0, 'timestamp']
        return handgestangles


    def handangles2windows(self):
        if (self.quat_reference == 'first') and self.std_quats:
            print(f"\033[91m{'Referencing to the first quaternion in the frame is not working properly if quaternions are standardized'}\033[0m")


        rng = np.random.default_rng(self.seed)
        print(f'permuting Samples using seed {self.seed}')

        pd.set_option('future.no_silent_downcasting', True)

        if self.n_classes == 14:
            handgestdata_angles = self.delete_nogesture(self.handgestdata_angles)
        elif self.n_classes == 15:
            handgestdata_angles = self.handgestdata_angles
        else:
            raise ValueError('n_classes must be 14 or 15')

        # 
        handgestdata_angles.loc[:,'frame_idx'] = handgestdata_angles.groupby(['n_subject', 'n_gesture', 'n_finger', 'n_essai']).cumcount()
        #handgestdata_angles['frame_idx'] = handgestdata_angles.loc[:,'frame_idx'].astype('int32')
        handgestdata_angles = handgestdata_angles.astype({'frame_idx': 'int32'})

        self.max_n_frames_per_sequence = max(handgestdata_angles['frame_idx']) + 1

        # data_angles = self.data2_tv_test(handgestdata_angles)
        data_angles = {'trval': handgestdata_angles} # all 20 persons are used

        # if self.config['data']['notest']:
        #     del data_angles['test']
        discarded_sequences = 0
        
        print('Make Windows and apply framreferences...')
        for setname, dataset in data_angles.items():

            #windows = pd.DataFrame()
            
            windows = []
            data_idx = dataset[['n_subject', 'n_gesture', 'n_finger', 'n_essai']].drop_duplicates().values
            dataset.set_index(['n_subject', 'n_gesture', 'n_finger', 'n_essai', 'frame_idx'], drop = False, inplace = True)

            subjectremember = 0
            

            if self.concatgestures:
                for subject in dataset['n_subject'].drop_duplicates().values:

                    window_idx_overall = 0 # index over all windows for the given subject

                    print(f'Processing subject {subject}')

                    sequence = dataset.loc[subject].reset_index(drop = True)
                    sequence['uniqueid'] = sequence.groupby(['n_gesture', 'n_finger', 'n_essai']).ngroup().astype('int32')
                    ids = sequence['uniqueid'].unique()
                    sequence.set_index(['uniqueid'], inplace = True)
                    rng.shuffle(ids, axis = 0)
                    sequence = sequence.loc[ids]
                    sequence.loc[:, 'timestamp'] = (np.arange(sequence.shape[0]) * self.config['infos']['mean_smallest_timediff']).astype('float32')
                    sequence.reset_index(inplace = True, drop = True)

                    window_idx =  np.int32(0)
                    for i in range(0, len(sequence) - self.window_size + 1, self.window_stepsize):
                    # Get the current frame
                        #print(f'Frames: {i} to {i+self.windowsize}')
                        window = sequence.loc[i:i+self.window_size-1, :].copy(deep = True)
                        window.loc[:, 'window_idx'] = window_idx
                        window.loc[:, 'window_idx_overall'] =  window_idx_overall

                        window.loc[:, 'frame_idx'] =  np.arange(len(window))
                        window = self.frameref(window)

                        windows.append(window)
                        window_idx += 1
                        window_idx_overall += 1
                        

            else:
                
                for subject, gesture, finger, essai in data_idx:
                    # window_idx is set to zero for each new sequence
                    # window_idx_global is set to zero for each new subject
                    if subject != subjectremember:
                        print(f'Processing subject {subject}')
                        subjectremember = subject
                        window_idx_overall = np.int32(0) # index over all windows in the given dataset

                    sequence = dataset.loc[(subject, gesture, finger, essai)]

                    if self.window_size >= self.max_n_frames_per_sequence:
                        # no cutting required
                        if (self.window_padding == 'zeros') or (self.window_padding == 'firstframe'): 
                            window = sequence.copy(deep = True)
                            window.loc[:, 'window_idx'] =  np.int32(0)
                            window.loc[:, 'window_idx_overall'] =  window_idx_overall
                            window.loc[:, 'frame_idx'] =  np.arange(len(window))
                            window = self.frameref(window)
                            window_idx_overall += 1

                            windows.append(window)
                        else:
                            raise ValueError('if windowsize exceeds number of frames per sequence, window_padding must be "zeros" or "firstframe"')
                    else:
                        # cuting requiered
                        
                        if self.slide_data_into_window:

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
                            extracted_windows_from_sequence = False
                            # start with full frame (equal to first models)
                            window_idx =  np.int32(0)
                            for i in range(0, len(sequence) - self.window_size + 1, self.window_stepsize):
                            # Get the current frame
                                #print(f'Frames: {i} to {i+self.windowsize}')
                                extracted_windows_from_sequence = True
                                window = sequence.loc[i:i+self.window_size-1, :].copy(deep = True)
                                window.loc[:, 'window_idx'] = window_idx
                                window.loc[:, 'window_idx_overall'] = window_idx_overall
                                window.loc[:, 'frame_idx'] =  np.arange(len(window))
                                window = self.frameref(window)

                                windows.append(window)
                                window_idx_overall += 1
                                window_idx += 1
                            if not extracted_windows_from_sequence:
                                discarded_sequences = discarded_sequences + 1

            self.window_sets[setname] = pd.concat(windows, axis = 0)
            #self.window_sets[setname] = self.window_sets[setname].groupby(['n_subject', 'window_idx_overall']).apply(self.frameref)

        if 'extracted_windows_from_sequence' in locals():
            print(f'{discarded_sequences} Sequences were discarded because they were shorter than the window size')


    def processwindows(self):
        pd.set_option('future.no_silent_downcasting', True)

        window_sets = self.trval2split(self.window_sets['trval'])

        labelmin = window_sets['train']['label'].min() # clean this up so that always 15 labels are used also in one hot coding
        labelmax = window_sets['train']['label'].max()

        print('Standardize and padding of data...')
        if self.standardize:
            self.mean = window_sets['train'].loc[:, 'Thumb_CMC_Spread':'Handpoint_Quaternion_s'].mean()
            self.std = window_sets['train'].loc[:, 'Thumb_CMC_Spread':'Handpoint_Quaternion_s'].std()

            # mean = window_sets['train'].loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'].mean()
            # std = window_sets['train'].loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'].std()

        for setname, dataset in window_sets.items():

            if self.standardize:
                print('Standardize data...' + setname)
                if self.std_quats:
                    dataset.loc[:, 'Thumb_CMC_Spread':'Handpoint_Quaternion_s'] = (dataset.loc[:, 'Thumb_CMC_Spread':'Handpoint_Quaternion_s'] - self.mean) / self.std
                else:
                    dataset.loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'] = (dataset.loc[:, 'Thumb_CMC_Spread':'Handpoint_Z'] - self.mean) / self.std
                

            #dataset.sort_values(['window_idx_overall', 'frame_idx'], axis='index', ascending=True, inplace=True)
            # TODO check if timestamps are maching sortation

            #data_idx = dataset[['n_subject', 'n_gesture', 'n_finger', 'n_essai', 'window_idx']].drop_duplicates().values
            dataset.set_index(['n_subject', 'window_idx_overall'], inplace = True, drop = False)
            global_windowidx = dataset.loc[:, ['n_subject', 'window_idx_overall']].drop_duplicates().values
            #aaaaaa
            n_windows = len(global_windowidx)
            X = np.zeros([n_windows, self.window_size, 27])
            Y = np.zeros([n_windows], dtype=int)
            Infovaiables = {'n_subject':int(), 'n_gesture':int(), 'n_finger':int(), 'n_essai':int(), 'window_idx':int(), 'window_idx_overall':int(), 'frame_idx':int(), 'timestamp':float(), 'label':int(), 'padded':''}
            Info = pd.DataFrame(Infovaiables, index = range(n_windows))

            print("Write data into windows...")
            #for i, (subject, gesture, finger, essai, window_idx) in enumerate(data_idx):
            for i, (subject, window_idx) in enumerate(global_windowidx):

                window_dframe = dataset.loc[(subject, window_idx),:]
                window = window_dframe.loc[:, "Thumb_CMC_Spread":"Handpoint_Quaternion_s"].values

                if self.window_padding == 'firstframe':
                    X[i, :, :] = np.tile(window[0, :], (self.window_size, 1)) # firstframe / oldest frame is used as reference for whole window
                
                T = window.shape[0]
                X[i, -T:, :] = window

                # delete????
                labels = window_dframe.loc[:,'label'].iloc[-self.n_decisionframes:]
                if len(labels.unique()) == 1:
                    Y[i] = labels.iloc[0]
                else:
                    Y[i] = 0
                # end delete
            
                #Y[i] = windowdframe.iloc[-1, :].loc['label'] # newest frame / last frame is used as label
                #Y[i] = windowdframe.iloc[int(self.window_size/2), :].loc['label']  # newest frame / last frame is used as label
                Info.loc[i, :] = window_dframe.iloc[-1, :].loc[['n_subject', 'n_gesture', 'n_finger', 'n_essai', 'window_idx', 'window_idx_overall', 'frame_idx', 'timestamp']]
                Info.loc[i, 'label'] = int(Y[i]) # rewrite / delete normaly not necessary in thos array
                Info.loc[i, 'padded'] = T < self.window_size
            
            Y_oh = label_binarize(Y, classes = np.arange(labelmin, labelmax+1))
            self.window_sets_processed[setname] = {'X': X, 'Y': Y, 'Info': Info, 'Y_oh': Y_oh}


    def delete_nogesture(self, handgestdata_angles):
        handgestdata_angles = handgestdata_angles.loc[handgestdata_angles['n_gesture'] == handgestdata_angles['label']].copy(deep = True)
        return handgestdata_angles

    
    def frameref(self, data_angles_window):
        # applied to a single window of arbitrary size, so can also be a sequence
        data_angles_window.reset_index(drop=True, inplace=True)

        if self.point_reference == 'first':
            data_angles_window.loc[:, self.naming['points']] = (data_angles_window.loc[:, self.naming['points']] - data_angles_window.loc[0, 'Handpoint_X':'Handpoint_Z']).astype('float32')

        elif self.point_reference == 'mean':
            data_angles_window.loc[:, self.naming['points']] = (data_angles_window.loc[:, self.naming['points']] - data_angles_window.loc[:, 'Handpoint_X':'Handpoint_Z'].mean()).astype('float32')

        if self.quat_reference == 'first':
            q_ref = Rotation.from_quat(data_angles_window.loc[0, self.naming['quats']].values)
            for j in range(len(data_angles_window)):
                data_angles_window.loc[0, self.naming['quats']] = (Rotation.from_quat(data_angles_window.loc[j, self.naming['quats']].values) * q_ref.inv()).as_quat()

        # return data_angles_window.set_index(['n_subject', 'n_gesture', 'n_finger', 'n_essai', 'window_idx'], drop = False)
        return data_angles_window



 

