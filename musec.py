import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import musec_plot
import seaborn


def subject_number():
    subject_path = os.path.join(os.path.expanduser('~'), 'musec/musec-env/musec/musec_nas/musec_data', 'subjects')    
    subject_numbers = os.listdir(subject_path)
    subject_numbers.sort()

    return subject_numbers

def DataFrame(favored_type='all', favored_condition=[1,1], nonfavored_condition=[0,0]):
#     os.chdir(os.path.expanduser('~') + '/musec/musec-env/musec/musec_nas/musec_data')
    musec_df = []
    for i, subject in enumerate(subject_number()):
      musec_df.append(eeg(subject, 'midi', '0').read_score()) 

    merge_df = pd.concat(musec_df)
    
    if favored_type is 'all':
        merge_df = merge_df
    
    elif favored_type is 'Favored':
        merge_df = merge_df.query(
            'like_midi_1 == {} and like_song_1 == {}'.format(favored_condition[0],
                                                             favored_condition[1]))
        
    elif favored_type is 'Non_favored':
        merge_df = merge_df.query(
            'like_midi_1 == {} and like_song_1 == {}'.format(nonfavored_condition[0],
                                                             nonfavored_condition[1]))

    return merge_df

class eeg:
    def __init__(self, subject_number, audio_type, start_time):
        self.subject_number = subject_number  
        self.audio_type = audio_type
        self.start_time = start_time
      
    def emotion_path(self):
        emotion_path = os.path.join(os.path.expanduser('~'), 'musec/musec-env/musec/musec_nas/musec_data',
                                    'subjects',
                                    self.subject_number,
                                    self.subject_number + '_emotion_score')
      
        return emotion_path

    def gtec_folder_path(self):
        gtec_folder_path = os.path.join(os.path.expanduser('~'), 'musec/musec-env/musec/musec_nas/musec_data',
                                        'subject',
                                        self.subject_number, self.subject_number + '_' + 'gtec',
                                        self.subject_number + '_' + 'gtec' + '_' + self.audio_type)
        
        return gtec_folder_path

    def npy_path(self):
        gtec_folder_path = self.gtec_folder_path()
        
        npy_path = os.path.join(gtec_folder_path,
                                self.subject_number + '_' + 'gtec' + '_' +
                                self.audio_type + '_' + 'npy' + '_' + self.start_time + 's')
    
        return npy_path 

    def npy_list(self):
        npy_path = self.npy_path()
        npy_list = os.listdir(npy_path)
        npy_list.sort()

        return npy_list

    def read_score(self):
        emotion_path = self.emotion_path()
        emotion_list = os.listdir(emotion_path)
        emotion_list.sort()
        read_score = pd.read_csv(os.path.join(emotion_path, emotion_list[0]))
        
        return read_score

    def read_eeg(self):
        data_list = []
        data_dict = {}
        
        npy_list = self.npy_list()
        npy_path = self.npy_path()

        for i in range(len(npy_list)):
            data = np.load(os.path.join(npy_path,npy_list[i]))
            data_list.append(data)

        return data_list

    def channels_name(self):
        channels_name = pd.read_csv('eeg_channels', sep=', ' , header=None, engine='python').values[0]

        return channels_name

    def valence_arousal(self):
        read_score = self.read_score()

        if self.audio_type == 'midi':
            print ('\nValence - Arousal of MIDI')
            print ('\nDescription : {emotion_type}_{audio_type}_{low value of score}_{round}')
            print ('\nemotion_type : Valence or Arousal')
            print ('audio_type : MIDI or Song')
            print ('low value of score : m1 = -1, 0 = 0    ; Max value always is 1')   
            print ('round : 1 = Normal round, 2 = Re-check round\n')

            valence_arousal = read_score[['valence_midi_m1_1', 'arousal_midi_m1_1',
                                                 'valence_midi_m1_2', 'arousal_midi_m1_2',
                                                 'valence_midi_0_1', 'arousal_midi_0_1',
                                                 'valence_midi_0_2', 'arousal_midi_0_2']]
            
        elif self.audio_type == 'song':
            print ('\nValence - Arousal of song')
            print ('\nDescription : {emotion_type}_{audio_type}_{low value of score}_{round}')
            print ('\nemotion_type : Valence or Arousal')
            print ('audio_type : MIDI or Song')
            print ('low value of score : m1 = -1, 0 = 0    ; Max value always is 1')   
            print ('round : 1 = Normal round, 2 = Re-check round\n')
            valence_arousal = read_score[['valence_song_m1_1', 'arousal_song_m1_1',
                                                 'valence_song_m1_2', 'arousal_song_m1_2',
                                                 'valence_song_0_1', 'arousal_song_0_1',
                                                 'valence_song_0_2', 'arousal_song_0_2']]

        return valence_arousal

    def split_filename(self):
        filename = []
        npy_list = self.npy_list()
        
        for i in range(len(npy_list)):
            filename.append(self.npy_list()[i].split('_'))

        return filename

    def song_sequence(self):
        song_sequence = []
        npy_list = self.npy_list()

        for i in range(len(npy_list)):
            sequence = self.split_filename()[i][3]
            song_sequence.append(sequence)

        return song_sequence

    def song_number(self):
        song_number = []
        npy_list = self.npy_list()

        for i in range(len(npy_list)):
            number = self.split_filename()[i][4]
            number = number.split('.')
            song_number.append(number[0])

        return song_number

    def to_dict(self):
        dict_keys = ['subject_number', 'audio_type', 'song_sequence', 'song_number', 'eeg_channels_name','eeg_midi', 'eeg_song']	
        dict_values = [self.subject_number, self.audio_type, self.song_sequence(),
                       self.song_number(), self.channels_name(), self.read_eeg(), self.read_eeg()]

        zip_data = zip(dict_keys, dict_values)
        dict_ = dict(zip_data)

        return dict_

  # def print_dict():
     # print('\n')
#     print('EEG Channels name : ', dict_['eeg_channels_name'])
#     print('\n')
    
#     for i in range(0,22):
#         print('Subject number : ', dict_['subject_number'])
#         print('Audio sequence : ', dict_['song_sequence'][i])
#         print('Audio number : ', dict_['song_number'][i])
#         print('MIDI - EEG Channel 1-62 & EOG Channel 63-64 :\n\n', dict_['eeg_midi'][i])
#         print('\n')
#         print('Song - EEG Channel 1-62 & EOG Channel 63-64 :\n\n', dict_['eeg_song'][i])
#         print('\n')
#         print('---------------------------------------------------------------------')
#         print('\n')
