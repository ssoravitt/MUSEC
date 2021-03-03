import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import musec
from statannot import add_stat_annotation

def plot_valence_arousal(valence_values, arousal_values, median):
    emotion_labels = []
    valence_levels = []
    arousal_levels = []
    
    fig, ax = plt.subplots(1,2, sharex=True, figsize=(13,8))
    plt.grid()

    for i, value in enumerate(valence_values):
        valence = valence_values
        arousal = arousal_values
        if (valence[i] > median) and (arousal[i] > median) : # Happy
            label, valence_level, arousal_level = 'Happy', 'High', 'High'
            happy_plot = ax[0].scatter(valence[i],arousal[i],
                                       color = 'green',
                                       label='Happy')

        elif (valence[i] < median) and (arousal[i] > median) : # Tense
            label, valence_level, arousal_level = 'Tense', 'Low', 'High'
            tense_plot = ax[0].scatter(valence[i],arousal[i],
                                       color = 'red',
                                       label='Tense')

        elif (valence[i] < median) and (arousal[i] < median) : # Sad
            label, valence_level, arousal_level  = 'Sad', 'Low', 'Low'
            sad_plot = ax[0].scatter(valence[i],arousal[i],
                                     color = 'black',
                                     label='Sad')

        elif (valence[i] > median) and (arousal[i] < median) : # Peaceful
            label, valence_level, arousal_level = 'Peaceful', 'High', 'Low'
            peaceful_plot = ax[0].scatter(valence[i],arousal[i],
                                          color = 'blue',
                                          label='Peaceful')

        else :
            print ('NaN')

        emotion_labels.append(label) 
        valence_levels.append(valence_level)
        arousal_levels.append(arousal_level)
    
#     return emotion_labels, valence_levels, arousal_levels

    ax[0].set_xlim(-1.1,1.1)
    ax[0].set_ylim(-1.1,1.1)
    ax[0].axvline(0, ls='-', color='black')
    ax[0].axhline(0, ls='-', color='black')
    ax[0].set_xlabel('Valence')
    ax[0].set_ylabel('Arousal')
    ax[0].set_title('Valence-Arousal')
    
    ax[0].legend(handles=[happy_plot, tense_plot, sad_plot, peaceful_plot], fontsize='large')
    percent_happy = np.round(((emotion_labels.count('Happy')
                               / len(valence_values)) *100), decimals=2)
    
    percent_tense = np.round(((emotion_labels.count('Tense')
                               / len(valence_values)) *100), decimals=2)
    
    percent_sad = np.round(((emotion_labels.count('Sad') 
                             / len(valence_values)) *100), decimals=2)
    
    percent_peaceful = np.round(((emotion_labels.count('Peaceful')
                                  / len(valence_values)) *100), decimals=2)
    
    
    plt.subplot(1,2,2)
    emotions = ['Happy', 'Tense', 'Sad', 'Peaceful']
    percentage_emotions = [percent_happy, percent_tense, percent_sad, percent_peaceful]
    ax_bar = plt.bar(x=range(0,4), height=percentage_emotions, color=['g','r', 'k', 'b'])
    plt.xticks(range(0,4), ('Happy', 'Tense', 'Sad', 'Peaceful'))
    plt.title('Percentage of Emotion')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{} %'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',fontsize=10)
        
    autolabel(ax_bar)

#     print (('Happy : %d songs, %f percent') %(emotion_labels.count('Happy'), percent_happy))
#     print (('Tense : %d songs, %f percent') %(emotion_labels.count('Tense'), percent_tense))
#     print (('Sad : %d songs, %f percent') %(emotion_labels.count('Sad'), percent_sad))
#     print (('Peaceful : %d songs, %f percent') %(emotion_labels.count('Peaceful'), percent_peaceful))
    
    return emotion_labels, valence_levels, arousal_levels


def plot_kind(kind='all'):
    if kind is 'all':
        plot_kind = ['violin', 'box']
        
    elif kind is 'box':
        plot_kind = ['box']

    elif kind is 'violin':
        plot_kind = ['violin']

    return plot_kind


class spending_time:
    
    def __init__(self, df, listen_round):
        self.df = df
        self.listen_round = listen_round

    def DataFrame(self, like_type='all'):
        
        self.like_type = like_type
        
        if self.like_type is 'all':
            self.df = musec.DataFrame()
        
        elif self.like_type is 'Favored':
            self.df = musec.DataFrame('Favored')
            
        elif self.like_type is 'Non_favored':
            self.df = musec.DataFrame('Non_favored')
        

        if self.listen_round == 'normal':
            dict_ = {'midi_time_{}'.format(self.like_type) : (self.df['time_submit_midi'] - self.df['time_stop_midi'])/1000,
                     'song_time_{}'.format(self.like_type) : (self.df['time_submit_song'] - self.df['time_stop_song'])/1000,}

        elif self.listen_round == 'recheck':
            dict_ = {'midi_time_{}'.format(self.like_type) : (self.df['spendtime_stop_midi'] - self.df['spendtime_start_midi'])/1000,
                     'song_time_{}'.format(self.like_type) : (self.df['spendtime_stop_song'] - self.df['spendtime_start_song'])/1000}

        df = pd.DataFrame(data=dict_)
        df_melt = df.melt(var_name='Audio_name', value_name='Time(s)')

        list_audiotype = []

        for i in range(len(df_melt)):
            if df_melt['Audio_name'][i] == 'midi_time_{}'.format(self.like_type):
                list_audiotype.append('MIDI')

            elif df_melt['Audio_name'][i] == 'song_time_{}'.format(self.like_type):
                list_audiotype.append('song'.capitalize())

        df_melt['log(Time)'] = np.log(df_melt['Time(s)'].values)
        df_melt['1/Time'] = 1/(df_melt['Time(s)'].values)
        df_melt['Audio'] = list_audiotype
        df_melt['Status'] = self.like_type.capitalize()

        return df_melt


    def plot(self,  kind='all', **kwargs):

        """
        Condition[0] = MIDI normal round
        Condition[1] = MIDI recheck round
        Condition[2] = Song normal round
        Condition[3] = Song recheck round
        """
        
        self.kind = kind 
        spending_time_concat_df = pd.concat([self.DataFrame('all',),
                                             self.DataFrame('Favored'),
                                             self.DataFrame('Non_favored')])

        for i, k in enumerate(plot_kind(kind)):
            spending_time_plot = sns.catplot(x='Time(s)',
                                             y="Audio", 
                                             col="Status",
                                             kind='{}'.format(k),
                                             palette="Set2",
                                             showmeans=True,
                                             data=spending_time_concat_df)

            spending_time_plot.despine(offset=10)
#             .set(xticks=np.arange(0,101,10))
            

class familiarity:
    def __init__(self, df):
        self.df = df
        
    def DataFrame(self, like_type='all'):
        self.like_type = like_type
        
        if self.like_type is 'all':
            self.df = musec.DataFrame()
        
        elif self.like_type is 'Favored':
            self.df = musec.DataFrame('Favored')
            
        elif self.like_type is 'Non_favored':
            self.df = musec.DataFrame('Non_favored')
        
        df_  = self.df[['familiar_midi_m1_1', 'familiar_midi_m1_2', 'familiar_song_m1_1', 'familiar_song_m1_2']]
#         df_name = self.df[['midi_name']]
        df_melt = df_.melt(var_name='Audio', value_name='Familiarity')
#         df_melt_name = df_name.melt(var_name='MIDI_name', value_name='S' )
#         print (df_melt_name)
        
    
        list_audiotype, list_round, list_audiotype_round = [], [], []
        
        for i in range(len(df_melt)):
            audio_ = df_melt['Audio'][i].split('_')[1]
            round_ = df_melt['Audio'][i].split('_')[3]

            if audio_ == 'midi' and round_ == '1':
                list_audiotype.append('MIDI')
                list_round.append('Normal')
                list_audiotype_round.append('midi_Normal'.capitalize())

            elif audio_ == 'midi' and round_ == '2':
                list_audiotype.append('MIDI')
                list_round.append('Re-check')
                list_audiotype_round.append('midi_Re-check'.capitalize())

            elif audio_ == 'song' and round_ == '1':
                list_audiotype.append('Song')
                list_round.append('Normal')
                list_audiotype_round.append('song_Normal'.capitalize())

            elif audio_ == 'song' and round_ == '2':
                list_audiotype.append('Song')
                list_round.append('Re-check'.capitalize())
                list_audiotype_round.append('song_Re-check'.capitalize())

        df_melt['Audio'] = list_audiotype
        df_melt['Round'] = list_round
        df_melt['Audio_Round'] = list_audiotype_round
        df_melt['Status'] = self.like_type.capitalize()

        return df_melt
    
    def plot(self, kind='all', **kwargs):
    
        familiarity_concat_df = pd.concat([self.DataFrame('all'),
                                           self.DataFrame('Favored'),
                                           self.DataFrame('Non_favored')])
    
        for i, kind in enumerate(plot_kind(kind)):
            familarity_plot = sns.catplot(x="Familiarity",
                                          y='Audio_Round', 
                                          col="Status",
                                          kind='{}'.format(kind),
                                          palette="Set2",
                                          showmeans=True,         
                                          data=familiarity_concat_df)
    
            familarity_plot.despine(offset=10).set(xticks=np.arange(-1,1.1)) #, trim=True)
        
class valence_arousal:
    def __init__(self, df, valence_arousal):
        self.df = df
        self.valence_arousal = valence_arousal
        
    def DataFrame(self, like_type='all'):
        
        self.like_type = like_type
        
        if self.like_type is 'all':
            self.df = musec.DataFrame()
        
        elif self.like_type is 'favored':
            self.df = musec.DataFrame('favored')
            
        elif self.like_type is 'non_favored':
            self.df = musec.DataFrame('non_favored')
            
        list_audio, list_round, list_audiotype_round = [], [], []

        if self.valence_arousal == 'valence':
            df_ = self.df[['valence_midi_m1_1', 'valence_midi_m1_2',
                           'valence_song_m1_1', 'valence_song_m1_2']]
            
            df_melt = df_.melt(var_name='Audio_name', value_name='Valence')

        elif self.valence_arousal == 'arousal':
            df_ = self.df[['arousal_midi_m1_1', 'arousal_midi_m1_2',
                           'arousal_song_m1_1', 'arousal_song_m1_2']]
            
            df_melt = df_.melt(var_name='Audio_name', value_name='Arousal')

        for i in range(len(df_melt)):
            audio_ = df_melt['Audio_name'][i].split('_')[1]
            round_ = df_melt['Audio_name'][i].split('_')[3]

            if audio_ == 'midi' and round_ == '1':
                list_audio.append('MIDI')
                list_round.append('Normal')
                list_audiotype_round.append('midi_Normal'.capitalize())

            elif audio_ == 'midi' and round_ == '2':
                list_audio.append('MIDI')
                list_round.append('Re-check')
                list_audiotype_round.append('midi_Re-check'.capitalize())

            elif audio_ == 'song' and round_ == '1':
                list_audio.append('Song')
                list_round.append('Normal')
                list_audiotype_round.append('song_Normal'.capitalize())

            elif audio_ == 'song' and round_ == '2':
                list_audio.append('Song')
                list_round.append('Re-check'.capitalize())
                list_audiotype_round.append('song_Re-check'.capitalize())

        df_melt['Audio'] = list_audio
        df_melt['Round'] = list_round
        df_melt['Audio_Round'] = list_audiotype_round
        df_melt['Status'] = self.like_type.capitalize()
        df_melt['Valence_Arousal'] = df_melt.keys()[1]
    
        return df_melt
    
    def plot(self, kind='all'):
        va_concat_df = pd.concat([self.DataFrame('all'),
                                  self.DataFrame('favored'),
                                  self.DataFrame('non_favored')])
        
        for i, kind in enumerate(plot_kind(kind)):
            va_plot_violin = sns.catplot(x="{}".format(self.valence_arousal.capitalize()),
                                         y='Audio_Round',
                                         col="Status",
                                         kind='{}'.format(kind),
                                         palette="Set2",
                                         showmeans=True,
                                         data=va_concat_df)

            va_plot_violin.despine(offset=10).set(xticks=np.arange(-1,1.1))
    
