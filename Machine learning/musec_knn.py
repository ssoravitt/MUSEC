import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler

path_csv = os.getcwd() + '/dataframe/df_frontal_svm.csv'
data = pd.read_csv(path_csv)
data = data.drop(columns=['Unnamed: 0'])

def normalize_list(list_normal):
    max_value, min_value = max(list_normal), min(list_normal)
    list_normalize = [(list_normal[i] - min_value) / (max_value - min_value) for i, value in enumerate(list_normal)]

    return list_normalize
    
def df_for_ml():
    audio_hemisphere = ['Melody | Frontal_left', 'Melody | Frontal_right', 'Song | Frontal_left', 'Song | Frontal_right']
    band = ['Theta', 'Alpha', 'Beta']
    psd, familiarity, times = [], [], []
    for i, a in enumerate(audio_hemisphere):
        for j,b in enumerate(band):
            data_psd = data[(data.Hemisphere_audio == a) & (data.Band == b)]
            psd.append(data_psd['psd_mean'].values) #psd_band_hemisphere_audio
    
    personality = data_psd[['Subject', 'MIDI_name', 'Status']].values.T
    data_behaviors = data[(data.Hemisphere_audio == a) & (data.Band == b)]
    familiarity.append(normalize_list(data_behaviors['Familiarity'].values))
    times.append(data_behaviors['1_Times'].values)

    arr_concat = np.concatenate((np.array(psd),
                                 np.array(familiarity),
                                 np.array(times)))
    
    features_arr = np.concatenate((personality.T, arr_concat.T), axis=1) 
    
    def _df():
        columns = ['Subjects', 'MIDI', 'Status',
                   'MIDI_Left_Theta', 'MIDI_Left_Alpha', 'MIDI_Left_Beta',
                   'MIDI_Right_Theta', 'MIDI_Right_Alpha', 'MIDI_Right_Beta',
                   'Song_Left_Theta', 'Song_Left_Alpha', 'Song_Left_Beta',
                   'Song_Right_Theta', 'Song_Right_Alpha', 'Song_Right_Beta',
                   'Familiarity', 'Times'] 
        
        return pd.DataFrame(features_arr, columns=columns)
        
    return _df()

def knn(kfold):
    features_ml = df_for_ml()

    familiarity_columns = ['Familiarity']
    times_columns = ['Times']
    behavior_columns = ['Familiarity', 'Times']
    psd_midi_columns = ['MIDI_Left_Theta', 'MIDI_Left_Alpha', 'MIDI_Left_Beta',
                        'MIDI_Right_Theta', 'MIDI_Right_Alpha', 'MIDI_Right_Beta']
    psd_song_columns = ['Song_Left_Theta', 'Song_Left_Alpha', 'Song_Left_Beta',
                        'Song_Right_Theta', 'Song_Right_Alpha', 'Song_Right_Beta']

    psd_midi_left_columns = ['MIDI_Left_Theta', 'MIDI_Left_Alpha', 'MIDI_Left_Beta']
    psd_midi_right_columns = ['MIDI_Right_Theta', 'MIDI_Right_Alpha', 'MIDI_Right_Beta']
    psd_song_left_columns = ['Song_Left_Theta', 'Song_Left_Alpha', 'Song_Left_Beta']
    psd_song_right_columns = ['Song_Right_Theta', 'Song_Right_Alpha', 'Song_Right_Beta']
 
    
    psd_left_columns = ['MIDI_Left_Theta', 'MIDI_Left_Alpha', 'MIDI_Left_Beta',
                        'Song_Left_Theta', 'Song_Left_Alpha', 'Song_Left_Beta']
    
    psd_right_columns = ['MIDI_Right_Theta', 'MIDI_Right_Alpha', 'MIDI_Right_Beta',
                         'Song_Right_Theta', 'Song_Right_Alpha', 'Song_Right_Beta']
    
    psd_theta_columns = ['MIDI_Left_Theta', 'MIDI_Right_Theta', 'Song_Left_Theta', 'Song_Right_Theta']
    psd_alpha_columns = ['MIDI_Left_Alpha', 'MIDI_Right_Alpha', 'Song_Left_Alpha', 'Song_Right_Alpha']
    psd_beta_columns = ['MIDI_Left_Beta', 'MIDI_Right_Beta', 'Song_Left_Beta', 'Song_Right_Beta']
    
    psd_midi_theta_columns = ['MIDI_Left_Theta', 'MIDI_Right_Theta']
    psd_midi_alpha_columns = ['MIDI_Left_Alpha', 'MIDI_Right_Alpha']
    psd_midi_beta_columns = ['MIDI_Left_Beta', 'MIDI_Right_Beta']
    
    psd_song_theta_columns = ['Song_Left_Theta', 'Song_Right_Theta']
    psd_song_alpha_columns = ['Song_Left_Alpha', 'Song_Right_Alpha']
    psd_song_beta_columns = ['Song_Left_Beta', 'Song_Right_Beta']
    
    psd_left_theta_columns = ['MIDI_Left_Theta', 'Song_Left_Theta']
    psd_left_alpha_columns = ['MIDI_Left_Alpha', 'Song_Left_Alpha']
    psd_left_beta_columns = ['MIDI_Left_Beta', 'Song_Left_Beta']
    
    psd_right_theta_columns = ['MIDI_Right_Theta', 'Song_Right_Theta']
    psd_right_alpha_columns = ['MIDI_Right_Alpha', 'Song_Right_Alpha']
    psd_right_beta_columns = ['MIDI_Right_Beta', 'Song_Right_Beta']

    
    data_dict = {
                 'psd' : features_ml[psd_midi_columns + psd_song_columns].values,
                 'psd_midi' : features_ml[psd_midi_columns].values,
                 'psd_song' : features_ml[psd_song_columns].values,
                 #'psd_left' : features_ml[psd_left_columns].values,
                 #'psd_right' : features_ml[psd_right_columns].values,
                 
                 'psd_midi_left' : features_ml[psd_midi_left_columns].values,
                 'psd_midi_right' : features_ml[psd_midi_right_columns].values,
                 'psd_song_left' : features_ml[psd_song_left_columns].values,
                 'psd_song_right' : features_ml[psd_song_right_columns].values,
                 
                 # 'psd_theta' : features_ml[psd_theta_columns].values,
                 # 'psd_alpha' : features_ml[psd_alpha_columns].values,
                 # 'psd_beta' : features_ml[psd_beta_columns].values,
                 
                 # 'psd_theta_alpha' : features_ml[psd_theta_columns + psd_alpha_columns].values,
                 # 'psd_theta_beta' : features_ml[psd_theta_columns + psd_beta_columns].values,
                 # 'psd_alpha_beta' : features_ml[psd_alpha_columns + psd_beta_columns].values,
                 
                 # 'psd_midi_theta' : features_ml[psd_midi_theta_columns].values,
                 # 'psd_midi_alpha' : features_ml[psd_midi_alpha_columns].values,
                 # 'psd_midi_beta' : features_ml[psd_midi_beta_columns].values,
                 
                 # 'psd_midi_theta_alpha' : features_ml[psd_midi_theta_columns + psd_midi_alpha_columns].values,
                 # 'psd_midi_theta_beta' : features_ml[psd_midi_theta_columns + psd_midi_beta_columns].values,
                 # 'psd_midi_alpha_beta' : features_ml[psd_midi_alpha_columns + psd_midi_beta_columns].values,
                 
                 # 'psd_song_theta' : features_ml[psd_song_theta_columns].values,
                 # 'psd_song_alpha' : features_ml[psd_song_alpha_columns].values,
                 # 'psd_song_beta' : features_ml[psd_song_beta_columns].values,
                 
                 # 'psd_song_theta_alpha' : features_ml[psd_song_theta_columns + psd_song_alpha_columns].values,
                 # 'psd_song_theta_beta' : features_ml[psd_song_theta_columns + psd_song_beta_columns].values,
                 # 'psd_song_alpha_beta' : features_ml[psd_song_alpha_columns + psd_song_beta_columns].values,
                    
                 # 'psd_left_theta' : features_ml[psd_left_theta_columns].values,
                 # 'psd_left_alpha' : features_ml[psd_left_alpha_columns].values,
                 # 'psd_left_beta' : features_ml[psd_left_beta_columns].values,
                 
                 # 'psd_left_theta_alpha' : features_ml[psd_left_theta_columns + psd_left_alpha_columns].values,
                 # 'psd_left_theta_beta' : features_ml[psd_left_theta_columns + psd_left_beta_columns].values,
                 # 'psd_left_alpha_beta' : features_ml[psd_left_alpha_columns + psd_left_beta_columns].values,
                 
                 # 'psd_right_theta' : features_ml[psd_right_theta_columns].values,
                 # 'psd_right_alpha' : features_ml[psd_right_alpha_columns].values,
                 # 'psd_right_beta' : features_ml[psd_right_beta_columns].values,
                 
                 # 'psd_right_theta_alpha' : features_ml[psd_right_theta_columns + psd_right_alpha_columns].values,
                 # 'psd_right_theta_beta' : features_ml[psd_right_theta_columns + psd_right_beta_columns].values,
                 # 'psd_right_alpha_beta' : features_ml[psd_right_alpha_columns + psd_right_beta_columns].values,
                 

                 'psd_behaviors' : features_ml[psd_midi_columns + psd_song_columns + behavior_columns].values,
                 'psd_midi_behaviors' : features_ml[psd_midi_columns + behavior_columns].values,
                 'psd_song_behaviors' : features_ml[psd_song_columns + behavior_columns].values,
                 #'psd_left_behaviors' : features_ml[psd_left_columns + behavior_columns].values,
                 #'psd_right_behaviors' : features_ml[psd_right_columns + behavior_columns].values,

                 'psd_midi_left_behaviors' : features_ml[psd_midi_left_columns + behavior_columns].values,
                 'psd_midi_right_behaviors' : features_ml[psd_midi_right_columns + behavior_columns].values,
                 'psd_song_left_behaviors' : features_ml[psd_song_left_columns + behavior_columns].values,
                 'psd_song_right_behaviors' : features_ml[psd_song_right_columns + behavior_columns].values,
                 
                 # 'psd_theta_behaviors' : features_ml[psd_theta_columns + behavior_columns].values,
                 # 'psd_alpha_behaviors' : features_ml[psd_alpha_columns + behavior_columns].values,
                 # 'psd_beta_behaviors' : features_ml[psd_beta_columns + behavior_columns].values,
                 
                 # 'psd_theta_alpha_behaviors' : features_ml[psd_theta_columns + psd_alpha_columns].values,
                 # 'psd_theta_beta_behaviors' : features_ml[psd_theta_columns + psd_beta_columns].values,
                 # 'psd_alpha_beta_behaviors' : features_ml[psd_alpha_columns + psd_beta_columns].values,
                    
                 # 'psd_midi_theta_behaviors' : features_ml[psd_midi_theta_columns + behavior_columns].values,
                 # 'psd_midi_alpha_behaviors' : features_ml[psd_midi_alpha_columns + behavior_columns].values,
                 # 'psd_midi_beta_behaviors' : features_ml[psd_midi_beta_columns + behavior_columns].values,
                 
                 # 'psd_midi_theta_alpha_behaviors' : features_ml[psd_midi_theta_columns + psd_midi_alpha_columns + behavior_columns].values,
                 # 'psd_midi_theta_beta_behaviors' : features_ml[psd_midi_theta_columns + psd_midi_beta_columns + behavior_columns].values,
                 # 'psd_midi_alpha_beta_behaviors' : features_ml[psd_midi_alpha_columns + psd_midi_beta_columns + behavior_columns].values,
                 
                 # 'psd_song_theta_behaviors' : features_ml[psd_song_theta_columns + behavior_columns].values,
                 # 'psd_song_alpha_behaviors' : features_ml[psd_song_alpha_columns + behavior_columns].values,
                 # 'psd_song_beta_behaviors' : features_ml[psd_song_beta_columns + behavior_columns].values,
                 
                 # 'psd_song_theta_alpha_behaviors' : features_ml[psd_song_theta_columns + psd_song_alpha_columns + behavior_columns].values,
                 # 'psd_song_theta_beta_behaviors' : features_ml[psd_song_theta_columns + psd_song_beta_columns + behavior_columns].values,
                 # 'psd_song_alpha_beta_behaviors' : features_ml[psd_song_alpha_columns + psd_song_beta_columns + behavior_columns].values,
                    
                 # 'psd_left_theta_behaviors' : features_ml[psd_left_theta_columns + behavior_columns].values,
                 # 'psd_left_alpha_behaviors' : features_ml[psd_left_alpha_columns + behavior_columns].values,
                 # 'psd_left_beta_behaviors' : features_ml[psd_left_beta_columns + behavior_columns].values,
                 
                 # 'psd_left_theta_alpha_behaviors' : features_ml[psd_left_theta_columns + psd_left_alpha_columns + behavior_columns].values,
                 # 'psd_left_theta_beta_behaviors' : features_ml[psd_left_theta_columns + psd_left_beta_columns + behavior_columns].values,
                 # 'psd_left_alpha_beta_behaviors' : features_ml[psd_left_alpha_columns + psd_left_beta_columns + behavior_columns].values,
                 
                 # 'psd_right_theta_behaviors' : features_ml[psd_right_theta_columns + behavior_columns].values,
                 # 'psd_right_alpha_behaviors' : features_ml[psd_right_alpha_columns + behavior_columns].values,
                 # 'psd_right_beta_behaviors' : features_ml[psd_right_beta_columns + behavior_columns].values,
                 
                 # 'psd_right_theta_alpha_behaviors' : features_ml[psd_right_theta_columns + psd_right_alpha_columns + behavior_columns].values,
                 # 'psd_right_theta_beta_behaviors' : features_ml[psd_right_theta_columns + psd_right_beta_columns + behavior_columns].values,
                 # 'psd_right_alpha_beta_behaviors' : features_ml[psd_right_alpha_columns + psd_right_beta_columns + behavior_columns].values,
                    
                 'behaviors' : features_ml[behavior_columns].values,

                 'psd_familiarity' : features_ml[psd_midi_columns + psd_song_columns + familiarity_columns].values,
                 'psd_midi_familiarity' : features_ml[psd_midi_columns + familiarity_columns].values,
                 'psd_song_familiarity' : features_ml[psd_song_columns + familiarity_columns].values,
                 #'psd_left_familiarity' : features_ml[psd_left_columns + familiarity_columns].values,
                 #'psd_right_familiarity' : features_ml[psd_right_columns + familiarity_columns].values,

                 'psd_midi_left_familiarity' : features_ml[psd_midi_left_columns + familiarity_columns].values,
                 'psd_midi_right_familiarity' : features_ml[psd_midi_right_columns + familiarity_columns].values,
                 'psd_song_left_familiarity' : features_ml[psd_song_left_columns + familiarity_columns].values,
                 'psd_song_right_familiarity' : features_ml[psd_song_right_columns + familiarity_columns].values,
                 
                 'psd_times' : features_ml[psd_midi_columns + psd_song_columns + times_columns].values,
                 'psd_midi_times' : features_ml[psd_midi_columns + times_columns].values,
                 'psd_song_times' : features_ml[psd_song_columns + times_columns].values,
                 #'psd_left_times' : features_ml[psd_left_columns + times_columns].values,
                 #'psd_right_times' : features_ml[psd_right_columns + times_columns].values,

                 'psd_midi_left_times' : features_ml[psd_midi_left_columns + times_columns].values,
                 'psd_midi_right_times' : features_ml[psd_midi_right_columns + times_columns].values,
                 'psd_song_left_times' : features_ml[psd_song_left_columns + times_columns].values,
                 'psd_song_right_times' : features_ml[psd_song_right_columns + times_columns].values,



    }

    def _classification():

        accuracy_mean_list, accuracy_min_list, accuracy_max_list = [], [], []
        accuracy_sd_list, accuracy_se_list = [], []

        f1_mean_list, f1_min_list, f1_max_list = [], [], []
        f1_sd_list, f1_se_list = [], []

        precision_mean_list, precision_min_list, precision_max_list = [], [], []
        precision_mean_list, precision_min_list, precision_max_list = [], [], []
        
        recall_sd_list, recall_se_list = [], []
        recall_sd_list, recall_se_list = [], []

        best_params_list = []
        kfold_accuracy_for_df = []
        kfold_f1_macro_for_df, kfold_f1_for_df, kfold_precision_for_df, kfold_recall_for_df = [], [], [], []
        params_for_df = []

        kfold_f1_for_csv, kfold_precision_for_csv, kfold_recall_for_csv  = [], [], []

        feature_columns = data_dict.keys()
        scaler = MinMaxScaler()
        
        for feature_column in feature_columns:
            X = data_dict[feature_column]
            y = features_ml['Status'].values
            kfold_accuracy_list = []
            kfold_f1_macro_list, kfold_f1_list, kfold_precision_list, kfold_recall_list = [], [], [], []
           

            params_list = []
            # kfold_accuracy_for_df
            kfold_f1_macro_for_df, kfold_f1_for_df, kfold_precision_for_df, kfold_recall_for_df = [], [], [], []
            # params_for_df = []

            
            print ('\n')
            print (path_csv)
            print ('K-fold : ', kfold)
            print ('Features : ', feature_column)
            print ('\n')
            
            if kfold is 'Stratified':
                k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Stratified
                k_fold_split = k_fold.split(X, y)
                
                
            elif kfold is 'LeaveOneSubjectOut':
                subjects = features_ml['Subjects'].values
                k_fold = LeaveOneGroupOut()       
                k_fold_split = k_fold.split(X, y, subjects)
            
            for train_index, test_index in k_fold_split: 
        #         print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train = scaler.fit_transform(X_train)
                # X_test = scaler.fit_transform(X_test)
                X_test = scaler.transform(X_test)

     
                # GridSearch CV
                KNN = KNeighborsClassifier()
                print (KNN)

                paramaters = {
                             'n_neighbors': np.arange(10, 50),
                             'weights' : ['uniform', 'distance'],
                             'metric' : ['euclidean', 'manhattan', 'minskowski']
                             }

                gridsearch = GridSearchCV(KNN,
                                          paramaters,
                                          cv=k_fold,
                                          verbose=1,
                                          n_jobs=-1)
                             

                groups = None if kfold is 'Stratified' else subjects[train_index]
                # if kfold is 'Stratified':
                  # groups = None

                # elif kfold is 'LeaveOneSubjectOut':
                  # groups = subjects[train_index]


                gridsearch.fit(X_train, y_train, groups=groups)    
                optimal_params = gridsearch.best_params_

                 # Support vector machine
                knn = KNeighborsClassifier(**optimal_params) # class_weight = 'balanced' = {Favored :3 and Non_favored :1) approximately.
                print (knn.get_params())
                knn.fit(X_train, y_train)
                knn_accuracy = knn.score(X_test, y_test) 
                y_pred = knn.predict(X_test)
                f1_ = f1_score(y_test, y_pred, average=None)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                precision_ = precision_score(y_test, y_pred, average=None)
                recall_ = recall_score(y_test, y_pred, average=None)

            
        #         print (confusion_matrix(y_test_stratified, y_pred))
        #         print (classification_report(y_test_stratified,y_pred))

#                 print ('Kernel: %s' % (gridsearch.best_params_['kernel']))
#                 print ("Accuracy: %0.2f %s\n" % (svc_accuracy.mean() * 100, '%'))
#                 print ("---------------------------------------------------------")
                # print (f1_[1])
                kfold_accuracy_list.append(knn_accuracy) # 10 models
                kfold_f1_list.append(f1_)
                kfold_f1_macro_list.append(f1_macro)
                kfold_precision_list.append(precision_)
                kfold_recall_list.append(recall_)
                params_list.append(optimal_params) # 10 paramete sets

            index_best_accuracy = kfold_accuracy_list.index(max(kfold_accuracy_list))
            index_worst_accuracy = kfold_accuracy_list.index(min(kfold_accuracy_list))

            index_best_f1 = kfold_f1_macro_list.index(max(kfold_f1_macro_list))
            index_worst_f1 = kfold_f1_macro_list.index(min(kfold_f1_macro_list))

            accuracy_mean_list.append(np.round(np.mean(kfold_accuracy_list)*100, decimals=2))
            accuracy_sd_list.append(np.round(np.std(kfold_accuracy_list),decimals=2))
            accuracy_se_list.append(np.round(np.std(kfold_accuracy_list)/np.sqrt(len(kfold_accuracy_list)),decimals=2))
            accuracy_min_list.append(np.round(kfold_accuracy_list[index_worst_accuracy]*100, decimals=2))
            accuracy_max_list.append(np.round(kfold_accuracy_list[index_best_accuracy]*100, decimals=2))

            f1_mean_list.append(np.round(np.mean(kfold_f1_macro_list)*100, decimals=2))
            f1_sd_list.append(np.round(np.std(kfold_f1_macro_list),decimals=2))
            f1_se_list.append(np.round(np.std(kfold_f1_macro_list)/np.sqrt(len(kfold_f1_list)),decimals=2))
            f1_min_list.append(np.round(kfold_f1_macro_list[index_worst_f1]*100, decimals=2))
            f1_max_list.append(np.round(kfold_f1_macro_list[index_best_f1]*100, decimals=2))

            best_params_list.append(params_list[index_best_accuracy])
            params_for_df.append([params_list])
            kfold_accuracy_for_df.append(kfold_accuracy_list)
            # kfold_f1_for_df.append([kfold_f1_list])
            # kfold_precision_for_df.append([kfold_precision_list])
            # kfold_recall_for_df.append([kfold_recall_list])
            
            for i in range(len(kfold_f1_list)):
              kfold_f1_for_df.append(list(kfold_f1_list[i]))
              kfold_precision_for_df.append([list(kfold_precision_list[i])])
              kfold_recall_for_df.append([list(kfold_recall_list[i])])

            print (f'{len(kfold_f1_for_df)}')
          
            kfold_f1_for_csv.append([kfold_f1_for_df])
            kfold_precision_for_csv.append([kfold_precision_for_df])
            kfold_recall_for_csv.append([kfold_recall_for_df])
            print (f'{len(kfold_f1_for_df)}')
            # # accuracy_dict = {'accuracy': kfold_accuracy_for_df}
            # params_dict = {'params': params_for_df}
            # f1_dict = {'f1_score' : kfold_f1_for_df}
            # precision_dict = {'precision' : kfold_precision_for_df}
            # recall_dict = {'recall' : kfold_recall_for_df}

            
            print ('\n')
            print ('Average of accuracy : %.2f (+/- %.2f)' % (np.mean(kfold_accuracy_list)*100, np.std(kfold_accuracy_list)))
            print ('Highest accuracy : %.2f \n' % (kfold_accuracy_list[index_best_accuracy]*100))
            print ('Average of F1-score : %.2f (+/- %.2f)' % (np.mean(kfold_f1_macro_list)*100, np.std(kfold_f1_macro_list)))  
            print ('Highest F1-score : %.2f \n' % (kfold_f1_macro_list[index_best_f1]*100))
            print ('Best parameters : %s' % (params_list[index_best_accuracy]))
            print ('\n-------------------------------------------------------------\n')

        df_columns_1 = [kfold + '_acc_mean', kfold + '_acc_sd', kfold + '_acc_se', kfold + '_acc_min', kfold + '_acc_max',
                        kfold + '_f1_mean', kfold + '_f1_sd', kfold + '_f1_se', kfold + '_f1_min', kfold + '_f1_max',
                        kfold + '_best_params', kfold + '_fold_accuracy', kfold + '_fold_best_params', kfold + '_fold_f1', kfold + '_fold_precision', kfold + '_fold_recall']

        print ('Writing dataframe ...')
      
        result_df_1 = pd.DataFrame(np.array([accuracy_mean_list,
                                             accuracy_sd_list,
                                             accuracy_se_list,
                                             accuracy_min_list,
                                             accuracy_max_list,
                                             f1_mean_list,
                                             f1_sd_list,
                                             f1_se_list,
                                             f1_min_list,
                                             f1_max_list,
                                             best_params_list,
                                             kfold_accuracy_for_df,
                                             params_for_df,
                                             kfold_f1_for_csv,
                                             kfold_precision_for_csv,
                                             kfold_recall_for_csv
                                             ], dtype=object).T,
                                            columns=df_columns_1,
                                   index=list(feature_columns)) #kfold_accuracy_for_df,params_for_df

        # result_df_2 = pd.DataFrame([np.array([
                                              # # params_for_df,
                                             # kfold_f1_for_df,
                                             # kfold_precision_for_df,
                                             # kfold_recall_for_df
        # ], dtype=object).T],
                                   # columns=df_columns_2,
                                   # index=list(feature_columns))

        return result_df_1

        # return pd.concat([result_df_1, result_df_2], axis=1)
        # return pd.DataFrame(np.array([accuracy_mean_list,
                                      # accuracy_sd_list,
                                      # accuracy_se_list,
                                      # accuracy_min_list,
                                      # accuracy_max_list,
                                      # f1_dict,
                                      # precision_dict,
                                      # recall_dict,
                                      # best_params_list,
                                      # kfold_accuracy_for_df,
                                      # params_for_df], dtype=object).T,
                            # columns=df_columns,
                            # index=list(feature_columns)) 
    
    return _classification()
 
if __name__ == '__main__':
  knn_stratified = knn(kfold='Stratified')
  #knn_loso = knn(kfold='LeaveOneSubjectOut')
  #result_df = knn_stratified.merge(right=svm_loso, left_index=True, right_index=True)
  #result_df.to_csv('knn_result_dividing_behaviour.csv')
  knn_stratified.to_csv('knn_result_hemisphere_final.csv')
  print ('Done')
