import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import mne

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--favored', type=str, required=True,
                    help='Please input favored/non_favored')
parser.add_argument('--a', type=str, required=True,
                    help='Please input midi/song')
parser.add_argument('--channels', type=str, required=True,
                    help='Please input all, left or right')
parser.add_argument('--lfilter', type=int, required=True,
                    help = 'Please input low filter')
parser.add_argument('--hfilter', type=int, required=True,
                    help = 'Please input high filter')
parser.add_argument('--typefilter', type=str, required=True,
                    help = 'Please input fir or iir')

args = parser.parse_args()

def channels():
  with open('channels.json', 'r') as ch:
    channels = json.load(ch)

  return channels

channels = channels()
eeg_all_channels = channels['eeg_channels']
eeg_right_channels = channels['eeg_right_channels']
eeg_left_channels = channels['eeg_left_channels']
eog_channels = channels['eog_channels']
gtec_channels = eeg_all_channels + eog_channels

def picked_channels():
  if args.channels == 'all':
    picked_channels = eeg_all_channels

  elif args.channels == 'left':
    picked_channels = eeg_left_channels

  elif args.channels == 'right':
    picked_channels = eeg_right_channels

  return  picked_channels

# print (picked_channels())

def PSD_processing():    
    path = 'Z:/musec_data/' + args.favored + '/' + args.favored + '_' + args.a + '/' + 'npy'
    os.chdir(os.path.expanduser('~') + path)

    print (os.getcwd())

PSD_processing()

def npy_files():
    npy_files = os.listdir(os.getcwd())
    npy_files.sort()

    return npy_files

def read_files():
    data = [np.load(npy_files()[i]) for i,value in enumerate(npy_files())]
    return data

def PSD():
    data = read_files()
    smp_freq = 1200
    psds_mean_list, freqs_list, psds_std_list = [], [], []
    second_filter_list = []

    for i in range(len(data)): 
        print (i)
        event_id = 1
        
        song_length =  len(data[i][1]) #len(data[1]) 
        song_time = np.around(song_length/1200, decimals=2)

        print ('EEG {} channels'.format(eeg_all_channels))
        print ('Song length (points) : {}'.format(song_length))
        print ('Time of song : {} s'.format(song_time))

        duration = np.floor(song_time)-0.01
        tmin = 0.
        tmax = song_time - 0.01

        ch_types = ['eeg' for i in range(len(eeg_all_channels))] + ['eog']*2
        info = mne.create_info(ch_names=gtec_channels, sfreq=smp_freq, ch_types=ch_types)
        raw = mne.io.RawArray(data[i], info) #(data, info)
        raw.set_montage("standard_1020")

        raw_car, _ = mne.set_eeg_reference(raw, 'average', projection=True)
        events_raw_car = mne.make_fixed_length_events(raw_car, event_id, duration=duration)

        tmin = 0.
        tmax = song_time - 0.01

        epochs_raw_car = mne.Epochs(raw_car, events=events_raw_car, event_id=event_id, tmin=tmin,
                                    tmax=tmax, baseline=None, verbose=True)
        epochs_raw_car_avg = epochs_raw_car.average()

        def second_filter():
            epochs_raw_car_avg.filter(l_freq=args.lfilter,
                                      h_freq=args.hfilter,
                                      method=args.typefilter)
            epochs_raw_car_avg_second_filtered = epochs_raw_car_avg.to_data_frame().values.T

            return epochs_raw_car_avg_second_filtered
        
        second_filter_list.append(second_filter())

        second_filter_noBaseline = second_filter()[:, :]  #for i in range(0,81)]
    
        song_length_noBaseline =  len(second_filter_noBaseline[0]) #len(data[1]) 
        song_time_noBaseline = np.around(song_length_noBaseline/1200, decimals=2)

        print ('EEG {} channels'.format(eeg_all_channels))
        print ('Song length (points) : {}'.format(song_length_noBaseline))
        print ('Time of song : {} s'.format(song_time_noBaseline))

        duration_noBaseline = np.floor(song_time_noBaseline)-0.01
        tmin_noBaseline = 0.
        tmax_noBaseline = song_time_noBaseline - 0.01

        ch_types_second_filtered = ['eeg' for i in range(len(second_filter()))]

        print ('second_filter', len(ch_types_second_filtered) )
        print (len(eeg_all_channels))
        
        info_second_filtered = mne.create_info(ch_names=eeg_all_channels,
                                               sfreq=smp_freq,
                                               ch_types=ch_types_second_filtered)
        
        raw_second_filtered = mne.io.RawArray(second_filter_noBaseline,
                                              info_second_filtered)
        
        raw_second_filtered.set_montage("standard_1020")
        
        events_raw_second_filtered = mne.make_fixed_length_events(raw_second_filtered,
                                                                  event_id,
                                                                  duration=duration_noBaseline)

        epochs_raw_second_filtered = mne.Epochs(raw_second_filtered,
                                                events=events_raw_second_filtered,
                                                event_id=event_id,
                                                tmin=tmin_noBaseline,
                                                tmax=tmax_noBaseline,
                                                baseline=None,
                                                verbose=True,
                                                picks=picked_channels())
        
        psds, freqs = mne.time_frequency.psd_multitaper(epochs_raw_second_filtered,
                                                        fmin=1, fmax=40)

        print ('First :',psds)
        
        psds = 10*np.log10(psds)

        print ('Second', psds)
        psds_mean = psds.mean(0).mean(0)

        print ('Third', psds)
        psds_std = psds.mean(0).std(0)
        psds_mean_list.append(psds_mean)
        psds_std_list.append(psds_std)
        freqs_list.append(freqs)


    # second_filter_list_transpose = [second_filter_list[i].T for i in range(len(second_filter_list))]
    # save_filename = 'second_filter_' + args.favored + '_' + args.a
    # save_foldername = 'second_filter' 
    # save_path = os.path.expanduser('~') + '/musec/musec-env/musec/musec_nas/musec_data/' + args.favored + '/' + args.favored + '_' + args.a + '/' + save_foldername
    # print (os.path.dirname(save_path))
    # np.save(os.path.join(save_path, save_filename), second_filter_list_transpose)


    return psds_mean_list, freqs_list, psds_std_list

def save_psd_to_npy():
  psd = PSD()
  psd_reshape = np.reshape(psd, (3,len(psd[0])))
  psd_save_filename = 'psd_' + args.favored + '_' + args.a + '_' + args.channels
  psd_foldername = 'psd_npy'
  psd_save_path = os.path.expanduser('~') + '/musec/musec-env/musec/musec_nas/musec_data/' + args.favored + '/' + args.favored + '_' + args.a + '/'
  
  try:
    os.mkdir(psd_save_path + psd_foldername)
  
  except OSError:
    print ('Exist')
  
  else:
    print ('Successfully')

  np.save(os.path.join(psd_save_path, psd_foldername, psd_save_filename), psd_reshape)

save_psd_to_npy()
