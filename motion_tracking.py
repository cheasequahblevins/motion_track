
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from operator import itemgetter
import itertools
import tkinter as tk
from tkinter import ttk
import math
import statistics
import random
from scipy import signal
from scipy import interpolate
from scipy.interpolate import interp1d
from moviepy.editor import *
from scipy.stats.stats import pearsonr

#-----------------------------------------------------------------
# CHECKS FOR MONOTONICITY & MISSING DATA IN TIME SERIES
#-----------------------------------------------------------------

def is_smooth(L, fraction_data_required):
    if(np.count_nonzero(L) >= round((fraction_data_required)*len(L))):
        L = [x for x in L if x != 0]
        return(all(round(x,3) <= round(y,3) for x, y in zip(L, L[1:])))
    else:
        return(False)
        
#-----------------------------------------------------------------
# RETURNS LIST OF INTERSECTING VALUES IN TWO LISTS
#-----------------------------------------------------------------


def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

#-----------------------------------------------------------------
# RETURNS .H5 FILE 'master_df' CONTAINING ENTIRE TIME SERIES FOR
# ONE SESSION FOR EACH EFFECTOR, FILTERED BY DLC LIKELIHOOD AND
# ORGANIZED BY STIM/CTRL REACH TYPE.
#-----------------------------------------------------------------

def process(original, playback, dlc_df, likelihood_cutoff):

    # Match Stim in original and playback dataframes (playback alignment)
    original = pd.read_csv(original, header=None, names=['time', 'x', 'y', 'z', 'stim'])
    playback = pd.read_csv(playback, header=None, names=['time', 'x', 'y', 'z', 'stim'])

    # get frames where stim occurs in original df
    stim_only = original[original['stim'] == 1]

    if len(stim_only) == 0:
        #return(print('no stim trials detected'))
        print('no stim trials detected, x_threshold assigned 0.16')
        x_threshold = 0.16
        
    else:
        x_threshold = stim_only.x.apply(lambda x: math.floor(x * 100) / 100).min()
        print('stim_x = ' + str(x_threshold))
            
        # ensures original and playback have same number of significant digits
        # so that they can be accurately matched to each other.
        original = round(original, 5)
        playback = round(playback, 5)
        
        # default value 8 in playback: means x,y,z were not tracked in playback.
        # Also used for unanalyzed values at beginning/end of playback vector.
        playback.stim = 8
        
        original.insert(5, 'threshold', 0)
        playback.insert(5, 'threshold', 8)
        
        # for loop to find all x threshold crossings in original df (putative reaches)
        baseline_count, pass_count, stim_detected, stim_missed = 0,0,0,0
        
        for i in range(1,(len(original.x) - 120)):

#             IMPORTANT DECISIONS to classify threshold crossings related to smoothness of data
            if (original.x[i] >= x_threshold and original.x[i-1] <= x_threshold and
                is_smooth(original.x[i-2:i+6], .50) and original.x[i] < x_threshold*1.25):
#               is_smooth(original.y[i-2:i+6], .50) and is_smooth(original.x[i-2:i+6], .5)):

#            if (original.x[i] >= x_threshold and original.x[i-1] <= x_threshold and
#                is_smooth(original.x[i:i+6], .5)):

                original.loc[i, 'threshold'] = 1
            
            if (original.loc[i, 'stim'] == 1 and original.loc[i, 'threshold'] == 0):
#                original.loc[i, 'stim'] = 0
#                print(str(i) + ' stim reach NOT identified')
                stim_missed = stim_missed + 1
                
            if (original.loc[i, 'stim'] == 1 and original.loc[i, 'threshold'] == 1):
                # print(str(i) + ' stim reach identified')
                stim_detected = stim_detected + 1
                
        # end for loop
        print(str(100*round(stim_detected/(stim_missed+stim_detected), 2)) +
              ' % stim reaches detected in session')
        
        # delete all rows where the tracking dot was not detected
        original = original[original.x != 0]

        # Loop to Match PLAYBACK and ORIGINAL dataframes
        for i in range(1, len(playback.x)):
        
            # Match playback and original data
            # if tracking dot is identified in playback, then match it to original
            if playback.x[i] != 0 or playback.y[i] !=0 or playback.z[i] !=0:
                subx = original[original.x == playback.x[i]]
                suby = subx[subx.y == playback.y[i]]
                matchRow = suby[suby.z == playback.z[i]] # frames in ORIGINAL video matching playback frame i
                matchRow = matchRow.reset_index()

                # if there is only one frame in original data that matches current playback frame
                # then copy the values from stim and threshold to the playback dataframe
                if len(matchRow.stim) == 1:
                    playback.loc[i, 'stim'] = matchRow.stim[0] # Either 1 or 0
                    playback.loc[i, 'threshold'] = matchRow.threshold[0] # Either 1 or 0
                
                # else if there is more than one frame in original data that matches current playback
                # assign  stim and threshold values 4 to signify this
                elif len(matchRow.x) > 1:
                    playback.loc[i, 'stim'] = 4 # meaning that this row was measured 2+ times in original
                    playback.loc[i, 'threshold'] = 4 # meaning that this row was measured 2+ times in original
                    
                # else if there are no frames in original data that match the playback,
                # assign stim and threshold values 6 to signify this
                else:
                    playback.loc[i, 'stim'] = 6 # dot tracked in playback, but it was not tracked in original
                    playback.loc[i, 'threshold'] = 6 # dot tracked in playback, but it was not tracked in original

        # Now that original and playback data are aligned, bring in DLC data, which is aligned to playback
        dlc_df = pd.read_csv(dlc_df, header = [1,2])
        
        # clean up dataframe and get list of effectors tracked
        master_df = dlc_df.drop(['bodyparts'], axis=1)
        effector_list = list(set(map(itemgetter(0), master_df)))
        
        # Filter master_df data by likelihood
        for effector in effector_list:
            master_df.loc[:,(effector, 'x')] = round((master_df[effector, 'x'])*(master_df[effector, 'likelihood'] > likelihood_cutoff),2)
            
            # invert y so that visualization makes more intuitive sense
            # ANALYSIS SPECIFIC
            master_df.loc[:,(effector, 'y')] = round((master_df[effector, 'y'] - 480)*-1,2)
            master_df.loc[:,(effector, 'y')] = master_df[(effector, 'y')]*(master_df[effector, 'likelihood'] > likelihood_cutoff)

        # prepare master_df dataframe for final output
        master_df['stim'] = playback.stim
        master_df['threshold'] = playback.threshold
        index = master_df.index.set_names(['frame_idx'])
        master_df.index = index
        playback = playback.drop(columns=['time', 'stim', 'threshold'])
        columns = pd.MultiIndex.from_product([['motive'],['x', 'y', 'z']])
        playback.columns = columns

        master_df = master_df.merge(playback,left_index=True, right_index=True)
        
    return(master_df)

#-----------------------------------------------------------------------------------
# CONCATENATES .H5 FILES PRODUCED BY process() AND MULTI-INDEXES BY SESSION NAME
#-----------------------------------------------------------------------------------

def concat_sessions(directory):
    session_list = os.listdir(directory)
    session_list.sort()

    if '.DS_Store' in session_list:
        session_list.remove('.DS_Store')

    for s in range(0,len(session_list)):

        if s == 0:
            concatenated = pd.read_hdf((directory + session_list[s]))
            concatenated = pd.concat([concatenated], keys = [session_list[s][0:15]], names = ['session'])
        elif s > 0:
            hold = pd.read_hdf((directory + session_list[s]))
            hold = pd.concat([hold], keys = [session_list[s][0:15]], names = ['session'])
            concatenated = pd.concat([concatenated,hold])
            
    return(concatenated)
    
#-----------------------------------------------------------------------------------
# CREATES DIRECTORIES FOR ANALYSES, SPECIFIC TO EACH COMPUTER
#-----------------------------------------------------------------------------------

def directories(animal, subfolder):

    base_dir = '/Users/cheasequah/Desktop/motion_track/'

    motive_original_dir = base_dir + '0-motive_original/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(motive_original_dir):
        if  not os.path.isdir(base_dir + '0-motive_original/' + animal + '/'):
            os.mkdir(base_dir + '0-motive_original/' + animal + '/')
        os.mkdir(motive_original_dir)

    motive_playback_dir = base_dir + '1-motive_playback/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(motive_playback_dir):
        if  not os.path.isdir(base_dir + '1-motive_playback/' + animal + '/'):
            os.mkdir(base_dir + '1-motive_playback/' + animal + '/')
        os.mkdir(motive_playback_dir)

    dlc_dir = base_dir + '2-DeepLabCut/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(dlc_dir):
        if not os.path.isdir(base_dir + '2-DeepLabCut/' + animal + '/'):
            os.mkdir(base_dir + '2-DeepLabCut/' + animal + '/')
        os.mkdir(dlc_dir)

    processed_dir = base_dir + '3-processed/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(processed_dir):
        if not os.path.isdir(base_dir + '3-processed/' + animal + '/'):
            os.mkdir(base_dir + '3-processed/' + animal + '/')
        os.mkdir(processed_dir)
        
    clipped_dir = base_dir + '4-clipped/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(clipped_dir):
        if not os.path.isdir(base_dir + '4-clipped/' + animal + '/'):
            os.mkdir(base_dir + '4-clipped/' + animal + '/')
        os.mkdir(clipped_dir)

    plot_dir = base_dir + '5-plots/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(plot_dir):
        if not os.path.isdir(base_dir + '5-plots/' + animal + '/'):
            os.mkdir(base_dir + '5-plots/' + animal + '/')
        os.mkdir(plot_dir)
        
    original_videos_dir = base_dir + '6-full_videos/original_videos/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(original_videos_dir):
        if not os.path.isdir(base_dir + '6-full_videos/original_videos/' + animal + '/'):
            os.mkdir(base_dir + '6-full_videos/original_videos/' + animal + '/')
        os.mkdir(original_videos_dir)
        
    labeled_videos_dir = base_dir + '6-full_videos/labeled_videos/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(labeled_videos_dir):
        if not os.path.isdir(base_dir + '6-full_videos/labeled_videos/' + animal + '/'):
            os.mkdir(base_dir + '6-full_videos/labeled_videos/' + animal + '/')
        os.mkdir(labeled_videos_dir)
        
    video_clips_dir = base_dir + '7-video_clips/' + animal + '/' + str(subfolder) + '/'
    if  not os.path.isdir(video_clips_dir):
        if not os.path.isdir(base_dir + '7-video_clips/' + animal + '/'):
            os.mkdir(base_dir + '7-video_clips/' + animal + '/')
        os.mkdir(video_clips_dir)
        
    return(motive_original_dir, motive_playback_dir, dlc_dir, processed_dir, clipped_dir, plot_dir,         original_videos_dir, labeled_videos_dir, video_clips_dir)

#-----------------------------------------------------------------------------------
# LISTS FILES IN DIRECTOR IN ALPHABETICAL ORDER, AND DELETES .DS_STORE FILE
# SO THAT A LIST OF FILES CAN BE INDEXED PROPERLY
#-----------------------------------------------------------------------------------

def list_files(dir):
    # Remove any .DS_Store files that self-populate
    list = os.listdir(dir)
    list.sort()
    if '.DS_Store' in list:
        list.remove('.DS_Store')
        
    return(list)
    
#-----------------------------------------------------------------------------------
# FINDS THE FRAME LOCATION OF EACH REACH AND RETURNS A DATAFRAME OF REACH
# LOCATION AND REACH TYPE (STIM VS CTRL)
#-----------------------------------------------------------------------------------

def find_reaches(master_df):
    
    input = master_df.copy()
    stim_loc = []
    ctrl_loc =[]

    session_list = input.index.unique(level=0)

    for session in range(0, len(session_list)):

        session_name = session_list[session]
        session_df = input.loc[session_name]
        
        stim_boolean = session_df['stim'] == 1
        threshold_boolean = session_df['threshold'] == 1
        not_stim_boolean = session_df['stim'] == 0

        stim_loc_session = pd.DataFrame(data = list(stim_boolean[stim_boolean].index))
        stim_loc_session = stim_loc_session.rename({0 : 'frame'}, axis=1)
        stim_loc_session = pd.concat([stim_loc_session], keys=[session_name])
        
        cross_threshold_loc = list(threshold_boolean[threshold_boolean].index)
        not_stim_loc = list(not_stim_boolean[not_stim_boolean].index)

        ctrl_loc_session = pd.DataFrame(data = list(set(cross_threshold_loc) & set(not_stim_loc)))
        ctrl_loc_session = ctrl_loc_session.rename({0 : 'frame'}, axis=1)
        ctrl_loc_session = pd.concat([ctrl_loc_session], keys=[session_name])

        if (len(stim_loc_session) > 0 and len(ctrl_loc_session) > 0):
        
            if len(stim_loc) == 0:
                stim_loc = pd.concat([stim_loc_session], axis=0, join='outer', keys=None, copy=True)
                ctrl_loc = pd.concat([ctrl_loc_session], axis=0, join='outer', keys=None, copy=True)
            elif len(stim_loc) > 0:
                stim_loc = pd.concat([stim_loc, stim_loc_session], axis=0, join='outer', keys=None, copy=True)
                ctrl_loc = pd.concat([ctrl_loc, ctrl_loc_session], axis=0, join='outer', keys=None, copy=True)
        
        elif len(stim_loc_session) == 0:
        
            if len(ctrl_loc) == 0:
                ctrl_loc = pd.concat([ctrl_loc_session], axis=0, join='outer', keys=None, copy=True)
            elif len(ctrl_loc) > 0:
                ctrl_loc = pd.concat([ctrl_loc, ctrl_loc_session], axis=0, join='outer', keys=None, copy=True)

            ctrl_loc['reach_type'] = 0

        elif len(ctrl_loc_session) == 0:
        
            if len(stim_loc) == 0:
                stim_loc = pd.concat([stim_loc_session], axis=0, join='outer', keys=None, copy=True)
            elif len(stim_loc) > 0:
                stim_loc = pd.concat([stim_loc, stim_loc_session], axis=0, join='outer', keys=None, copy=True)

    ctrl_loc['reach_type'] = 0
    stim_loc['reach_type'] = 1

    reach_locations = pd.concat([ctrl_loc, stim_loc])
    reach_locations['all_reach_count'] = range(0, len(reach_locations))

    return(reach_locations)
    
#-----------------------------------------------------------------------------------
# Uses reach locations to clip reaches from master_df from a set time before
# and after threshold crossing. Returns pandas dataframe with reach data from all
# tracked effectors. Filters out reaches that don't have minimum pct tracked frames
# Includes both interpolated and raw reach time-series
#-----------------------------------------------------------------------------------

def clip_reaches(master_df, limits, fraction_required, axis_motive, axis_dlc):

    input = master_df.copy()
    fps = 120; trial_counter = 0; session_counter = 0
    limits = [round(x/(1000/fps)) for x in limits]
    trim_range = range(0,(limits[0] + limits[1]))
    reach_locations = find_reaches(input)
    unwanted = (['stim', 'threshold', 'Left_wrist', 'Left_backofhand', 'L_Finger1',
                'L_Finger2', 'L_Finger3', 'L_Finger4', 'Left_elbow', 'Right_elbow'])
    input.drop(unwanted, axis=1, inplace = True)
    effector_list = list(set(map(itemgetter(0), input)))
    session_list = input.index.get_level_values(0).unique()
    total_iters = len(list(itertools.product(session_list,effector_list)))
    input['velocity'] = None

    for session,effector in itertools.product(session_list,effector_list):
        
        for reach in reach_locations.loc[session].frame:
            
            trial_df = input.loc[session,effector][(reach-limits[0]):(reach+limits[1])]
            # reset index, but also keep index of original, unclipped data
            trial_df.reset_index(inplace=True)
            trial_df.index = trial_df.index.set_names(['clipped_index'])
            trial_df.replace({0:None}, inplace=True)
            
            # Check to see if there is enough data before doing anything
            if (sum(trial_df.x[limits[0]:].notnull())/limits[1] > fraction_required):
                
                trial_df['trial'] = int(reach_locations.loc[session][reach_locations.loc[session].frame==reach].all_reach_count)
                trial_df['reach_type'] = int(reach_locations.loc[session][reach_locations.loc[session].frame==reach].reach_type)
                trial_df['frame'] = trim_range
                
                if effector == 'motive':
                    
                    trial_df['z_interp'] = trial_df['z']
                    trial_df['z_interp'] = trial_df.z_interp.replace({None:np.nan}).interpolate()
                    trial_df['z_interp'].replace({np.nan:None}, inplace = True)
                    trial_df['velocity'] = velocity(trial_df, axis_motive)
                    
                elif effector != 'motive':
                    
                    trial_df.drop('likelihood', axis=1, inplace = True)
                    trial_df['velocity'] = velocity(trial_df, axis_dlc)
                
                trial_df['x_interp'] = trial_df['x']; trial_df['y_interp'] = trial_df['y']
                trial_df['x_interp'] = trial_df.x_interp.replace({None:np.nan}).interpolate()
                trial_df['y_interp'] = trial_df.y_interp.replace({None:np.nan}).interpolate()
                trial_df['x_interp'].replace({np.nan:None}, inplace = True)
                trial_df['y_interp'].replace({np.nan:None}, inplace = True)
                trial_df['velocity_interp'] = trial_df.velocity
                trial_df['velocity_interp'] = trial_df.velocity_interp.replace({None:np.nan}).interpolate()
                trial_df['velocity_interp'].replace({np.nan:None}, inplace = True)
                    
                if trial_counter == 0:
                    reaches = pd.concat([trial_df], keys=[(session, effector)],
                                            names=['session', 'effector'], sort=True)
                else:
                    temp = pd.concat([trial_df], keys=[(session, effector)],
                                     names=['session', 'effector'], sort=True)
                    reaches = pd.concat([reaches, temp], sort=True)
                
                trial_counter = trial_counter + 1
        
        session_counter = session_counter + 1
        pct_finished = round(100*session_counter/total_iters)
        if pct_finished%10 == 0 and pct_finished > 0:
            print(str(pct_finished) + ' % finished clipping & calculating velocity')
        
    return(reaches)

#-----------------------------------------------------------------------------------
# Takes in an 1,2, or 3 column timeseries describing the position of a
# point in space, then calculates the speed of that point. Can be used for velocity
# or speed calculation depending on choice of dimensions along which to calculate.
#-----------------------------------------------------------------------------------

def velocity(trial_df, axis):

    fps = 120
    trial_df.reset_index(inplace=True)

    velocity_timeseries = np.empty(len(trial_df), dtype=object)

    for frame in range(1,len(trial_df)):
        snip = trial_df.loc[(frame-1):frame]
        if(snip.clipped_index.diff().sum() == 1 and int(snip[axis].isnull().sum()) == 0):
            distance = snip[axis].apply((np.diff), axis = 0)
            velocity_timeseries[frame] = float(distance.loc[0][axis]*fps)

    return(velocity_timeseries)
    
    
#-----------------------------------------------------------------------------------
# Calculates velocity along specified axes form motive and DeepLabCut, then appends
# the velocity and the interpolated velocity to ther clipped (reaches) dataframe
#-----------------------------------------------------------------------------------

def append_velocity(reaches, axis_motive, axis_dlc):

    input = reaches.copy()
    fps = 120; counter = 0;
    session_list = input.index.get_level_values(0).unique()
    effector_list = input.index.get_level_values(1).unique()
    total_iters = len(list(itertools.product(session_list,effector_list)))
    input['velocity'] = None
    input['acceleration'] = None
    input['velocity_interp'] = None

    total_combinations = list(itertools.product(session_list,effector_list))
    actually_exists = input.index.droplevel(2)
    iter_list = intersection(total_combinations, actually_exists)

    for session,effector in iter_list:

        trial_list = input.loc[session,effector].trial.unique()
        
        for t in trial_list:
        
            trial_df = input.loc[session,effector][input.loc[session,effector].trial == t]
            
            if effector == 'motive':
                trial_df['velocity'] = velocity(trial_df, axis_motive)
                #input.loc[session,effector]['acceleration'] = acceleration(input.loc[session,effector], axis_motive)
                            
            elif effector != 'motive':
                trial_df['velocity'] = velocity(trial_df, axis_dlc)
                #input.loc[session,effector]['acceleration'] = acceleration(input.loc[session,effector], axis_dlc)
                            
            trial_df['velocity_interp'] = trial_df.velocity
            trial_df['velocity_interp'] = trial_df.velocity_interp.replace({None:np.nan}).interpolate()
            trial_df['velocity_interp'].replace({np.nan:None}, inplace = True)
            
            input.loc[session,effector][input.loc[session,effector].trial == t].loc['velocity'] = trial_df.velocity
            input.loc[session,effector][input.loc[session,effector].trial == t].loc['velocity'] = trial_df.velocity_interp
                
        counter = counter + 1
        pct_finished = round(100*counter/total_iters)
        if pct_finished%10 == 0 and pct_finished > 0:
            print(str(pct_finished) + ' % finished calculating velocity')
            
    return(input)

#-----------------------------------------------------------------------------------
# Calculates Euclidian distance (in pixels) between two effectors during the reach
# clipped by clip_reaches
#-----------------------------------------------------------------------------------

def compute_distance(reaches, effector_1, effector_2):

    input = reaches.copy()
    fps = 120; trial_counter = 0; session_counter = 0
    session_list = input.index.get_level_values(0).unique()
    total_iters = len(session_list)
    input.drop(['velocity', 'z_interp', 'velocity_interp'], axis=1, inplace=True)

    for session in session_list:
        
        if (effector_1 in input.loc[session].index.get_level_values(0).unique() and
            effector_2 in input.loc[session].index.get_level_values(0).unique()) :
            
            ef1_session = input.loc[session].loc[effector_1]
            ef2_session = input.loc[session].loc[effector_2]
            trials = intersection(ef1_session.trial.unique(),ef2_session.trial.unique())

            for trial in trials:

                trial_df = ef1_session[ef1_session.trial == trial]
                trial_df[['distance', 'distance_interp']] = None

                position_timeseries_1 = ef1_session[ef1_session.trial == trial][['x', 'y']]
                position_timeseries_2 = ef2_session[ef2_session.trial == trial][['x', 'y']]
                trial_df['distance'] = distance(position_timeseries_1, position_timeseries_2)

                # uses interpolated position data to calculate distance
                position_timeseries_1 = ef1_session[ef1_session.trial == trial][['x_interp', 'y_interp']]
                position_timeseries_2 = ef2_session[ef2_session.trial == trial][['x_interp', 'y_interp']]
                trial_df['distance_interp'] = distance(position_timeseries_1, position_timeseries_2)

                # interpolates after calculating distance
                #trial_df['distance_interp'] = trial_df.distance
                #trial_df['distance_interp'] = trial_df.distance_interp.replace({None:np.nan}).interpolate()
                #trial_df['distance_interp'].replace({np.nan:None}, inplace = True)
                
                if trial_counter == 0:
                    distance_df = pd.concat([trial_df], keys=[session],
                                        names=['session'], sort=True)
                    trial_counter = trial_counter + 1
                else:
                    temp = pd.concat([trial_df], keys=[session],
                                        names=['session'], sort=True)
                    distance_df = pd.concat([distance_df, temp], sort=True)

            session_counter = session_counter + 1
            pct_finished = round(100*session_counter/total_iters)
            if pct_finished%10 == 0 and pct_finished > 0:
                print(str(pct_finished) + ' % finished calculating distance')
                
    return(distance_df)

#-----------------------------------------------------------------------------------
# Calculates Euclidian distance (in pixels) between two input points input
# as x-y timeseries
#-----------------------------------------------------------------------------------

def distance(position_timeseries_1, position_timeseries_2):

    distance_timeseries = np.empty(len(position_timeseries_1), dtype=object)

    for frame in range(0,len(position_timeseries_1)):
        
        snip_1 = position_timeseries_1.loc[frame]; snip_2 = position_timeseries_2.loc[frame]
        snip_1.replace({None:0}, inplace=True); snip_2.replace({None:0}, inplace=True)
        
        if ((snip_1 == 0).sum() + (snip_2 == 0).sum()) == 0:
            
            distance = ((snip_1-snip_2)**2).sum()**0.5
            distance_timeseries[frame] = distance
            
    return(distance_timeseries)
    
#-----------------------------------------------------------------------------------
# Plots mean and standard deviation of the distance time-series data by trial type
#-----------------------------------------------------------------------------------

def plot_distance(distance_df, limits, effector_1, effector_2):

    fps = 120
    limits = [round(x/(1000/fps)) for x in limits]

    ctrl_color = 'k'; stim_color = 'r'; stim_line = 'g';
    avgLineThickness = 1.5; alpha_shade = 0.25

    ctrl_trim = distance_df.loc[distance_df.reach_type == 0]
    stim_trim = distance_df.loc[distance_df.reach_type == 1]

    emptyList = [np.nan]*(sum(limits))
    emptyRange = list(range(0,sum(limits)))

    ctrl_plot = pd.DataFrame({'frame':emptyRange,'time':emptyRange,'num':emptyList,'num_interp':emptyList,'pct':emptyList,
                              'distance':emptyList,'distance_lower':emptyList,'distance_upper':emptyList,
                              'distance_interp':emptyList,'distance_lower_interp':emptyList,'distance_upper_interp':emptyList})

    stim_plot = pd.DataFrame({'frame':emptyRange,'time':emptyRange,'num':emptyList,'num_interp':emptyList,'pct':emptyList,
                              'distance':emptyList,'distance_lower':emptyList,'distance_upper':emptyList,
                              'distance_interp':emptyList,'distance_lower_interp':emptyList,'distance_upper_interp':emptyList})


    for frame in ctrl_trim.frame.unique():
        
        df = ctrl_trim.loc[ctrl_trim.frame == frame]
        temp = df[df.distance > 0]
        temp_interp = df[df.distance_interp > 0]
        
        if(len(temp) > 3):
            ctrl_plot.loc[frame, 'pct'] = 100*len(temp)/len(df)
            ctrl_plot.loc[frame, 'num'] = len(temp)

            ctrl_plot.loc[frame, 'distance'] = temp.distance.mean()
            ctrl_plot.loc[frame, 'distance_lower'] = temp.distance.mean()-temp.distance.std()/(len(temp)**0.5)
            ctrl_plot.loc[frame, 'distance_upper']= temp.distance.mean()+temp.distance.std()/(len(temp)**0.5)
        
        if(len(temp_interp) > 3):
            ctrl_plot.loc[frame, 'pct_interp'] = 100*len(temp_interp)/len(df)
            ctrl_plot.loc[frame, 'num_interp'] = len(temp_interp)

            ctrl_plot.loc[frame, 'distance_interp'] = temp_interp.distance_interp.mean()
            ctrl_plot.loc[frame, 'distance_lower_interp'] = temp_interp.distance_interp.mean()-temp_interp.distance_interp.std()/(len(temp_interp)**0.5)
            ctrl_plot.loc[frame, 'distance_upper_interp']= temp_interp.distance_interp.mean()+temp_interp.distance_interp.std()/(len(temp_interp)**0.5)
            ctrl_plot.loc[frame, 'time']= (1000/fps)*(frame-limits[0])

        
    for frame in stim_trim.frame.unique():
        
        df = stim_trim.loc[stim_trim.frame == frame]
        temp = df[df.distance > 0]
        temp_interp = df[df.distance_interp > 0]
        
        if(len(temp) > 3):
            stim_plot.loc[frame, 'pct'] = 100*len(temp)/len(df)
            stim_plot.loc[frame, 'num'] = len(temp)

            stim_plot.loc[frame, 'distance'] = temp.distance.mean()
            stim_plot.loc[frame, 'distance_lower'] = temp.distance.mean()-temp.distance.std()/(len(temp)**0.5)
            stim_plot.loc[frame, 'distance_upper']= temp.distance.mean()+temp.distance.std()/(len(temp)**0.5)

        if(len(temp_interp) > 3):
            stim_plot.loc[frame, 'pct_interp'] = 100*len(temp_interp)/len(df)
            stim_plot.loc[frame, 'num_interp'] = len(temp_interp)

            stim_plot.loc[frame, 'distance_interp'] = temp_interp.distance_interp.mean()
            stim_plot.loc[frame, 'distance_lower_interp'] = temp_interp.distance_interp.mean()-temp_interp.distance_interp.std()/(len(temp_interp)**0.5)
            stim_plot.loc[frame, 'distance_upper_interp']= temp_interp.distance_interp.mean()+temp_interp.distance_interp.std()/(len(temp_interp)**0.5)
            stim_plot.loc[frame, 'time']= (1000/fps)*(frame-limits[0])
        
    # distance plot
    fig, ((ax1, ax3), (ax5, ax6), (ax7, ax8))= plt.subplots(3,2, figsize=(15,25))

    fig.tight_layout(pad = 3)
    
    ax1.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)
    ax3.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)
    ax5.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)
    ax6.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)
    ax7.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)
    ax8.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)


    ax1.set_xlabel('time since stim (ms)')
    ax1.set_ylabel('distance (pixels)')
    ax1.set_xlim(round((1000/fps)*(-limits[0])), round((1000/fps)*(limits[1])))

    ax3.set_xlabel('time since stim (ms)')
    ax3.set_ylabel('distance (pixels)')
    ax3.set_xlim(round((1000/fps)*(-limits[0])), round((1000/fps)*(limits[1])))


    ax1.plot('time', 'distance', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax1.plot('time', 'distance_lower', data=ctrl_plot, color=ctrl_color, linewidth=0)
    ax1.plot('time', 'distance_upper', data=ctrl_plot, color=ctrl_color, linewidth=0)

    ax3.plot('time', 'distance_interp', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax3.plot('time', 'distance_lower_interp', data=ctrl_plot, color=ctrl_color, linewidth=0)
    ax3.plot('time', 'distance_upper_interp', data=ctrl_plot, color=ctrl_color, linewidth=0)

    ax1.plot('time', 'distance', data=stim_plot, color=stim_color, linewidth=avgLineThickness)
    ax1.plot('time', 'distance_lower', data=stim_plot, color=stim_color, linewidth=0)
    ax1.plot('time', 'distance_upper', data=stim_plot, color=stim_color, linewidth=0)
    ax1.set_ylim(5,60)


    ax3.plot('time', 'distance_interp', data=stim_plot, color=stim_color, linewidth=avgLineThickness)
    ax3.plot('time', 'distance_lower_interp', data=stim_plot, color=stim_color, linewidth=0)
    ax3.plot('time', 'distance_upper_interp', data=stim_plot, color=stim_color, linewidth=0)
    ax3.set_ylim(5,60)

    ax1.fill_between(stim_plot.time, stim_plot.distance_lower, stim_plot.distance_upper, color=stim_color, alpha = alpha_shade, lw=0)
    ax1.fill_between(ctrl_plot.time, ctrl_plot.distance_lower, ctrl_plot.distance_upper, color=ctrl_color, alpha = alpha_shade, lw=0)

    ax3.fill_between(stim_plot.time, stim_plot.distance_lower_interp, stim_plot.distance_upper_interp, color=stim_color, alpha = alpha_shade, lw=0)
    ax3.fill_between(ctrl_plot.time, ctrl_plot.distance_lower_interp, ctrl_plot.distance_upper_interp, color=ctrl_color, alpha = alpha_shade, lw=0)

    ax2 = plt.twinx(ax1)
    ax4 = plt.twinx(ax3)

    ax2.set_ylabel('% total data')
    ax2.plot('time', 'pct', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness*.20)
    ax2.plot('time', 'pct', data=stim_plot, color=stim_color, linewidth=avgLineThickness*.20)
    ax2.set_ylim(0,105)
    ax4.set_ylabel('% total data')
    ax4.plot('time', 'pct_interp', data=ctrl_plot, color=ctrl_color, linewidth = avgLineThickness*0.20)
    ax4.plot('time', 'pct_interp', data=stim_plot, color=stim_color, linewidth = avgLineThickness*0.20)
    ax4.set_ylim(0,105)

    ax1.set_title(('(raw) distance between ' + effector_1 + ' & ' + effector_2 + '\n stim n = ' +  str(np.nanmax(stim_plot.num)) +
              ' ctrl n = ' +  str(np.nanmax(ctrl_plot.num))), pad=10)
    ax3.set_title(('(interpolated) distance between ' + effector_1 + ' & ' + effector_2 + '\n stim n = ' + str(np.nanmax(stim_plot.num_interp)) +
              ' ctrl n = ' +  str(np.nanmax(ctrl_plot.num_interp))), pad=10)


    for trial in ctrl_trim.trial.unique():
        trial_df = ctrl_trim[ctrl_trim.trial == trial]
        trial_df['time'] = (1000/fps)*(trial_df.frame-limits[0])
        ax5.plot('time', 'distance', data=trial_df, color=ctrl_color, linewidth=avgLineThickness*.20, alpha=0.5)
        ax6.plot('time', 'distance_interp', data=trial_df, color=ctrl_color, linewidth=avgLineThickness*.20, alpha=0.5)
        
    for trial in stim_trim.trial.unique():
        trial_df = stim_trim[stim_trim.trial == trial]
        trial_df['time'] = (1000/fps)*(trial_df.frame-limits[0])
        ax7.plot('time', 'distance', data=trial_df, color=stim_color, linewidth=avgLineThickness*.20)
        ax8.plot('time', 'distance_interp', data=trial_df, color=stim_color, linewidth=avgLineThickness*.20)

    ax5.set_ylabel('distance (pixels)')
    ax5.set_xlabel('time since stim (ms)')
    ax5.set_xlim(round((1000/fps)*(-limits[0])), round((1000/fps)*(limits[1])))
    ax5.set_ylim(10,70)

    ax6.set_ylabel('distance (pixels)')
    ax6.set_xlabel('time since stim (ms)')
    ax6.set_xlim(round((1000/fps)*(-limits[0])), round((1000/fps)*(limits[1])))
    ax6.set_ylim(0,70)

    ax7.set_ylabel('distance (pixels)')
    ax7.set_xlabel('time since stim (ms)')
    ax7.set_xlim(round((1000/fps)*(-limits[0])), round((1000/fps)*(limits[1])))
    ax7.set_ylim(0,70)


    ax8.set_ylabel('distance (pixels)')
    ax8.set_xlabel('time since stim (ms)')
    ax8.set_xlim(round((1000/fps)*(-limits[0])), round((1000/fps)*(limits[1])))
    ax8.set_ylim(0,70)

    return

#-----------------------------------------------------------------------------------
# Plots velocity of the reaches by stim vs. ctrl reach type
#-----------------------------------------------------------------------------------


def plot_velocity(reaches, limits, effector):

        input = reaches.copy()
        fps = 120; limits = [round(x/(1000/fps)) for x in limits]
        input.index = input.index.droplevel(0)
        input = input.loc[effector]

        ctrl_color = 'k'; stim_color = 'r'; stim_line = 'g'
        avgLineThickness = 1.5; alpha_shade = 0.25

        emptyList = list(np.repeat(0,(limits[0]+limits[1])))
        emptyRange = list(range(0,(limits[0]+limits[1])))

        ctrl_trim = input[input.reach_type == 0]
        n_ctrl_trials = len(ctrl_trim.trial.unique())
        stim_trim = input[input.reach_type == 1]
        n_stim_trials = len(stim_trim.trial.unique())

        ctrl_plot = pd.DataFrame({'frame':emptyRange,'time':emptyRange,'num':emptyList,'pct':emptyList,
                                  'velocity':emptyList,'velocity_lower':emptyList,'velocity_upper':emptyList,
                                  'velocity_interp':emptyList,'velocity_lower_interp':emptyList,'velocity_upper_interp':emptyList})

        stim_plot = pd.DataFrame({'frame':emptyRange,'time':emptyRange,'num':emptyList,'pct':emptyList,
                                  'velocity':emptyList,'velocity_lower':emptyList,'velocity_upper':emptyList,
                                  'velocity_interp':emptyList,'velocity_lower_interp':emptyList,'velocity_upper_interp':emptyList})

        for frame in ctrl_trim.frame.unique():

            df = ctrl_trim.loc[ctrl_trim.frame == frame]
            df = df.dropna(subset=['velocity'])
            
            ctrl_plot.loc[frame, 'num'] = len(df)
            ctrl_plot.loc[frame, 'pct'] = 100*len(df)/n_ctrl_trials
            ctrl_plot.loc[frame, 'velocity'] = df.velocity.mean()
            ctrl_plot.loc[frame, 'velocity_lower'] = df.velocity.mean()-df.velocity.std()/((ctrl_plot.num[frame])**0.5)
            ctrl_plot.loc[frame, 'velocity_upper']= df.velocity.mean()+df.velocity.std()/((ctrl_plot.num[frame])**0.5)
            
            df = ctrl_trim.loc[ctrl_trim.frame == frame]
            df = df.dropna(subset=['velocity_interp'])
        
            ctrl_plot.loc[frame, 'num_interp'] = len(df)
            ctrl_plot.loc[frame, 'pct_interp'] = 100*len(df)/n_ctrl_trials
            ctrl_plot.loc[frame, 'velocity_interp'] = df.velocity_interp.mean()
            ctrl_plot.loc[frame, 'velocity_lower_interp'] = df.velocity_interp.mean()-df.velocity_interp.std()/((ctrl_plot.num_interp[frame])**0.5)
            ctrl_plot.loc[frame, 'velocity_upper_interp']= df.velocity_interp.mean()+df.velocity_interp.std()/((ctrl_plot.num_interp[frame])**0.5)
            
            ctrl_plot.loc[frame, 'time']= (1000/fps)*(frame-limits[0])

        for frame in stim_trim.frame.unique():
            
            df = stim_trim.loc[stim_trim.frame == frame]
            df = df.dropna(subset=['velocity'])
            
            stim_plot.loc[frame, 'num'] = len(df)
            stim_plot.loc[frame, 'pct'] = 100*len(df)/n_stim_trials
            stim_plot.loc[frame, 'velocity'] = df.velocity.mean()
            stim_plot.loc[frame, 'velocity_lower'] = df.velocity.mean()-df.velocity.std()/((ctrl_plot.num[frame])**0.5)
            stim_plot.loc[frame, 'velocity_upper']= df.velocity.mean()+df.velocity.std()/((ctrl_plot.num[frame])**0.5)
            
            df = stim_trim.loc[stim_trim.frame == frame]
            df = df.dropna(subset=['velocity_interp'])
            stim_plot.loc[frame, 'num_interp'] = len(df)
            stim_plot.loc[frame, 'pct_interp'] = 100*len(df)/n_stim_trials
            stim_plot.loc[frame, 'velocity_interp'] = df.velocity_interp.mean()
            stim_plot.loc[frame, 'velocity_lower_interp'] = df.velocity_interp.mean()-df.velocity_interp.std()/((stim_plot.num_interp[frame])**0.5)
            stim_plot.loc[frame, 'velocity_upper_interp']= df.velocity_interp.mean()+df.velocity_interp.std()/((stim_plot.num_interp[frame])**0.5)
            
            stim_plot.loc[frame, 'time']= (1000/fps)*(frame-limits[0])

        ctrl_plot = ctrl_plot[ctrl_plot.frame > 0]
        stim_plot = stim_plot[stim_plot.frame > 0]

        # velocity plot
        #fig, (ax1, ax3)= plt.subplots(2, figsize=(14,14))
        fig, (ax1)= plt.subplots(1, figsize=(12,9))


        fig.tight_layout(pad=5)

        ax1.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)
        #ax3.axvspan(0, 50, facecolor= stim_line, alpha=alpha_shade)

        ax1.set_xlabel('time since stim (ms)')
        ax1.set_ylabel('velocity (pixels/s)')
        #ax3.set_xlabel('time since stim (ms)')
        #ax3.set_ylabel('velocity (pixels/s)')

        ax1.plot('time', 'velocity', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
        ax1.plot('time', 'velocity_lower', data=ctrl_plot, color=ctrl_color, linewidth=0)
        ax1.plot('time', 'velocity_upper', data=ctrl_plot, color=ctrl_color, linewidth=0)

        #ax3.plot('time', 'velocity_interp', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
        #ax3.plot('time', 'velocity_lower_interp', data=ctrl_plot, color=ctrl_color, linewidth=0)
        #ax3.plot('time', 'velocity_upper_interp', data=ctrl_plot, color=ctrl_color, linewidth=0)

        ax1.plot('time', 'velocity', data=stim_plot, color=stim_color, linewidth=avgLineThickness)
        ax1.plot('time', 'velocity_lower', data=stim_plot, color=stim_color, linewidth=0)
        ax1.plot('time', 'velocity_upper', data=stim_plot, color=stim_color, linewidth=0)

        #ax3.plot('time', 'velocity_interp', data=stim_plot, color=stim_color, linewidth=avgLineThickness)
        #ax3.plot('time', 'velocity_lower_interp', data=stim_plot, color=stim_color, linewidth=0)
        #ax3.plot('time', 'velocity_upper_interp', data=stim_plot, color=stim_color, linewidth=0)

        ax1.fill_between(stim_plot.time, stim_plot.velocity_lower, stim_plot.velocity_upper,
                         color=stim_color, alpha = alpha_shade, lw=0)
        ax1.fill_between(ctrl_plot.time, ctrl_plot.velocity_lower, ctrl_plot.velocity_upper,
                         color=ctrl_color, alpha = alpha_shade, lw=0)

        #ax3.fill_between(stim_plot.time, stim_plot.velocity_lower_interp,
        #                 stim_plot.velocity_upper_interp, color=stim_color, alpha = alpha_shade, lw=0)
        #ax3.fill_between(ctrl_plot.time, ctrl_plot.velocity_lower_interp,
        #                 ctrl_plot.velocity_upper_interp, color=ctrl_color, alpha = alpha_shade, lw=0)

        ax2 = plt.twinx(ax1)
        #ax4 = plt.twinx(ax3)

        ax2.set_ylabel('% total data')
        ax2.plot('time', 'pct', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness*.20)
        ax2.plot('time', 'pct', data=stim_plot, color=stim_color, linewidth=avgLineThickness*.20)
        ax2.grid(False)
        ax2.set_ylim(0,100)
        
        #ax4.set_ylabel('% total data')
        #ax4.plot('time', 'pct_interp', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness*.20)
        #ax4.plot('time', 'pct_interp', data=stim_plot, color=stim_color, linewidth=avgLineThickness*.20)
        #ax4.grid(False)
        #ax4.set_ylim(0,100)
        
        ax1.set_title(('velocity of ' + effector + '\n stim n = ' +  str(n_stim_trials) +
                  ' ctrl n = ' +  str(n_ctrl_trials)), pad=10)
        #ax3.set_title(('interpolated velocity of ' + effector + '\n stim n = ' + str(n_stim_trials) +
        #          ' ctrl n = ' +  str(n_ctrl_trials)), pad=10)
                      
        return

#-----------------------------------------------------------------------------------
# Plots the x-y position of the average reach by stim vs. ctrl reach type
#-----------------------------------------------------------------------------------

def plot_position(reaches, before_threshold, after_threshold, effector):
    
    input = reaches.copy()
    fps = 120
    before_threshold = round(before_threshold/(1000/fps)) # converts the ms input into frames
    after_threshold = round(after_threshold/(1000/fps))

    ctrl_color = 'k'
    stim_color = 'r'
    stim_line = 'g'
    avgLineThickness = 1.5
    alphaShade = 0.25

    emptyList = list(np.repeat(0,(before_threshold+after_threshold)))
    emptyRange = list(range(0,(before_threshold+after_threshold)))

    ctrl_trim = input[input.reach_type == 0].loc[effector]
    stim_trim = input[input.reach_type == 1].loc[effector]

    ctrl_plot = pd.DataFrame({'frame':emptyRange,'time':emptyRange,'num':emptyList,
                              'x':emptyList,'x_lower':emptyList,'x_upper':emptyList,
                              'y':emptyList,'y_lower':emptyList,'y_upper':emptyList,
                              'x_interp':emptyList,'x_lower_interp':emptyList,'x_upper_interp':emptyList,
                              'y_interp':emptyList,'y_lower_interp':emptyList,'y_upper_interp':emptyList})

    stim_plot = pd.DataFrame({'frame':emptyRange,'time':emptyRange,'num':emptyList,
                              'x':emptyList,'x_lower':emptyList,'x_upper':emptyList,
                              'y':emptyList,'y_lower':emptyList,'y_upper':emptyList,
                              'x_interp':emptyList,'x_lower_interp':emptyList,'x_upper_interp':emptyList,
                              'y_interp':emptyList,'y_lower_interp':emptyList,'y_upper_interp':emptyList})

    for frame in ctrl_trim.frame.unique():

        
        df = ctrl_trim.loc[ctrl_trim.frame == frame]
        temp = df.loc[df.x > 0]
        temp_interp = df.loc[df.x_interp > 0]
        
        ctrl_plot.loc[frame, 'num'] = len(temp)
        ctrl_plot.loc[frame, 'pct'] = 100*len(temp)/len(df)
        ctrl_plot.loc[frame, 'x'] = temp.loc[:,'x'].mean()
        ctrl_plot.loc[frame, 'y'] = temp.loc[:,'y'].mean()
        ctrl_plot.loc[frame, 'x_interp'] = temp_interp.loc[:,'x_interp'].mean()
        ctrl_plot.loc[frame, 'y_interp'] = temp_interp.loc[:,'y_interp'].mean()
        ctrl_plot.loc[frame, 'x_lower'] = temp.loc[:,'x'].mean()-temp.loc[:,'x'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'x_upper']= temp.loc[:,'x'].mean()+temp.loc[:,'x'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'y_lower'] = temp.loc[:,'y'].mean()-temp.loc[:,'y'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'y_upper']= temp.loc[:,'y'].mean()+temp.loc[:,'y'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'x_lower_interp'] = temp_interp.loc[:,'x_interp'].mean()-temp_interp.loc[:,'x_interp'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'x_upper_interp']= temp_interp.loc[:,'x_interp'].mean()+temp_interp.loc[:,'x_interp'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'y_lower_interp'] = temp_interp.loc[:,'y_interp'].mean()-temp_interp.loc[:,'y_interp'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'y_upper_interp']= temp_interp.loc[:,'y_interp'].mean()+temp_interp.loc[:,'y_interp'].std()/((ctrl_plot.num[frame])**0.5)
        ctrl_plot.loc[frame, 'time']= (1000/fps)*(frame-before_threshold)

    for frame in stim_trim.frame.unique():
        
        df = stim_trim.loc[stim_trim.frame == frame]
        temp = df.loc[df.x > 0]
        temp_interp = df.loc[df.x_interp > 0]
        
        stim_plot.loc[frame, 'num'] = len(temp)
        stim_plot.loc[frame, 'pct'] = 100*len(temp)/len(df)
        stim_plot.loc[frame, 'x'] = temp.loc[:,'x'].mean()
        stim_plot.loc[frame, 'y'] = temp.loc[:,'y'].mean()
        stim_plot.loc[frame, 'x_interp'] = temp_interp.loc[:,'x_interp'].mean()
        stim_plot.loc[frame, 'y_interp'] = temp_interp.loc[:,'y_interp'].mean()
        stim_plot.loc[frame, 'x_lower'] = temp.loc[:,'x'].mean()-temp.loc[:,'x'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'x_upper']= temp.loc[:,'x'].mean()+temp.loc[:,'x'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'y_lower'] = temp.loc[:,'y'].mean()-temp.loc[:,'y'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'y_upper']= temp.loc[:,'y'].mean()+temp.loc[:,'y'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'x_lower_interp'] = temp_interp.loc[:,'x_interp'].mean()-temp_interp.loc[:,'x_interp'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'x_upper_interp']= temp_interp.loc[:,'x_interp'].mean()+temp_interp.loc[:,'x_interp'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'y_lower_interp'] = temp_interp.loc[:,'y_interp'].mean()-temp_interp.loc[:,'y_interp'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'y_upper_interp']= temp_interp.loc[:,'y_interp'].mean()+temp_interp.loc[:,'y_interp'].std()/((ctrl_plot.num[frame])**0.5)
        stim_plot.loc[frame, 'time']= (1000/fps)*(frame-before_threshold)

    # position plot
    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3,2, figsize=(16,12))

    ax1.axvline(x=0, c = stim_line, lw=3, alpha=0.5)
    ax2.axvline(x=0, c = stim_line, lw=3, alpha=0.5)
    ax4.axvline(x=0, c = stim_line, lw=3, alpha=0.5)
    ax5.axvline(x=0, c = stim_line, lw=3, alpha=0.5)

    ax1.set_xlabel('time since stim (ms)')
    ax2.set_xlabel('time since stim (ms)')
    ax3.set_xlabel('x-position')
    ax4.set_xlabel('time since stim (ms)')
    ax5.set_xlabel('time since stim (ms)')
    ax6.set_xlabel('x-position')

    ax1.set_ylabel('x-position')
    ax2.set_ylabel('y-position')
    ax3.set_ylabel('y-position')
    ax4.set_ylabel('x-position')
    ax5.set_ylabel('y-position')
    ax6.set_ylabel('y-position')

    ax1.plot('time', 'x', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax1.plot('time', 'x', data=stim_plot, color=stim_color, linewidth=avgLineThickness)

    ax2.plot('time', 'y', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax2.plot('time', 'y', data=stim_plot, color=stim_color, linewidth=avgLineThickness)

    ax3.plot('x', 'y', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax3.plot('x', 'y', data=stim_plot, color=stim_color, linewidth=avgLineThickness)

    ax4.plot('time', 'x_interp', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax4.plot('time', 'x_interp', data=stim_plot, color=stim_color, linewidth=avgLineThickness)

    ax5.plot('time', 'y_interp', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax5.plot('time', 'y_interp', data=stim_plot, color=stim_color, linewidth=avgLineThickness)

    ax6.plot('x_interp', 'y_interp', data=ctrl_plot, color=ctrl_color, linewidth=avgLineThickness)
    ax6.plot('x_interp', 'y_interp', data=stim_plot, color=stim_color, linewidth=avgLineThickness)

    ax1.fill_between(stim_plot.time, stim_plot.x_lower, stim_plot.x_upper, color=stim_color, alpha=alphaShade, lw=0)
    ax1.fill_between(ctrl_plot.time, ctrl_plot.x_lower, ctrl_plot.x_upper, color=ctrl_color, alpha=alphaShade, lw=0)

    ax2.fill_between(stim_plot.time, stim_plot.y_lower, stim_plot.y_upper, color=stim_color, alpha=alphaShade, lw=0)
    ax2.fill_between(ctrl_plot.time, ctrl_plot.y_lower, ctrl_plot.y_upper, color=ctrl_color, alpha=alphaShade, lw=0)

    ax4.fill_between(stim_plot.time, stim_plot.x_lower_interp, stim_plot.x_upper_interp, color=stim_color, alpha=alphaShade, lw=0)
    ax4.fill_between(ctrl_plot.time, ctrl_plot.x_lower_interp, ctrl_plot.x_upper_interp, color=ctrl_color, alpha=alphaShade, lw=0)

    ax5.fill_between(stim_plot.time, stim_plot.y_lower_interp, stim_plot.y_upper_interp, color=stim_color, alpha=alphaShade, lw=0)
    ax5.fill_between(ctrl_plot.time, ctrl_plot.y_lower_interp, ctrl_plot.y_upper_interp, color=ctrl_color, alpha=alphaShade, lw=0)

    ax1.set_title(('Position ' + effector), pad=20)

    return

#-----------------------------------------------------------------------------------
# The several functions below are for the video clipping scripts
#-----------------------------------------------------------------------------------

def run_bash(command):
    os.system(command)
        
#---------------------------------------------

def crop(input,output,start,duration):
    string = "ffmpeg "  + '-i ' + input + " -ss " + str(start) + " -t " + str(duration) + " -c copy " + output
    print (string)
    run_bash(string)
    
#---------------------------------------------

def scale_speed(input,output, fps, scale):
    string = "ffmpeg -i " + input + " -filter:v \"setpts=" + str(scale) + "*PTS\" -c:v mpeg4 -q:v 2 -an -r " + (str(fps/scale) + " ") + output
    print (string)
    run_bash(string)
    
#---------------------------------------------

def extract_frames(movie, times, imgdir, session, reach_type):

    clip = VideoFileClip(movie)

    for t in times:
        imgpath = os.path.join(imgdir, '{}.png'.format(session + '_' + str(int(times[0])) + '_' + str(reach_type) +
                                                       '_' + str(int(round((t-times[0])*1000, 0))) + 'ms'))
        clip.save_frame(imgpath, t)
        
#---------------------------------------------

def maximum(reaches, parameter, effector):
    
    input = reaches.copy()
    fps = 120
    if effector == 'nan':
        session_list = input.index.get_level_values(0).unique()
    else:
        session_list = input.index.get_level_values(1).unique()

    s=0
    for session in session_list:
        if effector == 'nan':
            ef = input.loc[session]
        else:
            ef = input.loc[effector].loc[session]
            
        t=0
        for trial in ef.trial.unique():
            
            trial_df = ef[['reach_type', 'trial', parameter]]
            trial_df = trial_df[trial_df.trial==trial]
            trial_vector = trial_df.loc[:, parameter]
            trial_vector = trial_vector.replace({None:np.nan})
            
            if(sum(pd.isna(trial_vector)) < len(trial_vector)):
                trial_df['maximum'] =  np.nanmax(trial_vector)
                #trial_df['maximum'] =  trial_vector[round(200/(1000/fps),0)]
                trial_df['maximum_idx'] = list(trial_vector).index(np.nanmax(trial_vector))
                trial_df = pd.DataFrame(trial_df.loc[0]).T

                # concatenate trial dataframes
                t=t+1
                if t == 1:
                    session_df = pd.concat([trial_df], keys=[session], names=['session'], sort=True)
                elif t > 1:
                    temp = pd.concat([trial_df], keys=[session],names=['session'], sort=True)
                    concat = [session_df, temp]
                    session_df = pd.concat(concat, sort=True)
                
        # concatenate session dataframes
        s=s+1
        if s == 1:
            output = pd.concat([session_df], sort=True)
        elif s > 1:
            temp = pd.concat([session_df], sort=True)
            concat = [output, temp]
            output = pd.concat(concat, sort=True)
                
    return(output)
