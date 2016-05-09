import pickle
import copy

import numpy as np
from scipy import signal, stats
import scipy.ndimage.filters as filters

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.close('all')

def analyze_conv_activity(ts, signal, pks, ix=0, active_pk=[-1,-1,-1], verbose=4, tonic_pts=[]):
    """
    
    Iterative function that classifies peaks and molehills
    
    ts            Array of time points corresponding to signal
    
    signal        Continuous convolution signal based on activity of channel
                  Optimal signal is spike times convolved with exponential decay
                  function smoothed with Gaussian kernel.
                  
    pks           Array of 3-element arrays: [[trough, pk, trough]]
    
    """
    
    ## PARAMETER DEFINITIONS
    # peak ratio burst threshold
    burst_threshold = 3.2    
    
    # burst on/offset lims
    burst_perc_pk = 0.5
    
    # peak ratio tonic threshold
    tonic_threshold = 0.5
    
    # tonic min perc to maintain
    tonic_min_thresh = 0.3
    
    # tonic activity std pk height limit
    tonic_std_pk_thresh = .8
    
    # tonic activity std inter peak interval limit
    tonic_std_ipi_thresh = 6.

    
    ## BEGIN ANALYSIS SCRIPT
    #bursts = []
    #tonics = []
    
    flag_burst = False
    flag_tonic = False


    tgh_ix_prior, pk_ix, tgh_ix_post = pks[ix]
    
    
    cur_base_width = (ts[tgh_ix_post] - ts[tgh_ix_prior])/1000.
    cur_height_prior = signal[pk_ix] - signal[tgh_ix_prior]
    cur_height_post = signal[pk_ix] - signal[tgh_ix_post]    
    if verbose > 0:
        print "\nPeak @ %.3f: %.3f | %.3f (%.3f, %.3f, %.3f)" % (ts[pk_ix]/1000., cur_height_prior/cur_base_width, cur_height_post/cur_base_width, cur_height_prior, cur_height_post, cur_base_width)    

    if verbose > 3:
        print "Len tonic_pts: %i" % len(tonic_pts)
    
    burst_onset_t = -1
    burst_offset_t = burst_onset_t
    
    max_pk = pk_ix
    active = copy.copy(active_pk)
    
    if verbose > 4:
        print active_pk

    if active_pk == [-1,-1,-1]:
        if verbose > 2:
            print "Analysis times: %.3f -> %.3f (%.3f)" % (ts[tgh_ix_prior]/1000., ts[tgh_ix_post]/1000., signal[pk_ix])
        
        active_base_width = 1.
        active_height_prior = 0.
        active_height_post = 0.
        
        last_base_width = 1.
        last_height_prior = 0.
        last_height_post = 0.
        
    else:
        if verbose > 2:
            print "Analysis times: %.3f -> %.3f (%.3f)" % (ts[active_pk[0]]/1000., ts[tgh_ix_post]/1000., signal[pk_ix])
        
        active_pk_arg = np.argmax([signal[active_pk[1]], signal[pk_ix]])
        if active_pk_arg == 1:
            active[1] = pk_ix
            
        active_base_width = (ts[tgh_ix_post] - ts[active[0]])/1000.
        active_height_prior = signal[active[1]] - signal[active[0]]
        active_height_post = signal[active[1]] - signal[tgh_ix_post]
        
        last_base_width = (ts[active_pk[2]] - ts[active_pk[0]])/1000.
        last_height_prior = signal[active_pk[1]] - signal[active_pk[0]]
        last_height_post = signal[active_pk[1]] - signal[active_pk[2]]
        
            
    if verbose > 2:
        print "Analysis Values: %.3f | %.3f (%.3f, %.3f, %.3f)\n" % ( \
            active_height_prior/active_base_width, \
            active_height_post/active_base_width, \
            active_height_prior, \
            active_height_post, \
            active_base_width )    
    
    ##
    ## ===== Single peak rise =====
    ## Peak rises above burst threshold
    ##
    if cur_height_prior/cur_base_width >= burst_threshold:
        ##
        ## ===== Single peak rise and fall =====
        ## Peak falls below burst threshold
        ##
        if cur_height_post/cur_base_width >= burst_threshold:
            if active_pk == [-1,-1,-1]:
                if verbose > 1:
                    print "Single peak burst!!"
                
                flag_burst = True
                active = [tgh_ix_prior, pk_ix, tgh_ix_post]
                tonic_pts.append(pk_ix)
                tonic_pts.append(tgh_ix_post)
            
            # If previous peak falling ratio is less than burst threshold, test for multi peak burst 
            # If it's not, this indicates the end of the previous burst
            elif last_height_post/last_base_width < burst_threshold and last_height_prior > last_height_post:
                if active_height_prior/active_base_width >= burst_threshold and active_height_post/active_base_width >= burst_threshold:
                    if verbose > 1:
                        print "Multi peak burst!!"
                    
                    flag_burst = True
                    active[2] = tgh_ix_post
                    tonic_pts.append(pk_ix)
                    tonic_pts.append(tgh_ix_post)
            else:
                if verbose > 1:
                    print "New burst!!"
        
        ##
        ## ===== Rising molehill? =====
        ## Single peak rise, but does not fall below threshold
        ##
        else:
            if verbose > 1:
                print "Rising molehill?"
               
            if active == [-1,-1,-1]:
                if verbose > 1:
                    print "Rising molehill!!"
                active = [tgh_ix_prior, pk_ix, tgh_ix_post]
                                
                tonic_pts.append(pk_ix)
                tonic_pts.append(tgh_ix_post)                
    
            # If previous peak falling ratio is less than burst threshold, test for multi peak burst
            # If it's greater, this indicates the end of the previous burst
            elif last_height_post/last_base_width < burst_threshold:
                if active_height_prior/active_base_width >= burst_threshold and active_height_post/active_base_width >= burst_threshold:
                    if verbose > 1:
                        print "Multi peak burst with rising molehill!!"
                    
                    flag_burst = True
                    active[2] = tgh_ix_post
                    tonic_pts.append(pk_ix)
                    tonic_pts.append(tgh_ix_post)
    
        
    ##
    ## ===== Falling molehill? =====
    ## Peak does not rise above burst threshold
    ##        
    else:
        ##
        ## ===== Falling molehill =====
        ## Peak does not rise ABOVE burst threshold, but peak falls BELOW burst threshold
        ##
        if cur_height_post/cur_base_width >= burst_threshold:
            if verbose > 1:
                print "Falling molehill?"
                
            # Falling molehills cannot be the first peak in a sequence
            if active <> [-1,-1,-1]:
                # If rising AND falling ratios for ACTIVE peak sequence is greater than burst threshold, multi peak burst!
                if active_height_prior/active_base_width >= burst_threshold and active_height_post/active_base_width >= burst_threshold:
                    if verbose > 1:
                        print "Multi peak burst with falling molehill!!"
                    
                    active[2] = tgh_ix_post
                    flag_burst = True
                else:
                    if verbose > 1:
                        print "End of tonic activity!!"
                    active[2] = tgh_ix_post
                    flag_tonic = True
          
        ##
        ## Small molehills and tonic activity
        ## 
        ##
        else:
            # Check if small falling molehill on falling side of burst
            if cur_height_prior < cur_height_post:
                if verbose > 1:
                    print "Small falling molehill!!"
                
                if active <> [-1,-1,-1]:
                    if active_height_prior/active_base_width >= burst_threshold and active_height_post/active_base_width >= burst_threshold:
                        if verbose > 1:
                            print "Multi peak burst with small falling molehill!!"
                        
                        flag_burst = True
                        
                        active[2] = tgh_ix_post
                        
            # Check if small rising molehill on rising side of burst
            if cur_height_post < cur_height_prior:
                if verbose > 1:
                    print "Small rising molehill!!"
                    
                if active <> [-1,-1,-1]:
                    #if active_height_prior/active_base_width >= burst_threshold and active_height_post/active_base_width >= burst_threshold:
                    if active_height_prior/active_base_width >= burst_threshold:
                        if verbose > 1:
                            print "Multi peak burst with small rising molehill!!"
                            
                        flag_burst = True
                        
                        active[2] = tgh_ix_post
                        
                        tonic_pts.append(pk_ix)
                        tonic_pts.append(tgh_ix_post)
                                
                else:
                    if signal[pk_ix]/cur_base_width >= tonic_threshold:
                        if verbose > 2:
                            print "Tonic window: %.3f >?= %.3f (Code 2)" % (signal[tgh_ix_post], cur_height_prior*tonic_min_thresh + signal[tgh_ix_prior])
                        
                        if signal[tgh_ix_post] >= cur_height_prior*tonic_min_thresh + signal[tgh_ix_prior]:
                            active = [tgh_ix_prior, pk_ix, tgh_ix_post]
                            
                            tonic_pts.append(pk_ix)
                            tonic_pts.append(tgh_ix_post)
            
            # If not a burst and the last peak was not a single peak burst...
            if not flag_burst:
                if verbose > 3:
                    print "No burst. Checking for tonic activity..."

                if active_pk <> [-1,-1,-1]:
                    if verbose > 2:
                        print "Tonic window: %.3f >?= %.3f (Code 1)" % (signal[tgh_ix_post], active_height_prior*tonic_min_thresh + signal[active[0]])             
                    
                    tonic_pts.append(pk_ix)                
                    tonic_pts.append(tgh_ix_post)                     
                    
                    if verbose > 2:
                        print "STD @ %.3f <?= %.3f " % (np.std(np.array(signal)[np.array(tonic_pts)]), tonic_std_pk_thresh)                    
                    if np.std(np.array(signal)[np.array(tonic_pts)]) <= tonic_std_pk_thresh:
                        if verbose > 2:
                            print "Is it still tonic?! --> %.3f s" % ((ts[active[2]] - ts[active[0]])/1000.)                    
                    
                        if verbose > 5:
                            print "Signal points:"
                            print np.array(signal)[tonic_pts]
                            print np.array(signal)[tonic_pts][:-2]
                            print "Peak time points:"
                            print np.array(ts)[tonic_pts][:-2]/1000.
                            print np.diff(np.array(ts)[tonic_pts][:-2])/1000.
                        if verbose > 3:
                            print "Avg Inter-Peak Int: %.3f +/- %.3f" % (np.average(np.diff(np.array(ts)[tonic_pts][:-2]))/1000., tonic_std_ipi_thresh*np.std(np.diff(np.array(ts)[tonic_pts[:-2]]))/1000.)
                            
                        tonic_ipi_avg = np.average(np.diff(np.array(ts)[tonic_pts][:-2]))/1000.
                        tonic_ipi_std = np.std(np.diff(np.array(ts)[tonic_pts][:-2]))/1000.                                                                      
                        
                        if verbose > 3:
                            print "Current Inter-Peak Int: %.3f" % ((ts[tonic_pts[-2]] - ts[tonic_pts[-3]])/1000.)
                                
                            
                        if (ts[tonic_pts[-2]] - ts[tonic_pts[-3]])/1000. <= tonic_ipi_avg + tonic_std_ipi_thresh*tonic_ipi_std:
                                           
                            if verbose > 2:
                                print "YES!!"

                            active[2] = tgh_ix_post
                                
                            flag_tonic = True  
                        else:
                            if verbose > 2:
                                print "NO!!"
                       
                else:
                    if verbose > 2:
                        print "Tonic threshold: %.3f >?= %.3f (Code 2)" % (signal[pk_ix]/cur_base_width, tonic_threshold)
                    if signal[pk_ix]/cur_base_width >= tonic_threshold:
                        if verbose > 2:
                            print "Tonic window: %.3f >?= %.3f (Code 2)" % (signal[tgh_ix_post], cur_height_prior*tonic_min_thresh + signal[tgh_ix_prior])
                        
                        if (signal[tgh_ix_post] >= (cur_height_prior*tonic_min_thresh + signal[tgh_ix_prior]) and cur_height_prior/cur_base_width <= burst_threshold) or (np.abs(cur_height_prior - cur_height_post) <= 0.001):
                            
                            tonic_pts.append(pk_ix)
                            tonic_pts.append(tgh_ix_post)
                            
                            active = [tgh_ix_prior, pk_ix, tgh_ix_post]
                                
                            flag_tonic = True                        
        
                            if verbose > 2:
                                print "Tonic onset? --> %.3f" % (ts[active[0]]/1000.)                    


    if verbose > 4:
        print active_pk
        print active

    if (active_pk[0] <> active[0] or active_pk[2] <> active[2]) and (ix < len(pks) - 1):
        if verbose > 2:
            if ix+1 < len(pks):
                print "\n--> Recursing in to t=%.3f!!" % (ts[pks[ix+1][1]]/1000.)
            else:
                print "\n--> Recursing in to t=%.3f!!" % (ts[-1]/1000.)
            
            
        if not (flag_burst and flag_tonic):
            activity, activity_type, ix, active_pk_rec, tonic_pts_rec = analyze_conv_activity(ts, signal, pks, ix+1, active, verbose=verbose, tonic_pts=tonic_pts)
        
        if activity_type == 'burst' or (activity_type == 'tonic' and not flag_burst):
            active = active_pk_rec
            
        if activity_type == 'burst':
            flag_burst = True
            
        if activity_type == 'tonic' and flag_tonic and not flag_burst:
            #print "Recursion = Tonic!!"
            flag_tonic = True
            
        elif activity_type == 'tonic' and not flag_tonic and not flag_burst:
            #print "Onset of tonic!!"
            active[0] = tgh_ix_prior
            active[1] = pk_ix
            flag_tonic = True
        
        if verbose > 2:
            print "\n--> --> Done Recursing at 5=%.3f..." % (ts[pk_ix]/1000.)
            
        if verbose >2:
            print "Activity Region: %.3f --> %.3f" % (ts[active[0]]/1000., ts[active[2]]/1000.)
        if verbose > 4:
            print active
    #elif (active_pk[0] <> active[0] or active_pk[2] <> active[2]) and (ix == len(pks) - 1):
        #activity_type = 'none'
        #if flag_burst:
            #activity_type = 'burst'
        #elif flag_tonic:
            #activity_type = 'tonic'        

        

    if verbose > 4:
            print active_pk
            print active

    onset_t = -1
    offset_t = onset_t
    
    if flag_burst:
        if verbose > 3:
            print "Rising threshold: %.3f" % ((signal[active[1]] - signal[active[0]])*burst_perc_pk + signal[active[0]])
        # Determine burst onset
        lims_on_ix = np.where(signal[active[0]:active[2]] >= (signal[active[1]] - signal[active[0]])*burst_perc_pk + signal[active[0]])[0] + active[0]
        # Determine burst onset spike
        onset_t = data[analysisChan]['data'][np.where(data[analysisChan]['data'] >= ts[lims_on_ix[0]]/1000.)[0][0]]
        
        if verbose > 3:
            print "Falling threshold: %.3f" % ((signal[active[1]] - signal[active[2]])*burst_perc_pk + signal[active[2]])
        # Determine burst offset            
        lims_off_ix = np.where(signal[active[0]:active[2]] >= (signal[active[1]] - signal[active[2]])*burst_perc_pk + signal[active[2]])[0] + active[0]
        # Determine burst offset spike
        offset_t = data[analysisChan]['data'][np.where(data[analysisChan]['data'] <= ts[lims_off_ix[-1]]/1000.)[0][-1]]                
        
        if verbose > 2:
            print "Activity from %.3f to %.3f" % (onset_t, offset_t)
            
    if flag_tonic and not flag_burst:
        # Determine burst onset spike
        onset_t = data[analysisChan]['data'][np.where(data[analysisChan]['data'] >= ts[active[0]]/1000.)[0][0]]

        # Determine burst offset spike
        offset_t = data[analysisChan]['data'][np.where(data[analysisChan]['data'] <= ts[active[2]]/1000.)[0][-1]]                
        
        if verbose > 2:
            print "Activity from %.3f to %.3f" % (onset_t, offset_t)    

    activity_type = 'none'
    if flag_burst:
        activity_type = 'burst'
    elif flag_tonic:
        activity_type = 'tonic'

    return ([onset_t, offset_t], activity_type, ix, active, tonic_pts)
        
        


files = ['FinalDissertationModel_Standalone-00054.dat',
 'FinalDissertationModel_Standalone-00247.dat',
 'FinalDissertationModel_Standalone-00288.dat',
 'FinalDissertationModel_Standalone-00329.dat',
 'FinalDissertationModel_Standalone-00490.dat',
 'FinalDissertationModel_Standalone-00520.dat',
 'FinalDissertationModel_Standalone-00540.dat', # 6
 'FinalDissertationModel_Standalone-00742.dat',
 'FinalDissertationModel_Standalone-00793.dat',
 'FinalDissertationModel_Standalone-00815.dat',
 'FinalDissertationModel_Standalone-00825.dat',
 'FinalDissertationModel_Standalone-00847.dat',
 'FinalDissertationModel_Standalone-00871.dat',
 'FinalDissertationModel_Standalone-00901.dat',
 'FinalDissertationModel_Standalone-01242.dat', # 15
 'FinalDissertationModel_Standalone-01302.dat',
 'FinalDissertationModel_Standalone-01396.dat',
 'FinalDissertationModel_Standalone-01405.dat',
 'FinalDissertationModel_Standalone-01472.dat',
 'FinalDissertationModel_Standalone-01504.dat',
 'FinalDissertationModel_Standalone-01584.dat',
 'FinalDissertationModel_Standalone-01618.dat',
 'FinalDissertationModel_Standalone-01770.dat',
 'FinalDissertationModel_Standalone-01812.dat',
 'FinalDissertationModel_Standalone-02085.dat',
 'FinalDissertationModel_Standalone-02234.dat',
 'FinalDissertationModel_Standalone-02411.dat',
 'FinalDissertationModel_Standalone-02512.dat',
 'FinalDissertationModel_Standalone-02525.dat',
 'FinalDissertationModel_Standalone-02702.dat',
 'FinalDissertationModel_Standalone-02806.dat',
 'FinalDissertationModel_Standalone-03059.dat',
 'FinalDissertationModel_Standalone-03387.dat',
 'FinalDissertationModel_Standalone-03395.dat', # 33
 'FinalDissertationModel_Standalone-03429.dat',
 'FinalDissertationModel_Standalone-03502.dat',
 'FinalDissertationModel_Standalone-03690.dat',
 'FinalDissertationModel_Standalone-03717.dat',
 'FinalDissertationModel_Standalone-03764.dat',
 'FinalDissertationModel_Standalone-03828.dat',
 'FinalDissertationModel_Standalone-03830.dat',
 'FinalDissertationModel_Standalone-03838.dat',
 'FinalDissertationModel_Standalone-03862.dat',
 'FinalDissertationModel_Standalone-04005.dat',
 'FinalDissertationModel_Standalone-04160.dat',
 'FinalDissertationModel_Standalone-04207.dat',
 'FinalDissertationModel_Standalone-04212.dat',
 'FinalDissertationModel_Standalone-04293.dat',
 'FinalDissertationModel_Standalone-04314.dat',
 'FinalDissertationModel_Standalone-04407.dat',
 'FinalDissertationModel_Standalone-04493.dat',
 'FinalDissertationModel_Standalone-04667.dat',
 'FinalDissertationModel_Standalone-04671.dat',
 'FinalDissertationModel_Standalone-04748.dat',
 'FinalDissertationModel_Standalone-05284.dat',
 'FinalDissertationModel_Standalone-05434.dat',
 'FinalDissertationModel_Standalone-05573.dat',
 'FinalDissertationModel_Standalone-05714.dat',
 'FinalDissertationModel_Standalone-05761.dat',
 'FinalDissertationModel_Standalone-05893.dat',
 'FinalDissertationModel_Standalone-05939.dat',
 'FinalDissertationModel_Standalone-06155.dat',
 'FinalDissertationModel_Standalone-06352.dat',
 'FinalDissertationModel_Standalone-06686.dat',
 'FinalDissertationModel_Standalone-06736.dat',
 'FinalDissertationModel_Standalone-07031.dat',
 'FinalDissertationModel_Standalone-07071.dat',
 'FinalDissertationModel_Standalone-07083.dat',
 'FinalDissertationModel_Standalone-07191.dat',
 'FinalDissertationModel_Standalone-07198.dat',
 'FinalDissertationModel_Standalone-07260.dat',
 'FinalDissertationModel_Standalone-07426.dat',
 'FinalDissertationModel_Standalone-07487.dat', # 73
 'FinalDissertationModel_Standalone-07549.dat',
 'FinalDissertationModel_Standalone-07623.dat',
 'FinalDissertationModel_Standalone-07654.dat',
 'FinalDissertationModel_Standalone-07669.dat',
 'FinalDissertationModel_Standalone-07916.dat',
 'FinalDissertationModel_Standalone-07923.dat',
 'FinalDissertationModel_Standalone-08045.dat',
 'FinalDissertationModel_Standalone-08076.dat',
 'FinalDissertationModel_Standalone-08118.dat',
 'FinalDissertationModel_Standalone-08189.dat',
 'FinalDissertationModel_Standalone-08268.dat',
 'FinalDissertationModel_Standalone-08308.dat',
 'FinalDissertationModel_Standalone-08722.dat',
 'FinalDissertationModel_Standalone-08809.dat',
 'FinalDissertationModel_Standalone-08842.dat',
 'FinalDissertationModel_Standalone-08992.dat',
 'FinalDissertationModel_Standalone-09105.dat', # 90
 'FinalDissertationModel_Standalone-09474.dat',
 'FinalDissertationModel_Standalone-09646.dat',
 'FinalDissertationModel_Standalone-09748.dat',
 'FinalDissertationModel_Standalone-09905.dat',
 'FinalDissertationModel_Standalone-09965.dat',
 'FinalDissertationModel_Standalone-10253.dat',
 'FinalDissertationModel_Standalone-10305.dat',
 'FinalDissertationModel_Standalone-10425.dat',
 'FinalDissertationModel_Standalone-10758.dat']


files = [
'FinalDissertationModel_Standalone-00141.dat',
'FinalDissertationModel_Standalone-00594.dat', 
'FinalDissertationModel_Standalone-01410.dat', 
'FinalDissertationModel_Standalone-01561.dat', 
'FinalDissertationModel_Standalone-02135.dat', 
'FinalDissertationModel_Standalone-02286.dat', 
'FinalDissertationModel_Standalone-03010.dat', 
'FinalDissertationModel_Standalone-03101.dat', 
'FinalDissertationModel_Standalone-03252.dat', 
'FinalDissertationModel_Standalone-03312.dat', 
'FinalDissertationModel_Standalone-09201.dat', 
'FinalDissertationModel_Standalone-09503.dat', 
'FinalDissertationModel_Standalone-10319.dat', 
'FinalDissertationModel_Standalone-10621.dat'
]



##
## ===== ===== ===== ===== =====
##

#filename = 'FinalDissertationModel_Standalone-00141.dat'

# 68,69 Bursts have a wide base that is not reflected in burst identification -- might miss burst dynamics?

# 73 --> Rhythmic bursts

#filename = 'F:/Users/bchung/Google Drive/_Research/AnimatLabDS/Burst Detect/TestData/RandomSelection/%s' % files[1] # 6, 15, 33, 73, 90
filename = 'F:/Users/bchung/Google Drive/_Research/AnimatLabDS/Burst Detect/TestData/%s' % files[13] # 6, 15, 33, 73, 90
files = [filename]

analysisChan = 'Tonic Lev MN'

audit = True

##
## ===== ===== ===== ===== =====
##

results = {}


for filename in files:
    print "\n\nLoading data for: %s" % filename
    data = pickle.load(open(filename, 'r'))
    
    x = data[analysisChan]['data']
    
    if audit:
        print "Plotting data..."
        fig = plt.figure(figsize=(21, 15))
        fig.suptitle(filename + '\nConvolution Method', fontsize='18')
        ax_joint = fig.add_subplot(511)
        ax_lev = fig.add_subplot(512, sharex=ax_joint)
        ax_levConv = fig.add_subplot(513, sharex=ax_joint)
        ax_dep = fig.add_subplot(514, sharex=ax_joint)
        ax_depConv = fig.add_subplot(515, sharex=ax_joint)
        
        ax_joint.plot(data['Time']['data'], data['CB_joint']['data'], 'g-')
        ax_lev.vlines(data['Phasic Lev  MN']['data'], ymin=0.52, ymax=0.8, color='c')
        ax_lev.vlines(data['Tonic Lev MN']['data'], ymin=0.2, ymax=0.48, color='b')
        ax_lev.set_ylim([0,1])
        ax_dep.vlines(data['Phasic Dep MN']['data'], ymin=0.52, ymax=0.8, color='m')
        ax_dep.vlines(data['Tonic Dep MN']['data'], ymin=0.2, ymax=0.48, color='r')
        ax_dep.set_ylim([0,1])
        
        fig.canvas.draw()
    
    
    
    print "Calculating bursts..."
    
    ##
    ## ===== ===== ===== ===== =====
    ##
    
    iter_verbose = 0
    gen_verbose = iter_verbose
    
    # Convolution array is in steps of 0.5 ms -- equal to neural time step
    # tau [ms]
    tau = 0.0005
    amp = 1.
    
    # step [ms]
    step = 0.5
    
    # width [ms]
    widthConv = 900
    widthGauss = 700
    
    # trough search
    tgh_min = 0.1
    
    # tonic activity onset
    min_tonic_len = 10 # 10
    
    # tonic minimum ratio
    min_tonic_ratio = 0.6
    
    
    ##
    ## ===== ===== ===== ===== =====
    ##
    
    convX = np.arange(0,widthConv+step, step)
    convY = np.zeros(len(convX)) + amp*np.exp(-tau*convX)
    
    #plt.figure()
    #plt.plot(convX, convY)
    #plt.title('Convolution array')
    
    # Calculate discrete spikes array
    ts = np.arange(data['Time']['data'][0]*1000., data['Time']['data'][-1]*1000., 0.5)
    if len(ts) <> len(data['Time']['data']):
        raise ValueError("Error generating times!!!")
    
    print "Analyzing Channel: Tonic Lev MN"
    ixs_tonLev = np.in1d(ts, np.around(data['Tonic Lev MN']['data']*1000., 1))
    sig_tonLev = np.zeros(len(ts))
    sig_tonLev[np.where(ixs_tonLev == True)[0]] = 1
    
    conv_tonLev = np.convolve(sig_tonLev, convY, 'same')
    smth_tonLev = filters.gaussian_filter(conv_tonLev, widthGauss)
    
    pks_tonLev = np.where(np.diff(np.sign(np.diff(smth_tonLev))) == -2)[0]
    tghs_tonLev = np.where(np.diff(np.sign(np.diff(smth_tonLev))) == 2)[0]
    
    if len(pks_tonLev) > 0 and len(tghs_tonLev) > 0:
    
        if pks_tonLev[0] < tghs_tonLev[0]:
            # Find first trough OR lowest point before the first peak
            tgh_ix_prior = 0
            if smth_tonLev[0] == np.min(smth_tonLev[:pks_tonLev[0]]):
                tgh_ix_prior = np.where(np.diff(np.sign(smth_tonLev[:pks_tonLev[0]] - np.min(smth_tonLev[:pks_tonLev[0]])*tgh_min)) > 0)[0]
                if len(tgh_ix_prior) == 0:
                    tgh_ix_prior = 0
                else:
                    tgh_ix_prior = tgh_ix_prior[-1]
            tghs_tonLev = np.append(np.array([tgh_ix_prior]), tghs_tonLev, axis=0)
                
        if pks_tonLev[-1] > tghs_tonLev[-1]:
            # Find last trough OR lowest point after the last peak
            tgh_ix_post = len(smth_tonLev)-1
            if smth_tonLev[-1] == np.min(smth_tonLev[pks_tonLev[-1]:-1]):
                tgh_ix_post = np.where(np.diff(np.sign(smth_tonLev[pks_tonLev[-1]:-1] - smth_tonLev[pks_tonLev[-1]]*tgh_min)) < 0)[0] + pks_tonLev[-1]
                if len(tgh_ix_post) == 0:
                    tgh_ix_post = len(smth_tonLev)-1
                else:
                    tgh_ix_post = tgh_ix_post[0]
            tghs_tonLev = np.append(tghs_tonLev, np.array([tgh_ix_post]), axis=0)
                
              
    if audit:  
        #ax_levConv.plot(ts/1000., conv_tonLev, color='b', linestyle='--', alpha=0.3)
        ax_levConv.plot(ts/1000., smth_tonLev, 'b-')
        #ax_levConv.axhline(np.mean(smth_tonLev), color='b', linestyle='-')
        #ax_levConv.axhline(np.mean(smth_tonLev)*0.25, color='b', linestyle='--')
        ax_levConv.plot(ts[pks_tonLev]/1000., smth_tonLev[pks_tonLev], 'go', markersize=5)
        ax_levConv.plot(ts[tghs_tonLev]/1000., smth_tonLev[tghs_tonLev], 'ro', markersize=5)
        
        fig.canvas.draw()
    
    
    if len(pks_tonLev) == 0 or len(tghs_tonLev) == 0:
        continue
    
    arr_pks = np.array([])
    for ix, pk in enumerate(pks_tonLev):
        pk = pks_tonLev[ix]
        
        #ax_levConv.plot(ts[pk]/1000., smth_tonLev[pk], 'r+', markersize=15)
        
        #
        # Identify prior and following troughs
        #
        tgh_ix_prior = np.where(np.diff(np.sign(tghs_tonLev - pk)) > 0)[0]
        tgh_ix_post = tgh_ix_prior + 1
        
    
        if ix > 0:
            # Is there a second peak between the current peak and the previous trough?
            if tghs_tonLev[tgh_ix_prior] < pks_tonLev[ix-1]:
                #print "\nFinding prior trough..."
                # Find the trough before the current peak and after the previous peak
                tgh_ix_prior = np.where(np.diff(np.sign(smth_tonLev[pks_tonLev[ix-1]:pks_tonLev[ix]] - smth_tonLev[pk]*tgh_min)) > 0)[0][-1]
                tgh_ix_prior += pks_tonLev[ix-1]
                
                if audit:
                    ax_levConv.plot(ts[tgh_ix_prior]/1000., smth_tonLev[tgh_ix_prior], 'co')
            else:
                tgh_ix_prior = tghs_tonLev[tgh_ix_prior]
        else:
            tgh_ix_prior = tghs_tonLev[tgh_ix_prior]
            
    
        if ix < len(pks_tonLev) - 1:
            # Is there a second peak between the current peak and the following trough?
            if tghs_tonLev[tgh_ix_post] > pks_tonLev[ix+1]:
                #print "\nFinding post trough..."
                # Find the trough after teh current peak and before the next peak
                tgh_ix_post = np.where(np.diff(np.sign(smth_tonLev[pks_tonLev[ix]:pks_tonLev[ix+1]] - smth_tonLev[pk]*tgh_min)) < 0)[0][0]
                tgh_ix_post += pks_tonLev[ix]
    
                if audit:
                    ax_levConv.plot(ts[tgh_ix_post]/1000., smth_tonLev[tgh_ix_post], 'mo')
            else:
                tgh_ix_post = tghs_tonLev[tgh_ix_post]
        else:
            tgh_ix_post = tghs_tonLev[tgh_ix_post]
                
                        
        if ix > 0:
            if len(np.where(smth_tonLev[pks_tonLev[ix-1]:pk] == np.min(smth_tonLev[pks_tonLev[ix-1]:pk]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points before peak @ %.3f (Code 1)!! --> %.3f" % (ts[pk]/1000., ts[tgh_ix_prior]/1000.)
                tgh_ix_prior = np.where(smth_tonLev[pks_tonLev[ix-1]:pk] <= smth_tonLev[pk]*tgh_min)[0][-1] + pks_tonLev[ix-1]
                
                if gen_verbose > 2:
                    print "--> %.3f" % (ts[tgh_ix_prior]/1000.)
                
                if audit:
                    ax_levConv.plot(ts[tgh_ix_prior]/1000., smth_tonLev[tgh_ix_prior], 'co')
                
        else:
            if len(np.where(smth_tonLev[0:pk] == np.min(smth_tonLev[0:pk]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points before peak @ %.3f!! --> %.3f (Code 2)" % (ts[pk]/1000., ts[tgh_ix_prior]/1000.)            
                    tgh_ix_prior = np.where(smth_tonLev[0:pk] <= smth_tonLev[pk]*tgh_min)[0][-1]
                    
                if gen_verbose > 2:
                    print " --> %.3f" % (ts[tgh_ix_prior]/1000.)            
                
                if audit:
                    ax_levConv.plot(ts[tgh_ix_prior]/1000., smth_tonLev[tgh_ix_prior], 'co')
                
        if ix < len(pks_tonLev) - 1:
            if len(np.where(smth_tonLev[pk:pks_tonLev[ix+1]] == np.min(smth_tonLev[pk:pks_tonLev[ix+1]]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points after peak @ %.3f!! --> %.3f (Code 3)" % (ts[pk]/1000., ts[tgh_ix_post]/1000.)
                tgh_ix_post = np.where(smth_tonLev[pk:pks_tonLev[ix+1]] <= smth_tonLev[pk]*tgh_min)[0][0] + pk
                
                if gen_verbose > 2:
                    print "--> %.3f" % (ts[tgh_ix_post]/1000.)
        
                if audit:
                    ax_levConv.plot(ts[tgh_ix_post]/1000., smth_tonLev[tgh_ix_post], 'mo')
        else:
            if len(np.where(smth_tonLev[pk:-1] == np.min(smth_tonLev[pk:-1]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points after peak @ %.3f!! --> %.3f (Code 4)" % (ts[pk]/1000., ts[tgh_ix_post]/1000.)            
                tgh_ix_post = np.where(smth_tonLev[pk:-1] <= smth_tonLev[pk]*tgh_min)[0][0] + pk
                
                if gen_verbose > 2:
                    print "--> %.3f" % (ts[tgh_ix_post]/1000.)
                
                if audit:
                    ax_levConv.plot(ts[tgh_ix_post]/1000., smth_tonLev[tgh_ix_post], 'mo')
    
        try:
            tgh_ix_prior = tgh_ix_prior[0]
        except:
            pass
        
        try:
            tgh_ix_post = tgh_ix_post[0]
        except:
            pass
    
        if len(arr_pks) == 0:
            arr_pks = np.expand_dims(np.array([tgh_ix_prior, pk, tgh_ix_post]), axis=0)
        else:
            arr_pks = np.append(arr_pks, np.expand_dims(np.array([tgh_ix_prior, pk, tgh_ix_post]), axis=0), axis=0)
    
    
    bursts = []
    tonics = []
           
    ix = 0       
    while ix < len(arr_pks):
        
        if gen_verbose > 1:
            print "\n\n--> ITERATING @ t=%.3f!!" % (ts[arr_pks[ix][1]]/1000.)
        activity, activity_type, new_ix, active, tonic_pts = analyze_conv_activity(ts, smth_tonLev, arr_pks, ix, verbose=iter_verbose, tonic_pts=[])    
        
        if activity_type == 'burst':
            if gen_verbose > 2:
                print "Bursting activity!!"
            
            base_width = (ts[active[2]] - ts[active[0]])/1000.
            pre_ratio = (smth_tonLev[active[1]] - smth_tonLev[active[0]])/base_width
            post_ratio = (smth_tonLev[active[1]] - smth_tonLev[active[2]])/base_width
            
            if gen_verbose > 3:
                print "Ratios: %.3f | %.3f" % (pre_ratio, post_ratio)
            
            bursts.append(activity)
            
            if audit:
                ax_levConv.axvline(activity[0], color='c')
                ax_levConv.axvline(activity[1], color='c')
            
            
            
        elif activity_type == 'tonic':
            if gen_verbose > 2:
                print "Tonic activity!!"
                
                #print [np.array([x[0] for x in arr_pks[ix:new_ix]])]
            
            if len(tonic_pts) >= min_tonic_len:            
                pks = np.array(smth_tonLev)[np.array([x[1] for x in arr_pks[ix:new_ix]])]
                tghs_pre = np.array(smth_tonLev)[np.array([x[0] for x in arr_pks[ix:new_ix]])]
                tghs_post = np.array(smth_tonLev)[np.array([x[2] for x in arr_pks[ix:new_ix]])]
                
                hts_pre = pks - tghs_pre
                hts_post = pks - tghs_post
                
                pre_ratio = np.max(hts_pre)/(activity[1] - activity[0])
                post_ratio = np.max(hts_post)/(activity[1] - activity[0])
    
                if gen_verbose > 3:
                    print "Ratios: %.3f | %.3f" % (pre_ratio, post_ratio)
                
                if pre_ratio <= min_tonic_ratio and post_ratio <= min_tonic_ratio:
                # if len(tonics) >= min_tonic_len:
                    tonics.append(activity)
                    
                    if audit:
                        ax_levConv.axvline(activity[0], color='g', linewidth=2)
                        ax_levConv.axvline(activity[1], color='r', linewidth=2)
    
        if new_ix - ix > 1:
            if gen_verbose > 1:
                print "\n--> Done Iteration... New t = %.3f (%i, %i)" % (ts[arr_pks[new_ix][1]]/1000., ix, new_ix)
            ix = new_ix
        else:
            if ix + 1 < len(arr_pks):
                if gen_verbose > 1:
                    print "\n--> Done Iteration... New t = %.3f (%i, %i)" % (ts[arr_pks[ix+1][1]]/1000., ix, new_ix)
            ix += 1
           
    if audit:       
        # Plot bursts    
        for burst in bursts:
            ax_lev.plot(burst, [0.1, 0.1], color='b', linewidth=5)
        for tonic in tonics:
            ax_lev.plot(tonic, [0.1, 0.1], color='b', linewidth=2)
        
        
        fig.canvas.draw()
        
    if gen_verbose > 0:
        print "\nBURSTS: %i" % len(bursts)
        print bursts
        print "\nTONICS: %i" % len(tonics)
        print tonics

    results.update({filename: {'tonics': tonics, 'bursts': bursts}})
