import os, glob
import csv
import sys
sys.setrecursionlimit(1500)

import h5py

import pickle
import copy

import numpy as np
import scipy.ndimage.filters as filters

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.ion()

# If you plan to run the script multiple times and want to compare results,
# comment the following line so that the figures do not get closed after
# each run.
plt.close('all')

#from data_analysis_lib import *


## ===== ===== ===== ===== =====
## ===== ===== ===== ===== =====

def make_data_to_signal(data, kwargs={}):
    """
    make_data_to_signal
    This function takes a sequence of spike times and convolves then smooths the data
    to produce a time-continuous signal that can be processed using other functions in
    this library. The output of this function is the smooth convolved signal and an
    array of [trough, peak, trough] lists.
    
    data            Iterable of spike times | list-like
    time            Iterable of time points | list-like
    
    gen_verbose     Setting to debug or see text output | integer
    audit           Setting for visual debug | boolean
    
    timeFactor      Value used to scale data in time | float
                    1. means 1 = 1 ms
                    1000. means 1 = 1 s # DEFAULT
                    NOTE: Calculations are based on a basis of 1 = 1 ms
    
    widthConv       Width of convolution function in time units | float
    step            Step value for data points in convolution function | float
    tau             Time scale of exponential decay function | float
    amp             Amplitude of exponential decay function | float
    
    widthGauss      Width of Gaussian smoothing function | float
    
    tgh_min         Adjustment factor for ambiguous troughs | float
                    If a trough is identified in a flat segment of the smooth
                    convolved signal, the trough point is adjusted to the point
                    in the signal where it passes through tgh_min*peak_height for
                    more accurate activity classification.
    
    """   

    ##
    ## Audit parameters
    ##        
    if kwargs.has_key('gen_verbose'):
        gen_verbose = kwargs['gen_verbose']
    else:
        gen_verbose = 0
        
    if kwargs.has_key('ax_audit'):
        ax_audit = kwargs['ax_audit']
    else:
        ax_audit = None
        
    if kwargs.has_key('conv_color'):
        conv_color = kwargs['conv_color']
    else:
        conv_color = 'b'
        
    if kwargs.has_key('conv_audit'):
        conv_audit = kwargs['conv_audit']
    else:
        conv_audit = True
        
    if kwargs.has_key('save_conv'):
        conv_save = True
    else:
        conv_save = False
    
    ##
    ## Data parameters
    ##
    if kwargs.has_key('data_ymin'):
        data_ymin = kwargs['data_ymin']
    else:
        data_ymin = 0.1
        
    if kwargs.has_key('data_ymax'):
        data_ymax = kwargs['data_ymax']
    else:
        data_ymax = 0.8
    
    if kwargs.has_key('time'):
        time = kwargs['time']
    else:
        time = None
    
    if kwargs.has_key('step'):
        step = kwargs['step']
    else:
        step = 0.5  

    if kwargs.has_key('timeFactor'):
        timeFactor = kwargs['timeFactor']
    else:
        timeFactor = 1000.        


    ##
    ## Algorithm parameters
    ##
    if kwargs.has_key('widthConv'): # units in ms
        widthConv = kwargs['widthConv']
    else:
        widthConv = 900.        
        
    if kwargs.has_key('tau'):
        tau = kwargs['tau']
    else:
        tau = 0.0005
        
    if kwargs.has_key('amp'):
        amp = kwargs['amp']
    else:
        amp = 1.0
        
    if kwargs.has_key('widthGauss'): # units in ms
        widthGauss = kwargs['widthGauss']
    else:
        widthGauss = 700.
    
    if kwargs.has_key('tgh_min'):
        tgh_min = kwargs['tgh_min']
    else:
        tgh_min = 0.1
        
        
    data = data*timeFactor # Take data into ms time base (1. u = 1. ms)
    if kwargs.has_key('ax_data'):
        if kwargs['ax_data'] is not None:
            kwargs['ax_data'].vlines(data/timeFactor, ymin=data_ymin, ymax=data_ymax, color='b')
            kwargs['ax_data'].set_ylim([0.,1.])
    
    ##
    ## Convolve signal    
    ##
    
    if gen_verbose > 0:
        print "\n\nGenerating basis functions..."
        
    ## Generate basis for convolution function
    convX = np.arange(0., widthConv+step, step)
    
    ## Design the convolution kernel
    ## Two options:
    ## 1. Kernel decays to 0 based on the width of the convolution kernel
    ## 2. Kernel decays at a rate, tau, designated by the user
    ## DEFAULT IS METHOD 1
    
    # METHOD 1
    # The next line uses a decay rate, tau, that guarantees the convolution
    # kernel reaches 0 within the relevant window.
    convY = np.zeros(len(convX)) + amp*np.exp(-(8*np.log(2)/widthConv)*convX)
    
    # METHOD 2
    # This line should be used if you want to be able to set the decay rate
    # via a function argument. Note that setting tau as a parameter means
    # that the convolution function may not reach 0 within the kernel window
    # and the resulting analysis signal may not be smooth.
    #convY = np.zeros(len(convX)) + amp*np.exp(-(tau/widthConv)*convX)
    
    conv = np.append(np.zeros(len(convY)-1), convY)
    
    #convY = convY/np.max(convY)
    
    ## This block can be used to visually verify convolution function
    #plt.figure()
    #plt.plot(conv)
    #plt.title('Convolution array')
    #1/0
    
    ## Calculate discrete time-step array
    bufferTime = 1.5*np.max([widthConv, widthGauss])

    ts = None
    if time is None:
        ts = np.round(np.arange(np.max([0.0, np.around(data[0]/step)*step-bufferTime]), np.around(data[-1]/step)*step+bufferTime, step), decimals=2)
    else:
        ts = np.round(np.arange(np.max([0.0, np.around(data[0]/step)*step-bufferTime]), np.around(data[-1]/step)*step+bufferTime, step), decimals=2)  

    
    if gen_verbose > 0:
        print "Generating signal for convolution..."
        
    ## Generate data arrays that can be used for convolution
    # Find t indices where spikes occur
    ixs = np.in1d(ts, np.around(data/step)*step)
    
    """
    print "step: %.3f" % step
    print "spikes >> ixs : (%i, %i)" % (len(data), len(ixs))
    
    print '\nts: %.3f - %.3f' % (ts[0], ts[-1])
    print ts[:5]
    print "to"
    print ts[-5:]
    
    d = np.around(data/step)*step
    print "\ndata: %.3f - %.3f" % (d[0], d[-1])
    print d[:5]
    print "to"
    print d[-5:]
    """
    
    sig = np.zeros(len(ts))
    sig[np.where(ixs == True)[0]] = 1
    
    #return (sig, convY, ts, convX)

    if gen_verbose > 0:
        print "Convolving data..."
    ## Perform convolution and smoothing
    conv_sig = np.convolve(sig, conv, 'same')

    if conv_save:   
        pickle.dump((ts, conv_sig), open('convolved_signal.pkl', 'w'))

    ## Plot convolution function
    #fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212, sharex=ax1)
    
    #ax1.vlines(data, ymin=0.1, ymax=0.5)
    #ax1.set_ylim([0,1])
    #ax2.plot(ts, conv_sig)
    
    if gen_verbose > 0:
        print "Smoothing data..."
    smth_sig = filters.gaussian_filter(conv_sig, widthGauss)
    
    if conv_save:
        pickle.dump((ts, smth_sig), open('smoothed_signal.pkl', 'w'))

    if gen_verbose > 0:
        print "Finding peaks and troughs..."
    
    ##
    ## Find peaks and troughs
    ## Analysis algorithm is based on an evaluating trough-peak-trough sets
    ##
    
    
    ## First attempt at finding peaks and troughs
    pks_sig = np.where(np.diff(np.sign(np.diff(smth_sig))) == -2)[0]
    tghs_sig = np.where(np.diff(np.sign(np.diff(smth_sig))) == 2)[0]
    
    ## Now manage the exceptions...
    ## Adjust peaks and troughs if counts don't match
    if len(pks_sig) > 0 and len(tghs_sig) > 0:
    
        ## First peak occurs BEFORE first trough
        if pks_sig[0] < tghs_sig[0]:
            # Find first trough OR lowest point before the first peak
            tgh_ix_prior = 0
            if smth_sig[0] == np.min(smth_sig[:pks_sig[0]]):
                tgh_ix_prior = np.where(np.diff(np.sign(smth_sig[:pks_sig[0]] - np.min(smth_sig[:pks_sig[0]])*tgh_min)) > 0)[0]
                if len(tgh_ix_prior) == 0:
                    tgh_ix_prior = 0
                else:
                    tgh_ix_prior = tgh_ix_prior[-1]
            tghs_sig = np.append(np.array([tgh_ix_prior]), tghs_sig, axis=0)
                
        ## Last peak occurs AFTER last trough
        if pks_sig[-1] > tghs_sig[-1]:
            # Find last trough OR lowest point after the last peak
            tgh_ix_post = len(smth_sig)-1
            if smth_sig[-1] == np.min(smth_sig[pks_sig[-1]:-1]):
                tgh_ix_post = np.where(np.diff(np.sign(smth_sig[pks_sig[-1]:-1] - smth_sig[pks_sig[-1]]*tgh_min)) < 0)[0] + pks_sig[-1]
                if len(tgh_ix_post) == 0:
                    tgh_ix_post = len(smth_sig)-1
                else:
                    tgh_ix_post = tgh_ix_post[0]
            tghs_sig = np.append(tghs_sig, np.array([tgh_ix_post]), axis=0)
    
              
    if ax_audit is not None:  
        #ax_audit.plot(ts/timeFactor, conv_tonLev, color='b', linestyle='--', alpha=0.3)
        ax_audit.plot(ts/timeFactor, smth_sig, color=conv_color, linestyle='-')
        #ax_audit.axhline(np.mean(smth_sig), color='b', linestyle='-')
        #ax_audit.axhline(np.mean(smth_sig)*0.25, color='b', linestyle='--')
        if conv_audit:
            ax_audit.plot(ts[pks_sig]/timeFactor, smth_sig[pks_sig], 'go', markersize=5)
            ax_audit.plot(ts[tghs_sig]/timeFactor, smth_sig[tghs_sig], 'ro', markersize=5)
        
        fig.canvas.draw()
    
    
    ## In the case that no peaks or troughs are found, return a False array
    #if len(pks_sig) == 0 or len(tghs_sig) == 0:
    if len(pks_sig) == 0:    
        return (smth_sig, False, ts)
    
    ##
    ## Structure the output array as a list of (trough, peak, trough) tuples
    ##
    
    
    arr_pks = np.array([])
    for ix, pk in enumerate(pks_sig):
        pk = pks_sig[ix]
        
        #if audit:
            #ax_levConv.plot(ts[pk]/timeFactor, smth_sig[pk], 'r+', markersize=15)
        
        ## Identify prior and following troughs
        if len(tghs_sig) > 0:
            tgh_ix_prior = np.where(np.diff(np.sign(tghs_sig - pk)) > 0)[0]
            tgh_ix_post = tgh_ix_prior + 1
        else:
            tgh_ix_prior = 0
            tgh_ix_post = 1
            
            tghs_sig = [0, len(smth_sig)-1]
        
    
        if ix > 0:
            ## If a trough is missing between this peak and the previous peak, find one!
            if tghs_sig[tgh_ix_prior] < pks_sig[ix-1]:
                
                if gen_verbose > 1:
                    print "\nFinding prior trough for peak @ t=%.3f..." % (ts[pk]/timeFactor)
                
                ## Find the trough before the current peak and after the previous peak
                tgh_ix_prior = np.where(np.diff(np.sign(smth_sig[pks_sig[ix-1]:pks_sig[ix]] - smth_sig[pk]*tgh_min)) > 0)[0][-1]
                tgh_ix_prior += pks_sig[ix-1]
                
                if ax_audit is not None and conv_audit:
                    ax_audit.plot(ts[tgh_ix_prior]/timeFactor, smth_sig[tgh_ix_prior], 'co')
            else:
                tgh_ix_prior = tghs_sig[tgh_ix_prior]
        else:
            tgh_ix_prior = tghs_sig[tgh_ix_prior]
            
    
        if ix < len(pks_sig) - 1:
            ## If a trough is missing between this peak and the following peak, find one!
            
            if tghs_sig[tgh_ix_post] > pks_sig[ix+1]:
                if gen_verbose > 1:
                    print "\nFinding post trough for peak @ t=%.3f..." % (ts[pk]/timeFactor)
                    
                ## Find the trough after the current peak and before the next peak
                tgh_ix_post = np.where(np.diff(np.sign(smth_sig[pks_sig[ix]:pks_sig[ix+1]] - smth_sig[pk]*tgh_min)) < 0)[0][0]
                tgh_ix_post += pks_sig[ix]
    
                if ax_audit is not None and conv_audit:
                    ax_audit.plot(ts[tgh_ix_post]/timeFactor, smth_sig[tgh_ix_post], 'mo')
            else:
                tgh_ix_post = tghs_sig[tgh_ix_post]
        else:
            tgh_ix_post = tghs_sig[tgh_ix_post]
                
                  
        ## If the previous trough is found along a flat minimum, adjust it towards the peak to tgh_min*peak_height along the analysis curve    
        if ix > 0:
            if len(np.where(smth_sig[pks_sig[ix-1]:pk] == np.min(smth_sig[pks_sig[ix-1]:pk]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points before peak @ %.3f (Code 1)!! --> %.3f" % (ts[pk]/timeFactor, ts[tgh_ix_prior]/timeFactor)
                tgh_ix_prior = np.where(smth_sig[pks_sig[ix-1]:pk] <= smth_sig[pk]*tgh_min)[0][-1] + pks_sig[ix-1]
                
                if gen_verbose > 2:
                    print "--> %.3f" % (ts[tgh_ix_prior]/timeFactor)
                
                if ax_audit is not None and conv_audit:
                    ax_audit.plot(ts[tgh_ix_prior]/timeFactor, smth_sig[tgh_ix_prior], 'co')
                
        else:
            if len(np.where(smth_sig[0:pk] == np.min(smth_sig[0:pk]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points before peak @ %.3f!! --> %.3f (Code 2)" % (ts[pk]/timeFactor, ts[tgh_ix_prior]/timeFactor)            
                    tgh_ix_prior = np.where(smth_sig[0:pk] <= smth_sig[pk]*tgh_min)[0][-1]
                    
                if gen_verbose > 2:
                    print " --> %.3f" % (ts[tgh_ix_prior]/timeFactor)            
                
                if ax_audit is not None and conv_audit:
                    ax_audit.plot(ts[tgh_ix_prior]/timeFactor, smth_sig[tgh_ix_prior], 'co')
                
        ## If the following trough is found along a flat minimum, adjust it towards the peak to tgh_min*peak_height along the analysis curve
        
        if ix < len(pks_sig) - 1:
            if len(np.where(smth_sig[pk:pks_sig[ix+1]] == np.min(smth_sig[pk:pks_sig[ix+1]]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points after peak @ %.3f!! --> %.3f (Code 3)" % (ts[pk]/timeFactor, ts[tgh_ix_post]/timeFactor)
                tgh_ix_post = np.where(smth_sig[pk:pks_sig[ix+1]] <= smth_sig[pk]*tgh_min)[0][0] + pk
                
                if gen_verbose > 2:
                    print "--> %.3f" % (ts[tgh_ix_post]/timeFactor)
        
                if ax_audit is not None and conv_audit:
                    ax_audit.plot(ts[tgh_ix_post]/timeFactor, smth_sig[tgh_ix_post], 'mo')
        else:
            if len(np.where(smth_sig[pk:-1] == np.min(smth_sig[pk:-1]))[0]) > 5:
                if gen_verbose > 2:
                    print "\nToo many min points after peak @ %.3f!! --> %.3f (Code 4)" % (ts[pk]/timeFactor, ts[tgh_ix_post]/timeFactor)            
                tgh_ix_post = np.where(smth_sig[pk:-1] <= smth_sig[pk]*tgh_min)[0][0] + pk
                
                if gen_verbose > 2:
                    print "--> %.3f" % (ts[tgh_ix_post]/timeFactor)
                
                if ax_audit is not None and conv_audit:
                    ax_audit.plot(ts[tgh_ix_post]/timeFactor, smth_sig[tgh_ix_post], 'mo')
    
    
        ## Format output arrays
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

    
    return (smth_sig, arr_pks, ts)


def do_recursion(data, ts, signal, pks, pk_ix, active, kwargs={}):
    if kwargs.has_key('gen_verbose'):
        verbose = kwargs['gen_verbose'] - 1
    else:
        verbose = 0
        
    if kwargs.has_key('timeFactor'):
        timeFactor = kwargs['timeFactor']
    else:
        timeFactor = 1000.
    
    if pk_ix+1 < len(pks):        
        
        if verbose >= 1:
            print "\n==  ==  ==  ==  ==  ==  ==  ==  ==\nRecursing into next peak %i @ t=%.3f" % (pk_ix, ts[pks[pk_ix+1][1]]/timeFactor)
            
        if verbose >= 3:
            print "Burst flag: %s" % kwargs['flag_burst']
            print "Tonic flag: %s" % kwargs['flag_tonic']
        
        activity, activity_type, pk_ix, active = analyze_conv_activity(data, ts, signal, pks,
                                                                       pk_ix=pk_ix+1, active_pk=active, kwargs=kwargs)

    else:
        if verbose >= 1:
            print "Reached the end of the line..."
            
        return ((-1,-1), (kwargs['flag_burst'], kwargs['flag_tonic']), pk_ix, active)
            
    if verbose >= 1:
        print "Done recursing...\n\n"
        
    return (activity, activity_type, pk_ix, active)

def check_tonic(data, ts, signal, pks, pk_ix, active, std_pts, kwargs={}): # active_std
    if kwargs.has_key('gen_verbose'):
        verbose = kwargs['gen_verbose'] - 1
    else:
        verbose = 0
        
    if kwargs.has_key('tonic_std_max'):
        tonic_std_max = kwargs['tonic_std_max']
    else:
        tonic_std_max = 8.
        
    if kwargs.has_key('tonic_min_perc'):
        tonic_min_perc = kwargs['tonic_min_perc']
    else:
        tonic_min_perc = 0.2
        
    if kwargs.has_key('flag_burst'):
        flag_burst = kwargs['flag_burst']
    else:
        flag_burst = False
        
    if kwargs.has_key('flag_tonic'):
        flag_tonic = kwargs['flag_tonic']
    else:
        flag_tonic = False
        
    #tonic_max = np.average(signal[std_pts]) + tonic_std_max*np.std(signal[std_pts])
    #tonic_min = np.average(signal[std_pts]) - tonic_std_max*np.std(signal[std_pts])
    
    #if verbose >= 3:
        #print "\nTonic Range: (%.3f, %.3f)" % (tonic_min, tonic_max)
        
    pk = pk_ix
    activity = (-1,-1)

    active_std = np.std(signal[std_pts])
    
    if verbose >= 3:
        print "\nTonic STD: %.3f" % active_std
        
    if active_std <= tonic_std_max:
    #if np.min(signal[std_pts]) >= tonic_min and np.max(signal[std_pts]) <= tonic_max:
        if verbose >= 2:
            print "Tonic STD <= Max Tonic STD"
            
        #if active[1] == active[3]:
            #min_arr_sig = [signal[active[1]]]
        #else:
        min_arr_sig = signal[active[2]:active[4]]
            
        if verbose >= 3:
            if verbose > 1:
                print "\nMin Tonic: %.3f" % np.min(min_arr_sig)
                print "Tonic Min Threshold: %.3f" % ((signal[active[2]])*tonic_min_perc)
        
        if np.min(min_arr_sig) >= (signal[active[2]])*tonic_min_perc and signal[active[4]] >= (signal[active[2]])*tonic_min_perc:
            if verbose >= 2:
                print "Min Tonic >= Rising Active Peak * Tonic Peak Ratio"
                
            kwargs.update({'flag_burst': flag_burst, 'flag_tonic': True})                                   
            activity, activity_type, pk, active = do_recursion(data, ts, signal, pks, pk_ix, active, kwargs)
            flag_burst, flag_tonic = activity_type            
        
        else:
            if verbose > 1:
                print "Min Tonic < Rising Active Peak * Tonic Peak Ratio"
                print "We've hit rock bottom."
            #active = kwargs['active_pk']

    else:
        if verbose >= 2:
            print "Tonic Std > Max Tonic STD"
            
        active = kwargs['active_pk']
        
    if flag_burst and not flag_tonic:
        kwargs.update({'flag_burst': flag_burst, 'flag_tonic': flag_tonic})
        activity, activity_type, pk, active = do_recursion(data, ts, signal, pks, pk_ix, active, kwargs)
        flag_burst, flag_tonic = activity_type        
        
    return (activity, (flag_burst, flag_tonic), pk, active)


def analyze_conv_activity(data, ts, signal, pks, pk_ix=0, active_pk=[-1,-1,-1,-1,-1], kwargs={}):
    """
    
    Iterative function that classifies peaks and molehills
    
    ts            Array of time points corresponding to signal
    
    signal        Continuous convolution signal based on activity of channel
                  Optimal signal is spike times convolved with exponential decay
                  function smoothed with Gaussian kernel.
                  
    pks           Array of 3-element arrays: [[trough, pk, trough]]
    
    """ 

    ##
    ## Audit parameters
    ##
    if kwargs.has_key('gen_verbose'):
        verbose = kwargs['gen_verbose'] - 1
    else:
        verbose = 0    
        
    if kwargs.has_key('ax_audit'):
        ax_audit = kwargs['ax_audit']
    else:
        ax_audit = None
        
    ##
    ## Data parameters
    ##
    # time scaling factor
    if kwargs.has_key('timeFactor'):
        timeFactor = kwargs['timeFactor']
    else:
        timeFactor = 1000.        
        
    ##    
    ## Algorithm Parameters
    ##
    # peak ratio burst threshold    
    if kwargs.has_key('burst_peak_ratio'):
        burst_peak_ratio = kwargs['burst_peak_ratio']
    else:
        burst_peak_ratio = 3.2
        
    # minimum burst ratio to exclude as single burst
    if kwargs.has_key('burst_exclusion'):
        burst_exclusion = kwargs['burst_exclusion']
    else:
        burst_exclusion = 0.5     

    # burst on/offset lims   
    if kwargs.has_key('burst_perc_pk'):
        burst_perc_pk = kwargs['burst_perc_pk']
    else:
        burst_perc_pk = 0.5
        
    # tonic on/offset lims
    if kwargs.has_key('tonic_perc_pk'):
        tonic_perc_pk = kwargs['tonic_perc_pk']
    else:
        tonic_perc_pk = 0.2
       
    # tonic min perc to maintain        
    if kwargs.has_key('tonic_peak_ratio'):
        tonic_peak_ratio = kwargs['tonic_peak_ratio']
    else:
        tonic_peak_ratio = 0.2
        
    # tonic activity std inter peak interval limit        
    if kwargs.has_key('tonic_std_max'):
        tonic_std_max = kwargs['tonic_std_max']
    else:
        tonic_std_max = 8.
        
    
    if kwargs.has_key('peaks_adjust_thresh'):
        peaks_adjust_thresh = kwargs['peaks_adjust_thresh']
    else:
        peaks_adjust_thresh = 100.
        

    ## Iteration Flags
    if kwargs.has_key('flag_burst'):
        flag_burst = kwargs['flag_burst']
    else:
        flag_burst = False
        
    if kwargs.has_key('flag_tonic'):
        flag_tonic = kwargs['flag_tonic']
    else:
        flag_tonic = False
    
    ## BEGIN ANALYSIS SCRIPT
    
    if verbose >= 3:
        print "\n------------------------------\nActive\n[%i, %i, %i, %i, %i]" % (active_pk[0], active_pk[1], active_pk[2], active_pk[3], active_pk[4])

        
    pk = pk_ix
    current = pks[pk_ix]   
    
    # Calculate current rising ratio
    current_rising_peak = (signal[current[1]] - signal[current[0]])/(ts[current[2]] - ts[current[0]])*timeFactor
    current_falling_peak = (signal[current[1]] - signal[current[2]])/(ts[current[2]] - ts[current[0]])*timeFactor

    active = copy.copy(active_pk)
    # Track peaks and troughs for analysis
    if active_pk[1] == -1:
        active[0] = pks[pk_ix][0] # Leading trough
        active[1] = pks[pk_ix][1] # First peak of active sequence
        active[2] = pks[pk_ix][1] # Highest peak in active sequence
        active[3] = pks[pk_ix][1] # Last peak of active sequence
        active[4] = pks[pk_ix][2] # Lagging trough
        
    else:
        # Update peaks and troughs for analysis
        active[2] = [active_pk[1], pks[pk_ix][1]][np.argmax([signal[active_pk[1]], signal[pks[pk_ix][1]]])] # Update the index of the maximum peak in the analysis range
        active[3] = pks[pk_ix][1]
        active[4] = pks[pk_ix][2] # Extend analysis range to lagging trough of current peak
    
    if verbose >= 3:
        print "\n------------------------------\nUpdated Active\n[%i, %i, %i, %i, %i]\n(%.3f, %.3f, %.3f, %.3f, %.3f)" % \
              (active[0], active[1], active[2], active[3], active[4], \
               ts[active[0]]/timeFactor, ts[active[1]]/timeFactor, ts[active[2]]/timeFactor, ts[active[3]]/timeFactor, ts[active[4]]/timeFactor)
        
        
    # Calculate rising ratio:
    # {Rising trough to peak} / {Time from first trough to last trough}
    active_rising_peak = (signal[active[2]] - signal[active[0]])/(ts[active[4]] - ts[active[0]])*timeFactor
    
    # Calculate falling ratio:
    # {Falling peak to trough} / {Time from first trough to last trough}
    active_falling_peak = (signal[active[2]] - signal[active[4]])/(ts[active[4]] - ts[active[0]])*timeFactor
    
    # Calculate standard deviation of peaks and troughs
    pks_ix_on = np.where(np.array(pks).T[0] >= active[0])[0][0]
    pks_ix_off = np.where(np.array(pks).T[2] <= active[4])[0][-1]
    
    #print "on:off >> %i:%i" % (pks_ix_on, pks_ix_off)
    
    if pks_ix_on == pks_ix_off:
        std_pts = [pks[pk_ix][1], pks[pk_ix][2]]
    else:
        std_pts = []
        __temp = [std_pts.extend([pks[ix][1], pks[ix][2]]) for ix in np.arange(pks_ix_on, pks_ix_off)]
    
    if verbose >= 4:       
        print "\nStats"
        print "Average +/- SEM (STD): %.3f +/- %.3f (%.3f)" % \
              (np.average(signal[std_pts]), np.std(signal[std_pts])/np.sqrt(len(signal[std_pts])), np.std(signal[std_pts]))
        print "25th Quartile, Median, 75th Quartile: %.3f, %.3f, %.3f" % \
              (np.percentile(signal[std_pts], 25), np.median(signal[std_pts]), np.percentile(signal[std_pts], 75)) 
        print "Range: (%.3f, %.3f)" % (np.min(signal[std_pts]), np.max(signal[std_pts]))
    
    if verbose >= 5:
        print "\nTonic Points for STD Evaluation: (ts, signal)" 
        print np.array(ts)[std_pts]/1000.
        print np.array(signal[std_pts])
        
    
    active_std = np.std(signal[std_pts])
    
    if verbose >= 1:
        print "\n------------------------------\nAnalysis region: %.3f - %.3f (%.3f | %.3f, %.3f)" % \
              (ts[active[0]]/timeFactor, ts[active[4]]/timeFactor, active_rising_peak, active_falling_peak, active_std)
        
        
    if np.max(signal[pks[pk_ix]]) >= peaks_adjust_thresh:
        active_adjustment = [(signal[active[1]] - signal[active[0]])/signal[active[1]], \
                             (signal[active[3]] - signal[active[4]])/signal[active[3]]]
        
        current_adjustment = [(signal[current[1]] - signal[current[0]])/signal[current[1]], \
                              (signal[current[1]] - signal[current[2]])/signal[current[1]]]
    else:
        active_adjustment = [1.,1.]
        current_adjustment = [1.,1.]
    
    active_rising_peak = active_rising_peak * np.max(active_adjustment)
    active_falling_peak = active_falling_peak * np.max(active_adjustment)
    
    current_rising_peak = current_rising_peak * np.max(current_adjustment)
    current_falling_peak = current_falling_peak * np.max(current_adjustment)
    
    if verbose >= 3:
        #print "\nActive Adjustments: %.3f" % active_adjustment
        #print "Current Adjustments: %.3f" % current_adjustment

        print "\nActive Adjustments: %.3f | %.3f" % (active_adjustment[0], active_adjustment[1])
        print "Current Adjustments: %.3f | %.3f" % (current_adjustment[0], current_adjustment[1])    

    if verbose >= 2:
            print "\nAdjusted Active Ratios: %.3f, %.3f" % (active_rising_peak, active_falling_peak)    
            print "Adjusted Current Ratios: %.3f, %.3f" % (current_rising_peak, current_falling_peak)
            
            temp_adjust = (ts[current[2]]-ts[current[0]])/1000.*(ts[current[2]]+ts[current[0]])/2.

    if verbose >= 3:
        print "Burst flag: %s" % flag_burst
        print "Tonic flag: %s" % flag_tonic

    if verbose >= 1:
        print "\n------------------------------\nRUNNING BURST ALGORITHM"

    # Rising Active Peak >= Tonic Peak Ratio
    if active_rising_peak >= tonic_peak_ratio:
        if verbose >= 3:
            print "\nActive Rising Peak: %.3f" % active_rising_peak
        
        if verbose >= 2:
            print "Rising Active Peak >= Tonic Peak Ratio"

        if active_pk[2] == -1:
            flag_tonic = True
        
        # Rising Active peak >= Burst Peak Ratio
        if active_rising_peak >= burst_peak_ratio:
            if verbose >= 2:
                print "Rising Active Peak >= Burst Peak Ratio"
                
            if active_pk[2] == -1:
                flag_burst = True
                
            if flag_burst:
                
                # Falling Active Peak >= Burst Peak Ratio
                if active_falling_peak >= burst_peak_ratio:
                    if verbose >= 3:
                        print "\nActive Falling Peak: %.3f" % active_falling_peak
                    if verbose >= 2:
                        print "Falling Active Peak >= Burst Peak Ratio"
                        
                    flag_burst = True
                    
                    if verbose >= 3:
                        print "\nExclusion Ratio: %.3f" % \
                              (np.min([active_falling_peak/active_rising_peak, active_rising_peak/active_falling_peak]))
                        
                    # Falling Peak to Rising Peak >= Exclusion Criteria
                    if active_falling_peak/active_rising_peak >= burst_exclusion:
                        if verbose >= 2:
                            print "Peak Ratios >= Exclusion Criteria"
                        flag_tonic = False
                    else:
                        if verbose >= 2:
                            print "Peak Ratios <= Exclusion Criteria"
                            
                        kwargs.update({'flag_burst': flag_burst, 'flag_tonic': flag_tonic})                        
                        activity, activity_type, pk, active = do_recursion(data, ts, signal, pks, pk_ix, active, kwargs)
                        flag_burst, flag_tonic = activity_type
                
                else:
                    if verbose >= 2:
                        print "Falling Active Peak < Burst Peak Ratio"

                    kwargs.update({'flag_burst': flag_burst, 'flag_tonic': flag_tonic})  
                    kwargs.update({'active_pk': active_pk})
                    activity, (flag_burst, flag_tonic), pk, active = check_tonic(data, ts, signal, pks, pk_ix, active, std_pts, kwargs)
                    
            else:
                if verbose >= 2:
                    print "Burst is breaking up tonic activity..."
                    
                active = active_pk
                pk -= 1
                
            
        else:
            if verbose >= 2:
                print "Rising Active Peak < Burst Peak Ratio"
            flag_burst = False
            
            if verbose >= 3:
                print "\nRising Current Ratio: %.3f" % current_rising_peak
                
            if current_rising_peak >= burst_peak_ratio:
                if verbose >= 2:
                    print "Rising Current Ratio >= Burst Peak Ratio"
                    
                active = active_pk
                pk -= 1
                    
            else:
                if verbose >= 2:
                    print "Rising Current Ratio < Burst Peak Ratio"
                    
                kwargs.update({'flag_burst': flag_burst, 'flag_tonic': flag_tonic})    
                kwargs.update({'active_pk': active_pk})
                activity, (flag_burst, flag_tonic), pk, active = check_tonic(data, ts, signal, pks, pk_ix, active, std_pts, kwargs=kwargs)
        
    elif flag_tonic:
        # Check to see if tonic activity continues
        ## Rising peak ratio will be less than tonic peak ratio because duration of tonic activity can be exceedingly long...
        ## Consequently the rising peak ratio will fall below the tonic peak ratio threshold..
        ## This snippet assumes that the rising peak ratio was valid at some point -- otherwise we wouldn't be here :D
        ## Then it checks to see if the tonic peaks/troughs standard deviation is still valid.
        
        kwargs.update({'flag_burst': flag_burst, 'flag_tonic': flag_tonic}) 
        kwargs.update({'active_pk': active_pk})
        activity, (flag_burst, flag_tonic), pk, active = check_tonic(data, ts, signal, pks, pk_ix, active, std_pts, kwargs)
            
        if active_std > tonic_std_max:
            flag_burst = False

    if verbose >= 3:
        print "\nBurst flag: %s" % flag_burst
        print "Tonic flag: %s" % flag_tonic
    
    onset_t = ts[active[0]]/timeFactor
    offset_t = onset_t
    
    ## Check for validity of peaks and troughs if burst
    if flag_burst and not flag_tonic and active[1] != active[3]:
        pk_first_ix = np.where(np.array(pks).T[1] >= active[1])[0][0]
        pk_high_ix = np.where(np.array(pks).T[1] >= active[2])[0][0]
        pk_last_ix = np.where(np.array(pks).T[1] <= active[3])[0][-1]
        
        if verbose >= 3:
            print "\nFirst peak: %i (%.3f)" % (pk_first_ix, ts[pks[pk_first_ix][1]]/timeFactor)
            print "High peak: %i (%.3f)" % (pk_high_ix, ts[pks[pk_high_ix][1]]/timeFactor)
            print "Last peak: %i (%.3f)" % (pk_last_ix, ts[pks[pk_last_ix][1]]/timeFactor)
        
        rising_tghs = []
        __temp = [rising_tghs.append(signal[_pk[0]]) for _pk in pks[pk_first_ix:pk_high_ix]]
        rising_tghs.append(signal[pks[pk_high_ix][0]])
        
        rising_pks = []
        __temp = [rising_pks.append(signal[_pk[1]]) for _pk in pks[pk_first_ix:pk_high_ix]]
        rising_pks.append(signal[pks[pk_high_ix][1]])
    
    
        flag_rising_tghs = all([x <= y for x,y in zip(rising_tghs, rising_tghs[1:])])
        flag_rising_pks = all([x <= y for x,y in zip(rising_pks, rising_pks[1:])])

        if verbose >= 3:
            print "\n"
            for ix, tgh in enumerate(rising_tghs):
                print "Rising Trough %i: %.3f, %.3f" % (ix, ts[tgh]/timeFactor, tgh)
            print "Rising: %s" % flag_rising_tghs
            
            print "\n"
            for ix, _pk in enumerate(rising_pks):
                print "Rising Peak %i: %.3f, %.3f" % (ix, ts[_pk]/timeFactor, _pk)
            print "Rising: %s" % flag_rising_pks
                
        
        falling_tghs = []
        #falling_tghs.append(pks[pk_high_ix][2])
        __temp = [falling_tghs.append(signal[_pk[2]]) for _pk in pks[pk_high_ix:pk_last_ix]]
        falling_tghs.append(signal[pks[pk_last_ix][2]])
        
        falling_pks = []
        __temp = [falling_pks.append(signal[_pk[1]]) for _pk in pks[pk_high_ix:pk_last_ix]]
        falling_pks.append(signal[pks[pk_last_ix][1]])
        
        
        flag_falling_tghs = all([x >= y for x,y in zip(falling_tghs, falling_tghs[1:])])
    
        flag_falling_pks = all([x >= y for x,y in zip(falling_pks, falling_pks[1:])])        

        if verbose >= 3:
            print "\n"
            for ix, tgh in enumerate(falling_tghs):
                print "Falling Trough %i: %.3f, %.3f" % (ix, ts[tgh]/timeFactor, tgh)
            print "Falling: %s" % flag_falling_tghs
                
            print "\n"
            for ix, _pk in enumerate(falling_pks):
                print "Falling Peak %i: %.3f, %.3f" % (ix, ts[_pk]/timeFactor, _pk)
            print "Falling: %s" % flag_falling_pks
        
        
        if not (flag_rising_tghs and flag_rising_pks and flag_falling_tghs and flag_falling_pks):
            flag_burst = False
            active = active_pk
            pk -= 1
    elif flag_burst and not flag_tonic and active[1] == active[3]:
        rising_ratio = (signal[active[2]] - signal[active[0]])/(ts[active[4]] - ts[active[0]])*timeFactor
        falling_ratio = (signal[active[2]] - signal[active[4]])/(ts[active[4]] - ts[active[0]])*timeFactor
        
        if rising_ratio >= burst_peak_ratio and falling_ratio >= burst_peak_ratio:
            flag_burst = True
        else:
            flag_burst = False
    
    activity_type = ''
    if verbose >= 3:
        print "\nActive\n[%i, %i, %i, %i, %i]" % (active[0], active[1], active[2], active[3], active[4])

    if verbose >= 4:
        print "\nActive Pk\n[%i, %i, %i, %i, %i]" % (active_pk[0], active_pk[1], active_pk[2], active_pk[3], active_pk[4])
    
    if verbose >= 3:
        print "\nBurst flag: %s" % flag_burst
        print "Tonic flag: %s" % flag_tonic

    if flag_burst and not flag_tonic:
        if verbose >= 1:
            print "\nIt's a BURST!!"
            
    if not flag_burst and flag_tonic:
        if verbose >= 1:
            print "\nIt's TONIC ACTIVITY!!"
        
    if flag_burst != flag_tonic and active_pk[1] == -1 and active[2] != -1:
        if flag_tonic and active[1] == active[3]:
            flag_burst = False
            flag_tonic = False
        else:
            if flag_burst:
                perc_pk = burst_perc_pk
            elif flag_tonic:
                perc_pk = tonic_perc_pk
    
            if verbose >= 2:
                print "Activity Time Range: %.3f - %.3f" % (ts[active[0]]/timeFactor, ts[active[4]]/timeFactor)
    
            if ax_audit != None and verbose >= 5:
                ax_audit.plot(ts[active[0]]/timeFactor, signal[active[0]], 'g+', markersize=15)
                ax_audit.plot(ts[active[1]]/timeFactor, signal[active[1]], 'c+', markersize=15)
                ax_audit.plot(ts[active[2]]/timeFactor, signal[active[2]], 'r+', markersize=15)
                ax_audit.plot(ts[active[3]]/timeFactor, signal[active[3]], 'm+', markersize=15)
                ax_audit.plot(ts[active[4]]/timeFactor, signal[active[4]], 'k+', markersize=15)                
                
            if ax_audit != None and verbose >= 5:
                ax_audit.plot(ts[active[0]:active[1]]/timeFactor, signal[active[0]:active[1]], 'b.')
                
            if verbose >= 3:
                print "\nRising (min, max): (%.3f, %.3f)" % (np.min(signal[active[0]:active[1]]), np.max(signal[active[0]:active[1]]))
                print "Onset Threshold: %.3f" % ((signal[active[1]] - signal[active[0]])*perc_pk+signal[active[0]])            
                
            ix = np.where(signal[active[0]:active[1]] >= (signal[active[1]] - signal[active[0]])*perc_pk+signal[active[0]])[0][0]
            onset_t = data[np.where(data >= ts[ix + active[0]]/timeFactor)[0][0]]
            
            if verbose >= 2:
                print "\nOnset time: %.3f" % onset_t
            
    
            if ax_audit != None and verbose >= 5:
                ax_audit.plot(ts[active[3]:active[4]]/timeFactor, signal[active[3]:active[4]], 'b.')
    
            if verbose >= 3:
                print "\nFalling (min, max): (%.3f, %.3f)" % (np.min(signal[active[3]:active[4]]), np.max(signal[active[3]:active[4]]))
                print "Offset Threshold: %.3f" % ((signal[active[3]] - signal[active[4]])*perc_pk+signal[active[4]])
                
            ix = np.where(signal[active[3]:active[4]] <= (signal[active[3]] - signal[active[4]])*perc_pk+signal[active[4]])[0][0]
            offset_t = data[np.where(data <= ts[ix + active[3]]/timeFactor)[0][-1]]
            
            if verbose >= 2:
                print "\nOffset time: %.3f" % offset_t
    if not flag_burst and not flag_tonic:
        if verbose > 1:
            print "Peaks: (%i, %i)" % (pk_ix, pk)
            
        pk = pk_ix
        #if pk_ix == pk + 1:
            #pk += 1
        

    return ([onset_t, offset_t], (flag_burst, flag_tonic), pk, active)



def do_ehv(data, **kwargs):
    
    ##
    ## Audit parameters
    ##
    if kwargs.has_key('ax_data'):
        ax_data = kwargs['ax_data']
    else:
        ax_data = None

    if kwargs.has_key('ax_audit'):
        ax_audit = kwargs['ax_audit']
    else:
        ax_audit = None
                    
    if kwargs.has_key('conv_audit'):
        conv_audit = kwargs['conv_audit']
    else:
        conv_audit = True
        
    if kwargs.has_key('gen_verbose'):
        gen_verbose = kwargs['gen_verbose']
    else:
        gen_verbose = 0

    if kwargs.has_key('iter_verbose'):
        iter_verbose = kwargs['iter_verbose']
    else:
        iter_verbose = 0
        
        
    ##
    ## Algorithm parameters
    ##
    if kwargs.has_key('min_tonic_len'):
        min_tonic_len = kwargs['min_tonic_len']
    else:
        # tonic activity onset
        min_tonic_len = 3 # 10

    if kwargs.has_key('min_tonic_ratio'):
        min_tonic_ratio = kwargs['min_tonic_ratio']
    else:
        # tonic minimum ratio
        min_tonic_ratio = 0.6        
    
    smth_signal, arr_pks, ts = make_data_to_signal(data, kwargs)
    
    if arr_pks is False:
        return (False, False)
    
    bursts = []
    tonics = []

    if gen_verbose >= 1:
        print "\nAnalyzing Peaks and Troughs..."
        
    ix = 0       
    while ix < len(arr_pks):
        
        if gen_verbose > 1:
            print "\n\n\n==================================="
            print "--> ITERATING %i @ t=%.3f!!" % (ix, ts[arr_pks[ix][1]]/1000.)
            
        kwargs.update({'flag_burst': False, 'flag_tonic': False})
        activity, activity_type, ix, active = analyze_conv_activity(data, ts, smth_signal, arr_pks, ix, kwargs=kwargs)          

        if activity[1] != activity[0]:
            if activity_type[0] is True and activity_type[1] is False:
                if gen_verbose > 2:
                    print "\n\nBursting activity!!"
                            
                bursts.append(activity)
                
                if ax_audit is not None and conv_audit:
                    ax_audit.axvline(activity[0], color='c')
                    ax_audit.axvline(activity[1], color='c')
                
                
            elif activity_type[0] is False and activity_type[1] is True:
                if gen_verbose > 2:
                    print "\n\nTonic activity!!"
    
                tonics.append(activity)
                
                if ax_audit is not None and conv_audit:
                    ax_audit.axvline(activity[0], color='g', linewidth=2)
                    ax_audit.axvline(activity[1], color='r', linewidth=2)
                
        ix += 1
        #1/0
                

    if gen_verbose > 0:
        print "\nBURSTS: %i" % len(bursts)
    if gen_verbose > 2:
        print bursts
        
    if gen_verbose > 0:
        print "\nTONICS: %i" % len(tonics)
    if gen_verbose > 2:
        print tonics    
        
        
    del data
    
    return {'bursts': bursts, 'tonics': tonics, 'ts': ts, 'signal': smth_signal, 'peaks': arr_pks, 'pars': kwargs}
#except:
    #return {name: {'bursts': ['ERROR'], 'tonics': ['ERROR']}}   
    

## ===== ===== ===== ===== =====
## ===== ===== ===== ===== =====


## 00594
## 03010
## 03101
## 07549
## 10319

## 021
## 03608

## Lev-04
## Lev-05
## Lev-06

"""
series = sys.argv[1]

analysisChan = 'Tonic Lev MN'
data = None


print "Loading data file..."
try:
    fname = 'FinalDissertationModel_Standalone-%s.hd5' % series
    fpath = '/Users/brycechung/Google Drive/_Research/AnimatLabDS/SamplingData/data/%s' % fname
    h5file = h5py.File(fpath, 'r')
    tScale = 1000.
    
except:
    
    print "File didn't work... %s" % fname
    try:
        fname = 'GainRatioModel-v2_Standalone-%s.hd5' % series
        fpath = '/Users/brycechung/Google Drive/_Research/AnimatLabDS/SamplingData/data/%s' % fname
        h5file = h5py.File(fpath, 'r')
        tScale = 1000.
        
    except:    
        try:
            fname = '%s.txt' % series
            fpath = '/Users/brycechung/Google Drive/_Research/AnimatLabDS/SamplingData/data/%s' % fname
            h5file = {analysisChan: np.loadtxt(fpath)/1000.}
            tScale = 1000.
            
        except:
            raise ValueError("Ack! No file found!\nFilename: %s" % fname)
        
try:
    data = h5file[analysisChan][:]

except:
    raise KeyError("Whoops! Analysis channel not found in data file!\nChannel Name: %s" % analysisChan)


ehv_pars = {'widthConv': 1600.,
            'amp': 2.8,
            'widthGauss': 1100.,
            'tgh_min': 0.1,
            'burst_peak_ratio': 2.5,
            'burst_exclusion': 0.8,
            'tonic_peak_ratio': 0.15,
            'tonic_min_perc': 0.3,
            'tonic_std_max': 20,
            'burst_perc_pk': 0.4,
            'tonic_perc_pk': 0.3,
            'peaks_adjust_thresh': 100 ## N/A
            }


print "Running analysis..."
results = do_ehv(data, step=0.5, timeFactor=tScale, gen_verbose=1, **ehv_pars)

#results = do_hill_valley(data, step=0.5, timeFactor=1000.,
                         #ax_data=ax_raster, ax_audit=ax_conv, gen_verbose=5,
                         #widthConv=2400., amp=2.4, widthGauss=800.,
                         #tgh_min=0.1,
                         #burst_peak_ratio = 3.3, burst_exclusion = 0.4, # burst_peak_ratio = 3.5
                         #tonic_peak_ratio = 0.15, tonic_min_perc = 0.25, tonic_std_max = 16, # Tonic STD Max = 9
                         #burst_perc_pk = 0.4, tonic_perc_pk = 0.3,
                         #peaks_adjust_thresh = 100
                         #)

print "Making figures..."
fig = plt.figure()
fig.suptitle(fname)

axRaster = fig.add_subplot(211)
axSignal = fig.add_subplot(212, sharex=axRaster)

axRaster.vlines(data, ymin=0.1, ymax=0.2)

yline = 0.25
for burst in results['bursts']:
    axRaster.plot(burst, [yline]*2, color='r', linewidth=2)
    
for tonic in results['tonics']:
    axRaster.plot(tonic, [yline]*2, color='r', linewidth=4)
    
axRaster.set_ylim([0, yline+0.3])

print "\nBursts"
hand_bursts = np.loadtxt('/Users/brycechung/Google Drive/_Research/Publications/Neural Activity Classify/Data/FinalDissertationModel_Standalone-%s_hand-bursts.txt' % series)
for burst in hand_bursts:
    print "Plotting: (%.3f, %.3f)" % (burst[0]/1000.+30, (burst[0]+burst[1])/1000.+30.)
    axRaster.plot([burst[0]/1000.+30, (burst[0]+burst[1])/1000.+30.], [0.3]*2, color='k', linewidth=2)

print "\nTonics"
hand_tonics = np.loadtxt('/Users/brycechung/Google Drive/_Research/Publications/Neural Activity Classify/Data/FinalDissertationModel_Standalone-%s_hand-tonic.txt' % series)
for tonic in hand_tonics:
    print "Plotting: (%.3f, %.3f)" % (tonic[0]/1000.+30, (tonic[0]+tonic[1])/1000.+30.)
    axRaster.plot([tonic[0]/1000.+30, (tonic[0]+tonic[1])/1000.+30.], [0.3]*2, color='k', linewidth=4)


axSignal.plot(results['ts']/1000., results['signal'], 'b-')

plt.draw()

ehv_pars = {'widthConv': 1600.,
            'amp': 2.8,
            'widthGauss': 1100.,
            'tgh_min': 0.1,
            'burst_peak_ratio': 2.5,
            'burst_exclusion': 0.8,
            'tonic_peak_ratio': 0.15,
            'tonic_min_perc': 0.3,
            'tonic_std_max': 20,
            'burst_perc_pk': 0.2,
            'tonic_perc_pk': 0.3,
            'peaks_adjust_thresh': 100 ## N/A
            }


print "Running analysis 2..."
results = do_ehv(data, step=0.5, timeFactor=tScale, gen_verbose=1, **ehv_pars)

#results = do_hill_valley(data, step=0.5, timeFactor=1000.,
                         #ax_data=ax_raster, ax_audit=ax_conv, gen_verbose=5,
                         #widthConv=2400., amp=2.4, widthGauss=800.,
                         #tgh_min=0.1,
                         #burst_peak_ratio = 3.3, burst_exclusion = 0.4, # burst_peak_ratio = 3.5
                         #tonic_peak_ratio = 0.15, tonic_min_perc = 0.25, tonic_std_max = 16, # Tonic STD Max = 9
                         #burst_perc_pk = 0.4, tonic_perc_pk = 0.3,
                         #peaks_adjust_thresh = 100
                         #)

yline = 0.35
for burst in results['bursts']:
    axRaster.plot(burst, [yline]*2, color='b', linewidth=2)
    
for tonic in results['tonics']:
    axRaster.plot(tonic, [yline]*2, color='b', linewidth=4)

#pickle.dump(results, open('/Users/brycechung/Google Drive/_Research/Publications/Neural Activity Classify/data/Final Analysis/data/results-%s-EHV.dat' % series, 'w'))
"""
