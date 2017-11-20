import os, sys

import numpy as np
import h5py, pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.ion()
plt.close('all')

from lib_final_ehv import do_ehv
from lib_final_cma import do_cma
from lib_final_poisson import do_poisson

# 07549
# 00594
# 03010
# 03101
# 10319
# 021
# 04
# 05
# 06

#series = sys.argv[1]
#fnames = ['07549', '00594', '03010', '03101', '03608', '10319', '021', '04', '05', '06']
fnames = ['04', '05', '06']

setName = 'data'

saveAnalysis = False

analysisChan = 'Tonic Lev MN'

machine = 'mac'

if machine == 'mac':
    fldrName = '/Users/brycechung/Google Drive/_Research/Publications/Neural Activity Classify/Data/'
elif machine == 'pc':
    fldrName = 'F:/Users/BChung/Google Drive/_Research/Publications/Neural Activity Classify/Data/'
else:
    raise


def jaccard_index(a, b, sampling=0.5, plot=False):
    if plot:
        plt.figure()
    ts = np.arange(0., 240., sampling/1000.)
    
    
    a_arr = np.zeros(len(ts))
    for burst in a:
        a_arr[np.where((ts >= burst[0]) & (ts <= burst[1]))[0]] = 1.  
    if plot:
        plt.plot(ts, a_arr + 5*(1+0.5), color='g', label='A')
    
    b_arr = np.zeros(len(ts))
    for burst in b:
        b_arr[np.where((ts >= burst[0]) & (ts <= burst[1]))[0]] = 1.
    if plot:
        plt.plot(ts, b_arr + 4*(1+0.5), color='r', label='B')    

    ab_intersect = len(np.where((a_arr == 1) & (b_arr == 1))[0])
    a_1 = np.count_nonzero(a_arr)
    b_1 = np.count_nonzero(b_arr)
    
    try:
        return float(ab_intersect)/float(a_1 + b_1 - ab_intersect)
    except:
        if float(a_1 + b_1 - ab_intersect) == 0:
            return 0
        else:
            return False
for series in fnames:
    plt.close()
    
    try:
        fName_spikes = 'FinalDissertationModel_Standalone-%s.dat' % series
        fData_spikes = pickle.load(open(os.path.join(fldrName, fName_spikes), 'r'))
        
    except:
        try:
            fName_spikes = 'GainRatioModel-v2_Standalone-%s.dat' % series
            fData_spikes = pickle.load(open(os.path.join(fldrName, fName_spikes), 'r'))
        except:
            try:
                fName_spikes = 'Lev-%s.txt' % series
                fData_spikes = np.loadtxt(os.path.join(fldrName, fName_spikes), delimiter='\t')
            except:
                pass
            
    try:
        spike_ts = fData_spikes[analysisChan]['data']
    except:
        spike_ts = fData_spikes/1000.
    spike_isi = np.diff(spike_ts)
    

    print "Making raster plot..."
    plt.figure()
    plt.title('%s\nSpike Raster' % fName_spikes)
    plt.vlines(spike_ts, ymin=0.1, ymax = 0.5)
    
    
    print "Loading visual analysis..."
    vizBurstsData = []
    try:
        fname_vizBursts = 'FinalDissertationModel_Standalone-%s_hand-bursts.txt' % series
        vizBurstsData = np.loadtxt(os.path.join(fldrName, fname_vizBursts))
    except:
        try:
            fname_vizBursts = 'GainRatioModel-v2_Standalone-%s_hand-bursts.txt' % series
            vizBurstsData = np.loadtxt(os.path.join(fldrName, fname_vizBursts))
        except:
            pass
    if np.shape(vizBurstsData) == (2,):
        vizBurstsData = [vizBurstsData]
            
    for row in vizBurstsData:
        plt.plot(np.array([row[0], row[0]+row[1]])/1000. + 30., [0.6, 0.6], 'k', linewidth=2)
        
    vizTonicsData = []
    try:
        fname_vizTonics = 'FinalDissertationModel_Standalone-%s_hand-tonic.txt' % series
        vizTonicsData = np.loadtxt(os.path.join(fldrName, fname_vizTonics))
    except:
        try:
            fname_vizTonics = 'GainRatioModel-v2_Standalone-%s_hand-tonic.txt' % series
            vizTonicsData = np.loadtxt(os.path.join(fldrName, fname_vizTonics))
        except:
            pass
    if np.shape(vizTonicsData) == (2,):
        vizTonicsData = [vizTonicsData]
            
    for row in vizTonicsData:
        plt.plot(np.array([row[0], row[0]+row[1]])/1000. + 30., [0.6, 0.6], 'k', linewidth=4)
        
    
    print "Analyzing EHV..."
    ## Par Set 1
    #ehv_burstData = do_ehv(spike_ts, step=0.5, timeFactor=1000., gen_verbose=0,
                             #widthConv=1800., amp=2.2, widthGauss=1100.,tgh_min=0.1,
                             #burst_peak_ratio = 3.3, burst_exclusion = 0.8,
                             #tonic_peak_ratio = 0.5, tonic_min_perc = 0.3, tonic_std_max = 18,
                             #burst_perc_pk = 0.2, tonic_perc_pk = 0.3,
                             #peaks_adjust_thresh = 100 )
                             
    # Par Set 2
    ehv_burstData = do_ehv(spike_ts, step=0.5, timeFactor=1000., gen_verbose=0,
                             widthConv=1500., amp=2.25, widthGauss=1100.,tgh_min=0.1,
                             burst_peak_ratio = 3.0, burst_exclusion = 0.8,
                             tonic_peak_ratio = 0.15, tonic_min_perc = 0.3, tonic_std_max = 14,
                             burst_perc_pk = 0.2, tonic_perc_pk = 0.3,
                             peaks_adjust_thresh = 100 )
    
    for burst in ehv_burstData['bursts']:
        plt.plot(burst, [0.7, 0.7], 'm', linewidth=2)
    for tonic in ehv_burstData['tonics']:
        plt.plot(tonic, [0.7, 0.7], 'm', linewidth=4)
    pickle.dump(ehv_burstData, open(fldrName+'Final Analysis/%s/results-%s-EHV.dat' % (setName, series), 'w'))
    
    print "Analyzing CMA..."
    cma_burstData = do_cma(spike_ts, tScale=1000., minLen=3, histBins=100, verbose=0)
    
    for burst in cma_burstData['bursts']:
        plt.plot(burst, [0.8, 0.8], 'g', linewidth=2)
    pickle.dump(cma_burstData, open(fldrName+'Final Analysis/%s/results-%s-CMA.dat' % (setName, series), 'w'))
    
    
    print "Analyzing Poisson..."
    # Par Set 1
    poisson_burstData = do_poisson(spike_ts,
                                   minBurstLen = 7,
                                   maxInBurstLen = 5, 
                                   maxBurstIntStart = 2.5, 
                                   maxBurstIntTerm = 2.0, 
                                   surprise = 0.3)
                                   
    ## Par Set 2
    #poisson_burstData = do_poisson(spike_ts,
                                   #minBurstLen = 7,
                                   #maxInBurstLen = 8, 
                                   #maxBurstIntStart = 3, 
                                   #maxBurstIntTerm = 2.0, 
                                   #surprise = 0.6)
    
    
    for burst in poisson_burstData['bursts']:
        plt.plot(burst, [0.9, 0.9], 'b', linewidth=2)
    pickle.dump(poisson_burstData, open(fldrName+'Final Analysis/%s/results-%s-PS.dat' % (setName, series), 'w'))
    
    
    plt.ylim([0,1])
    
    fig = plt.figure(1)
    ax = fig.axes[0]
    
    plt.subplots_adjust(bottom=0.15)
    
    legendHandles = []
    legendHandles.append(patches.Patch(color='m', label='Extended Hill-Valley'))
    legendHandles.append(patches.Patch(color='g', label='Cumulative Moving Average'))
    legendHandles.append(patches.Patch(color='b', label='Poisson Surprise'))
    ax.legend(handles=legendHandles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=11)
    
    tOffset = 0
    if series not in ['04', '05', '06']:
        tOffset = 30.
    
    print "\n\nJACCARD INDEX for %s" % series
    print "Method\tBursts\tTonics"
    print "IN VITRO JACCARD NOT CORRECT!!"
    print "EHV\t%.3f\t%.3f" % (jaccard_index(np.array([[tonic[0], tonic[0]+tonic[1]] for tonic in vizBurstsData])/1000.+tOffset, ehv_burstData['bursts']), jaccard_index(np.array([[tonic[0], tonic[0]+tonic[1]] for tonic in vizTonicsData])/1000.+tOffset, ehv_burstData['tonics']))
    print "CMA\t%.3f\t%.3f" % (jaccard_index(np.array([[tonic[0], tonic[0]+tonic[1]] for tonic in vizBurstsData])/1000.+tOffset, cma_burstData['bursts']), jaccard_index(np.array([[tonic[0], tonic[0]+tonic[1]] for tonic in vizTonicsData])/1000.+tOffset, []))
    print "PS\t%.3f\t%.3f" % (jaccard_index(np.array([[tonic[0], tonic[0]+tonic[1]] for tonic in vizBurstsData])/1000.+tOffset, poisson_burstData['bursts']), jaccard_index(np.array([[tonic[0], tonic[0]+tonic[1]] for tonic in vizTonicsData])/1000.+tOffset, []))
    
    #1/0