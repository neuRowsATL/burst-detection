import sys, copy
import h5py

import numpy as np
from scipy import signal, stats

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.close('all')


def do_cma(data, tScale=1000., minLen=3, histBins=100, plot=False, verbose=4):
    if verbose > 0:
        print "\nRUNNING CUMULATIVE MOVING AVERAGE ANALYSIS\n"
    
    if plot:
        # Make spike raster plot
        fig1 = plt.figure()
        fig1.suptitle('Spike Raster')
        ax1 = fig1.add_subplot(111)
        ax1.vlines(data, ymin=0.1, ymax=0.5, color='k')
        ax1.set_ylim([0,1])
    
    ## Calculate inter spike intervals and histogram
    isi = np.diff(data*tScale)
    xN, xBins = np.histogram(isi, bins=histBins)
    
    if verbose > 0:
        print "Mean ISI: %.3f" % np.mean(isi)
    
    if plot:
        # Make isi histogram plot
        fig2 = plt.figure()
        fig2.suptitle('ISI Histogram')
        ax2 = fig2.add_subplot(111)
        ax2.bar(xBins[1:], xN, width=0.6*(xBins[1]-xBins[0]))
    
    
    ## Calculate cumulative sum and moving average
    csum = np.array([np.sum(xN[:ix]) for ix in range(len(xN)+1)])
    cma = np.array([np.sum(xN[:ix])/(ix+1) for ix in range(len(xN)+1)])
    
    ## Determine peak of moving average curve
    xm = np.argmax(cma)
    
    ## Calculate skew of ISI histgoram
    skew = stats.skew(xN)
    
    
    ## Determine alpha1 and alpha2 based on published parameter values
    ## Kapucu, et al. 2012. Burst analysis tool for developing neuronal networks
    ## exhibiting highly varying action potential dynamics. Front. Comp Neurosci.
    ## 8(38).
    
    alph1 = 1.
    alph2 = 1.
    
    if skew >= 9:
        alph1 = 0.3
        alph2 = 0.1
    elif skew < 9 and skew >= 4:
        alph1 = 0.5
        alph2 = 0.3
    elif skew < 4:
        alph2 = 0.5
        if skew >= 1:
            alph1 = 0.7
        else:
            alph1 = 0.75
            
            
    ## Calculate isi threshold from histogram bins
    x1 = np.max(cma)*alph1    
    ixBurstThresh = np.where(np.diff(np.sign(cma-x1)) < 0)[0]
    if len(ixBurstThresh) > 0:
        burstThresh = xBins[ixBurstThresh[-1]]
    else:
        burstThresh = xBins[-1]
        
    x2 = np.max(cma)*alph2        
    ixBurstRelThresh = np.where(np.diff(np.sign(cma-x2)) < 0)[0]
    if len(ixBurstRelThresh) > 0:
        burstRelThresh = xBins[ixBurstRelThresh[-1]]
    else:
        burstRelThresh = xBins[-1]
        
        
    if verbose > 1:        
        print "\nBurst Thresh: %.3f" % burstThresh
        print "Burst Rel Thresh: %.3f" % burstRelThresh
    
    if plot:
        ax2.plot(xBins, cma, 'c-', linewidth=2)
        ax2.axvline(burstThresh, color='r', linestyle='-')
        ax2.axvline(burstRelThresh, color='r', linestyle='--')


    ## Determine spike events that are in a burst or are burst-related    
    burstSpikes = np.where(isi <= burstThresh)[0] + 1
    burstRelSpikes = np.where((isi <= burstRelThresh) & (isi > burstThresh))[0] + 1
    breakSpikes = np.where(isi > burstThresh)[0]+1
    
    if plot:
        ax1.plot(np.array(data)[burstSpikes], [0.65]*len(burstSpikes), 'kx')
        ax1.plot(np.array(data)[burstRelSpikes], [0.65]*len(burstRelSpikes), 'r+')
    
    
    bursts = []
    breaks = np.where(np.diff(burstSpikes) > 1)[0]

    if verbose > 1:
        print "\n\nFilter: Min Burst Length"

    for ix in range(len(breaks)):
        if verbose > 2:
            print "\n%i: %i >> %i" % (ix, burstSpikes[breaks[ix-1]+1], burstSpikes[breaks[ix]])
            
        if ix > 0:
            bursts.append([burstSpikes[breaks[ix-1]+1], burstSpikes[breaks[ix]]])
        else:
            bursts.append([burstSpikes[0], burstSpikes[breaks[ix]]])
            
    bursts.append([breaks[-1]+1, burstSpikes[-1]])
       
    validBursts = np.where(np.diff(bursts) >= minLen)[0]
    
    bursts = np.array(bursts)[validBursts]
    
    
    burstsFiltered = []
    
    if verbose > 1:
        print "\n\nFilter: Burst Related Spikes"
        
    for ix, burst in enumerate(bursts):
        
        if verbose > 2:
            print "\n%i: %i >> %i" % (ix, burst[0], burst[1])
            
        burstsFiltered.append(copy.copy(burst))
        ix = np.where(burstRelSpikes - burst[0] == -1)[0]
        if len(ix) > 0:
            if verbose > 3:
                print "Found Leading Burst-Related Spike! (t, dt) = (%.3f, %.3f)" % (data[burstRelSpikes[ix]], (data[burst[0]] - data[burstRelSpikes[ix]])*1000.)
                print "(%i, %i)" % (data[burst[0]], data[burstRelSpikes[ix]])
                
            ix = ix[0]
            if (data[burst[0]] - data[burstRelSpikes[ix]])*tScale <= burstRelThresh:
                if ix > 0:
                    while burstRelSpikes[ix] - burstRelSpikes[ix-1] == 1:
                        if ix > 0:
                            ix -= 1
                        else:
                            break
                        
                if verbose > 3:
                    print "Adding: %i" % burstRelSpikes[ix]
                    
                burstsFiltered[-1][0] = burstRelSpikes[ix]
            
        ix = np.where(burstRelSpikes - burst[1] == 1)[0]
        if len(ix) > 0:
            if verbose > 3:
                print "Found Following Burst-Related Spike! (t, dt) = (%.3f, %.3f)" % (data[burstRelSpikes[ix]], (data[burstRelSpikes[ix]] - data[burst[1]])*1000.)
                print "(%i, %i)" % (burstRelSpikes[ix], burst[1])
                
            ix = ix[0]
            if (data[burstRelSpikes[ix]] - data[burst[1]])*tScale <= burstRelThresh:
                if ix < len(burstRelSpikes) - 1:
                    while burstRelSpikes[ix+1] - burstRelSpikes[ix] == 1:
                        if ix < len(burstRelSpikes) - 2:
                            ix += 1
                        else:
                            break
                        
                if verbose > 3:
                    print "Adding: %i" % burstRelSpikes[ix]
                    
                burstsFiltered[-1][1] = burstRelSpikes[ix]
            
    
    burstsFinal = [list(burstsFiltered[0])]

    for burst in burstsFiltered:
        if burst[0] <= burstsFinal[-1][1]:
            burstsFinal[-1][1] = burst[1]
        else:
            burstsFinal.append(list(burst))
            
    burstsFinal = np.array(burstsFinal)  
    
    if plot:
        for burst in bursts:
            ax1.plot(np.array([data[burst[0]], data[burst[1]]]), [0.75, 0.75], 'b-', linewidth=2)
            
        for burst in burstsFiltered:
            ax1.plot(np.array([data[burst[0]], data[burst[1]]]), [0.85, 0.85], 'r-', linewidth=2)
            
        for burst in burstsFinal:
            ax1.plot(np.array([data[burst[0]], data[burst[1]]]), [0.95, 0.95], 'm-', linewidth=2)    
    
    return {'bursts': np.array(data)[burstsFinal], 'pars': {'alph1': alph1, 'alph2': alph2, 'burstThresh': burstThresh, 'burstRelThresh': burstRelThresh}}
    

    
