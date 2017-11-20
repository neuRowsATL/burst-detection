import numpy as np
import sys

import pickle
import h5py

import copy

import scipy.stats as stat

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.close('all')
plt.ion()
        
        
def do_poisson(spikeEvs, minBurstLen=3, maxInBurstLen=10, maxBurstIntStart=0.5, maxBurstIntTerm=2.0, surprise=3, verbose=0, log=None):    
    
    # minBurstLen
    ## Minimum length of spikes to be considered a burst
    
    # maxInBurstLen
    ## Maximum number of spikes within a burst that are evaluated
    ## for increasing Poisson Surprise
    
    # maxBurstIntStart
    ## Maximum ISI required to initiate a burst
    
    # maxBurstIntTerm
    ## Maximum ISI allowed to terminate a burst
    
    # surprise
    ## Minimum Poisson Surprise value
    
    spikeISI = np.diff(spikeEvs)
    
    maxSpikeIntStart = np.average(spikeISI)*maxBurstIntStart
    maxSpikeIntTerm = np.average(spikeISI)*maxBurstIntTerm
    
    avgRate = np.average(1./spikeISI)

    #print "(Min, Max) = (%.3f, %.3f)" % (np.min(spikeISI), np.max(spikeISI))
    

    bursts = []
    activeBurst = [-1,-1]

    ix = 0
    while ix < len(spikeISI):
        isi = spikeISI[ix]
        
        # Initiate burst sequence
        ## Find a spike with ISI < {minSpikeIntStart}
        if activeBurst[0] == -1:
            if isi <= maxSpikeIntStart:
                activeBurst = [ix, ix]
                
                burstExtendForward = 0
                burstRemoveFront = 0
                
                burstExtendForwardSurprise = 0
            
                if verbose > 0:
                    print "\n\n===== ===== ===== ===== ====="
                    print "Initiating Burst @ t=%.3f..." % spikeEvs[ix]
                    
                if verbose > 2:
                    print "\n1: (%i, %i) @ (%.3f, %.3f)" % (activeBurst[0], activeBurst[1], spikeEvs[activeBurst[0]], spikeEvs[activeBurst[1]])

            else:
                if verbose > 1:
                    print "\n--> ISI exceeds max burst onset ISI"
            
        else:
            # Test for minimum burst criterion
            ## {minBurstLen} consecutive spikes have ISI of (AVG ISI)*minBurstIntStart
            
            if isi <= maxSpikeIntTerm:
                if activeBurst[1] - activeBurst[0] < minBurstLen + 1:                    
                    if isi <= maxSpikeIntStart:
                        activeBurst[1] = ix
                        
                        if verbose > 2:
                            print "\n2: (%i, %i) @ (%.3f, %.3f)" % (activeBurst[0], activeBurst[1], spikeEvs[activeBurst[0]], spikeEvs[activeBurst[1]])

                        if verbose > 0:
                            print "Extending initial burst sequence..."
                        
                    else:
                        if verbose > 1:
                            print "/n--> Initial burst sequence not long enough..."
                        ix = activeBurst[0]
                        activeBurst = [-1,-1]

                else: 
                    # Test for Poisson Surprise criterion in forward direction
                    evCount = activeBurst[1] - activeBurst[0]
                    evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]
                                        
                    # Calculate probability that n or more spikes are found in sample
                    ## survivorFunc(evCount - 1) is equivalent to "n or more"
                    ## survivorFunc = 1 - cumDistFunc
                    ## cumDistFunc = probability of n or fewer events
                    sf_old = -stat.poisson.logsf(evCount-1, evTime*avgRate)                    
                    
                    temp_ix = ix+1
                    evCount = temp_ix - activeBurst[0]
                    evTime = spikeEvs[temp_ix] - spikeEvs[activeBurst[0]]
                    
                    sf = -stat.poisson.logsf(evCount-1, evTime*avgRate)                    

                    if verbose > 0:
                        print "\nTesting Poisson Surprise: (%i, %.5f), %.3f (%.3f)" % (evCount, evTime*avgRate, sf, burstExtendForwardSurprise) 
                        
                    if verbose > 1:
                        print "Checking Poisson Surprise forward %i..." % burstExtendForward
                        
                    if verbose > 2:
                        print "\n3: (%i, %i) @ (%.3f, %.3f)" % (activeBurst[0], temp_ix, spikeEvs[activeBurst[0]], spikeEvs[temp_ix])
                        
                    

                    if verbose > 3:
                        #print "CDF (old, new): (%.3f, %.3f)" % (-np.log(cdf_old), -np.log(cdf))
                        print "SF (old, new): (%.3f, %.3f), %.3f" % (sf_old, sf, burstExtendForwardSurprise)
                
                    # Check up to 10 spikes forward to see if Surprise value increases
                    if sf > burstExtendForwardSurprise and burstExtendForward < maxInBurstLen:
                        activeBurst[1] = ix
                        if verbose > 2:
                            print "Extending burst sequence: (%i, %i)" % (activeBurst[0], activeBurst[1])                        
                        
                        burstExtendForward = 0
                        burstExtendForwardSurprise = sf
                        
                    elif burstExtendForward < maxInBurstLen:
                        #activeBurst[1] = ix
                        if verbose > 2:
                            print "Searching forward: (%i, %i)" % (activeBurst[0], ix+1)                        
                        
                        burstExtendForward += 1
                        
                    else:
                        # Test for Poisson Surprise criterion when removing spikes from beginning of burst
                        
                        if verbose > 0:
                            print "\nRemoving spikes from beginning..."
                            
                        burstRemoveFront = 0
                        i = 1
                        while burstRemoveFront < maxInBurstLen:
                             
                            sf_old = -stat.poisson.logsf(evCount-1, evTime*avgRate)
                            
                            temp_ix = activeBurst[0] + i
                            evCount = activeBurst[1] - temp_ix
                            evTime = spikeEvs[activeBurst[1]] - spikeEvs[temp_ix]
                                                   
                            sf = -stat.poisson.logsf(evCount-1, evTime*avgRate)
                            
                            if verbose > 0:
                                print "Testing Poisson Surprise: (%i, %.5f), %.3f" % (evCount, evTime*avgRate, sf)                              

                            if verbose > 3:
                                #print "CDF (old, new): (%.3f, %.3f)" % (-np.log(cdf_old), -np.log(cdf))
                                print "SF (old, new): (%.3f, %.3f)" % (sf_old, sf)
                             
                            if sf < sf_old:
                                if verbose > 1:
                                    print "Removing from beginning %i..." % i
                                
                                burstRemoveFront = 0
                                i += 1
                                continue
                             
                            else:
                                if verbose > 1:
                                    print "Surprise decreased..."
                                    print "End of burst?"
                                
                                activeBurst[0] = temp_ix
                                break
                            
                        if verbose > 0:
                            print "End of burst!!!"
                            print "\n----- ----- ----- ----- -----"
                        
                        # Check that the surprise value meets minimum surprise parameter and burst contains enough spikes
                        if -np.log(sf) > surprise and activeBurst[1] - activeBurst[0] > minBurstLen:
                            if verbose > 1:
                                print "Poisson is surprised!"
                            bursts.append(np.array(spikeEvs)[activeBurst])
                        else:
                            if verbose > 1:
                                print "Failed to meet Poisson Surprise criterion..."
                            ix = activeBurst[0]
                            
                        activeBurst = [-1,-1]
                        
            
            else:
                if verbose > 1:
                    print "\n--> ISI exceeds max in-burst ISI"
                    print "\n----- ----- ----- ----- -----"
                                 
                # Check that burst sequence meets minimum burst length                    
                if activeBurst[1] - activeBurst[0] > minBurstLen:
                    if verbose > 0:
                        print "End of burst!!!"
                        print "\n----- ----- ----- ----- -----"
                
                    # Test for Poisson Surprise criterion in backwards direction
                    ## Extend burst sequence backwards
                    
                    if verbose > 0:
                        print "Removing spikes from beginning..."
                        
                    burstRemoveFront = 0
                    i = 1
                    while burstRemoveFront < maxInBurstLen:
                        
                        evCount = activeBurst[1] - activeBurst[0]
                        evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]                        
                        
                        sf_old = -stat.poisson.logsf(evCount-1, evTime*avgRate)
                        
                        temp_ix = activeBurst[0] + i
                        evCount = activeBurst[1] - temp_ix
                        evTime = spikeEvs[activeBurst[1]] - spikeEvs[temp_ix]
                        
                        sf = -stat.poisson.logsf(evCount-1, evTime*avgRate)
                        
                        if verbose > 0:
                            print "Testing Poisson Surprise: (%i, %.5f), %.3f" % (evCount, evTime*avgRate, sf)                         

                        if verbose > 3:
                            #print "CDF (old, new): (%.3f, %.3f)" % (-np.log(cdf_old), -np.log(cdf))
                            print "SF (old, new): (%.3f, %.3f)" % (sf_old, sf)
                        
                        if sf < sf_old:
                            if verbose > 2:
                                print "Removing from beginning %i..." % i
                            
                            i += 1
                            continue
                            
                        else:
                            if verbose > 1:
                                print "Surprise decreased..."
                                print "End of burst?"
                            
                            activeBurst[0] = temp_ix
                            break

                    
                    # Check that burst sequence meets Poisson Surprise criterion
                    
                    evCount = activeBurst[1] - activeBurst[0]
                    evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]                        
                    
                    sf = -stat.poisson.logsf(evCount-1, evTime*avgRate)                    

                    if verbose > 0:
                        print "Testing Poisson Surprise: (%i, %.5f), %.3f" % (evCount, evTime*avgRate, sf) 
                        
                    if sf >= surprise and activeBurst[1] - activeBurst[0] > minBurstLen:
                        if verbose > 1:
                            print "Poisson is surprised!"
                            
                        bursts.append(np.array(spikeEvs[activeBurst]))
                    else:
                        if verbose > 1:
                            print "Failed to meet Poisson Surprise criterion..."
                            
                        ix = activeBurst[0]
                else:
                    if verbose > 0:
                        print "Burst sequence too short..."
                    ix = activeBurst[0]
                    
                activeBurst = [-1,-1]
                

        ix += 1
        
        if ix == len(spikeEvs)-1:
            evCount = activeBurst[1] - activeBurst[0]
            evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]
            sf = -stat.poisson.logsf(evCount-1, evTime*avgRate)
            
            if verbose > 5:
                print "Final Poisson Surprise test: (%i, %.3f), %.3f" % (evCount, evTime*avgRate, sf)            
                
            if sf >= surprise and activeBurst[1] - activeBurst[0] > minBurstLen:
                bursts.append(np.array(spikeEvs)[activeBurst])
            

    if len(bursts) > 0:
        burstsFiltered = [bursts[0]]
        for i in range(len(bursts)-1):
            if burstsFiltered[-1][1] >= bursts[i][0]:
                burstsFiltered[-1][1] = bursts[i][1]
            else:
                burstsFiltered.append(bursts[i])
                
    else:
        burstsFiltered = []
        
    return {'bursts': burstsFiltered,
            'pars': {'minBurstLen': minBurstLen,
                     'maxInBurstLen': maxInBurstLen,
                     'maxBurstIntStart': maxBurstIntStart,
                     'maxBurstIntTerm': maxBurstIntTerm,
                     'surprise': surprise},
            'calcPars': {'avgRate': avgRate,
                         'maxSpikeIntStart': maxSpikeIntStart,
                         'maxSpikeIntTerm': maxSpikeIntTerm} 
            }


"""
bursts = burstDetect(data[analysisChan]['data'],
                     minBurstLen=3, maxInBurstLen=10,
                     maxBurstIntStart=0.5, maxBurstIntTerm=2.0,
                     surprise=0.1, verbose=2)
"""

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



poisson_pars = {'minBurstLen': 7.,
                'maxInBurstLen': 8.,
                'maxBurstIntStart': 3.,
                'maxBurstIntTerm': 2.,
                'surprise': 0.6
                }   

#data = data/1000.
results = do_poisson(data, verbose=2, **poisson_pars)


print "Running analysis..."
print "Making figures..."
fig = plt.figure()
fig.suptitle(fname)

axRaster = fig.add_subplot(111)
#axSignal = fig.add_subplot(212, sharex=axRaster)

axRaster.vlines(data, ymin=0.1, ymax=0.2)

yline = 0.25
for burst in results['bursts']:
    axRaster.plot(burst, [yline]*2, color='k', linewidth=2)
    
#for tonic in results['tonics']:
    #axRaster.plot(tonic, [yline]*2, color='k', linewidth=4)
    
axRaster.set_ylim([0, yline+0.2])


#axSignal.plot(results['ts']/1000., results['signal'], 'b-')

plt.draw()

#pickle.dump(results, open('/Users/brycechung/Google Drive/_Research/Publications/Neural Activity Classify/data/Final Analysis/data/results-%s-PS.dat' % series, 'w'))
"""