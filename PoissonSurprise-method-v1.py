import numpy as np
import pickle

import scipy.stats as stat

import matplotlib.pyplot as plt

plt.close('all')
plt.ion()



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


filename = 'F:/Users/bchung/Google Drive/_Research/AnimatLabDS/Burst Detect/TestData/%s' % files[13]

# 6, 15, 33, 73, 90
#filename = 'F:/Users/bchung/Google Drive/_Research/AnimatLabDS/Burst Detect/TestData/RandomSelection/%s' % files[90]
files = [filename]

analysisChan = 'Tonic Lev MN'

def burstDetect(spikeEvs, minBurstLen=3, spikePercentile=75, surprise=3, verbose=0, log=None):    
    spikeISI = np.diff(spikeEvs)
    maxInBurstInt = np.percentile(spikeISI,spikePercentile)
    
    spikeRate = 1/spikeISI
    spikeAvgRate = np.average(spikeRate)
    
    if verbose > 0:
        print "MaxInBurstInt=%.3f\nspikeAvgRate=%.3f" % (maxInBurstInt, spikeAvgRate)
        #log.write("MaxInBurstInt=%.3f\nspikeAvgRate=%.3f\n" % (maxInBurstInt, spikeAvgRate))
    
    spikeDiffInts = np.diff((spikeISI < maxInBurstInt).astype(int))
    
    
    burstOns = np.where(spikeDiffInts==1)[0]+ 1
    if spikeISI[0] < maxInBurstInt:
        burstOns = np.append([0], burstOns, axis=0)
    
    burstOffs = np.where(spikeDiffInts==-1)[0] + 1
    if len(burstOffs) < len(burstOns):
        burstOffs = np.append(burstOffs, [len(spikeEvs)-1], axis=0)


    print burstOns
    print burstOffs

    archive_burst_start = []
    archive_burst_length = []
    
    
    i = 0    
    for i0 in burstOns:
        iLen = burstOffs[i] - i0
        i += 1

        if verbose > 0:
            print "\n\nOptimization on Burst Starting @ t=%.3f ms" % spikeEvs[i0]
            print "Initial Specs: %.3f (%i) - %.3f (%i)" % (spikeEvs[i0], i0, spikeEvs[i0 + iLen], i0+iLen)        
            print "Initial Length: %.3f (%i)" % (spikeEvs[i0 + iLen] - spikeEvs[i0], iLen)
            
            #log.write("\n\nOptimization on Burst Starting @ t=%.3f ms\n" % spikeEvs[i0])
            #log.write("Initial Specs: %.3f (%i) - %.3f (%i)\n" % (spikeEvs[i0], i0, spikeEvs[i0 + iLen], i0+iLen))
            #log.write("Initial Length: %.3f (%i)\n" % (spikeEvs[i0 + iLen] - spikeEvs[i0], iLen))
        
        j = 0
        S_old = 0
        if verbose > 0:
            print "\nFORWARD ITERATION FROM END"
            #log.write("\nFORWARD ITERATION FROM END\n")
            
        while i0+iLen+j < len(spikeEvs)-1:
            if verbose > 1:
                print "ISI=%.3f (%i) >> n=%i, mu=%.3f" % (spikeISI[i0+iLen+j], (i0+iLen+j), iLen+j, (spikeEvs[i0+iLen+j] - spikeEvs[i0])*spikeAvgRate)
                #log.write("ISI=%.3f (%i) >> n=%i, mu=%.3f\n" % (spikeISI[i0+iLen+j], (i0+iLen+j), iLen+j, (spikeEvs[i0+iLen+j] - spikeEvs[i0])*spikeAvgRate))
            if spikeISI[i0+iLen+j] < maxInBurstInt:
                S = -np.log(stat.poisson.cdf(iLen+j, (spikeEvs[i0+iLen+j] - spikeEvs[i0])*spikeAvgRate))
                if verbose > 1:
                    print "Length %.3f >> S=%.5f" % ((spikeEvs[i0+iLen] - spikeEvs[i0+j]), S)            
                    #log.write("Length %.3f >> S=%.5f\n" % ((spikeEvs[i0+iLen] - spikeEvs[i0+j]), S))
                if S > S_old:
                    S_old = S
                else:
                    break
                
                j += 1
            else:
                break
            
        iLen += j        

        if verbose > 0:
            print "\nChecking addition of events to end"
            #log.write("\nChecking addition of events to end\n")
        if verbose > 1:
            print "SpikeEvs len: %i (i0=%i, iLen=%i)" % (len(spikeEvs), i0, iLen)
            #log.write("SpikeEvs len: %i (i0=%i, iLen=%i)\n" % (len(spikeEvs), i0, iLen))
            
        n = 0
        for m in range( max(min(len(spikeEvs)-(i0+iLen), 10) - 1, 0) ):
            if verbose > 1:
                print "ISI %.3f (%i)" % (spikeISI[i0+m+iLen], i0+m+iLen)
                #log.write("ISI %.3f (%i)\n" % (spikeISI[i0+m+iLen], i0+m+iLen))
            
            if spikeISI[i0+m+iLen] < maxInBurstInt:
                S = -np.log(stat.poisson.cdf(iLen+m, (spikeEvs[i0+iLen+m] - spikeEvs[i0])*spikeAvgRate))
                if verbose > 1:
                    print "Adding %i: %.3f >> S=%.5f (ix=%i)" % (m, (spikeEvs[i0+iLen+m]-spikeEvs[i0]), S, i0+iLen+m)
                    #log.write("Adding %i: %.3f >> S=%.5f (ix=%i)\n" % (m, (spikeEvs[i0+iLen+m]-spikeEvs[i0]), S, i0+iLen+m))
                if S > S_old:
                    n = m
                    S_old = S
            else:
                break
        
        iLen += n-1
        if verbose > 0:
            print "Adding to end: %i" % (j + n)
            #log.write("Adding to end: %i\n" % (j + n))
        
        k = 0
        S_old = 0
        
        if verbose > 0:
            print "\nFORWARD ITERATION FROM BEGINNING"
            #log.write("\nFORWARD ITERATION FROM BEGINNING\n")
        for k in range(0, iLen+j):
            S = -np.log(stat.poisson.cdf(iLen, (spikeEvs[i0+iLen] - spikeEvs[i0+k])*spikeAvgRate))
            if verbose > 1:
                print "Start %.3f >> S=%.5f (Len=%.3f)" % (spikeEvs[i0+k], S, (spikeEvs[i0+iLen] - spikeEvs[i0+k]))
                #log.write("Start %.3f >> S=%.5f (Len=%.3f)\n" % (spikeEvs[i0+k], S, (spikeEvs[i0+iLen] - spikeEvs[i0+k])))
            if S > S_old:
                k += 1
                S_old = S
                continue
            else:
                break
            
        if verbose > 0:
            print "Removing from beginning: %i" % (k-1)
            #log.write("Removing from beginning: %i\n" % (k-1))
        
        iLen -= k-1
        i0 += max(k-1, 0)

        if verbose > 0:
            print "Final Specs: %.3f (%i) - %.3f (%i) >> Len=%.3f" % (spikeEvs[i0], i0, spikeEvs[i0+iLen], i0+iLen, spikeEvs[i0+iLen]-spikeEvs[i0])
            #log.write("Final Specs: %.3f (%i) - %.3f (%i) >> Len=%.3f\n" % (spikeEvs[i0], i0, spikeEvs[i0+iLen], i0+iLen, spikeEvs[i0+iLen]-spikeEvs[i0]))
        if iLen < minBurstLen:
            if verbose > 0:
                print "EXCLUDING: Too short"
                #log.write("EXCLUDING: Too short\n")
            continue
        archive_burst_start.append(i0)
        archive_burst_length.append(iLen)
        
            
        #print "\nFinal Specs: %.3f - %.3f\n" % (spikeEvs[archive_burst_start[-1]], spikeEvs[archive_burst_start[-1] + archive_burst_length[-1]])
    
    burstInfo = []
    #print "Spike evs len: %i" % len(spikeEvs)
    #print "On len: %i" % len(archive_burst_start)
    #print "Length len: %i" % len(archive_burst_length)
    for i in range(len(archive_burst_start)):        
        print "Start: %.3f - %.3f (%i - %i)" % (spikeEvs[archive_burst_start[i]], spikeEvs[archive_burst_start[i]+archive_burst_length[i]], archive_burst_start[i], archive_burst_start[i]+archive_burst_length[i])
        burstInfo.append( np.array([ spikeEvs[archive_burst_start[i]], spikeEvs[ archive_burst_start[i] + archive_burst_length[i] ] ]) )

    return burstInfo





for filename in files:
    print "\n\nLoading data for: %s" % filename
    data = pickle.load(open(filename, 'r'))    
    levBursts = burstDetect(data[analysisChan]['data'], spikePercentile=95, verbose=1)
    
    fig = plt.figure(figsize=(18,9))
    fig.suptitle(filename + '\nPoisson Surprise Method', fontsize=18)
    
    ax_cbJoint = fig.add_subplot(211)
    ax_levTon = fig.add_subplot(212, sharex=ax_cbJoint)

    print "Plotting data..."
    ax_cbJoint.plot(data['Time']['data'], data['CB_joint']['data'], 'g-')
    ax_levTon.vlines(data[analysisChan]['data'], ymin=0.2, ymax=0.8, color='b')
    ax_levTon.set_ylim([0,1])

    for burst in levBursts:
        ax_levTon.plot(burst, np.array([1.,1.])*0.1, color='b', linewidth=4)