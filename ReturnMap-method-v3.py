import pickle
import copy

import numpy as np
from scipy import signal, stats

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.close('all')


files = ['FinalDissertationModel_Standalone-00054.dat',
 'FinalDissertationModel_Standalone-00247.dat',
 'FinalDissertationModel_Standalone-00288.dat',
 'FinalDissertationModel_Standalone-00329.dat',
 'FinalDissertationModel_Standalone-00490.dat',
 'FinalDissertationModel_Standalone-00520.dat',
 'FinalDissertationModel_Standalone-00540.dat',
 'FinalDissertationModel_Standalone-00742.dat',
 'FinalDissertationModel_Standalone-00793.dat',
 'FinalDissertationModel_Standalone-00815.dat',
 'FinalDissertationModel_Standalone-00825.dat',
 'FinalDissertationModel_Standalone-00847.dat',
 'FinalDissertationModel_Standalone-00871.dat',
 'FinalDissertationModel_Standalone-00901.dat',
 'FinalDissertationModel_Standalone-01242.dat',
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
 'FinalDissertationModel_Standalone-03395.dat',
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
 'FinalDissertationModel_Standalone-07487.dat',
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
 'FinalDissertationModel_Standalone-09105.dat',
 'FinalDissertationModel_Standalone-09474.dat',
 'FinalDissertationModel_Standalone-09646.dat',
 'FinalDissertationModel_Standalone-09748.dat',
 'FinalDissertationModel_Standalone-09905.dat',
 'FinalDissertationModel_Standalone-09965.dat',
 'FinalDissertationModel_Standalone-10253.dat',
 'FinalDissertationModel_Standalone-10305.dat',
 'FinalDissertationModel_Standalone-10425.dat',
 'FinalDissertationModel_Standalone-10758.dat']

files = ['FinalDissertationModel_Standalone-00141.dat',
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

#filename = 'FinalDissertationModel_Standalone-04407.dat'
# 6, 15, 33, 73, 90
filename = files[13] # STOPPED @ 32
analysisChan = 'Tonic Lev MN'


print "Loading data for: %s" % filename
#data = pickle.load(open('F:/Users/bchung/Google Drive/_Research/AnimatLabDS/Burst Detect/TestData/RandomSelection/%s' % filename, 'rb'))
data = pickle.load(open('F:/Users/bchung/Google Drive/_Research/AnimatLabDS/Burst Detect/TestData/%s' % filename, 'rb'))

x = data[analysisChan]['data']
y = (np.max(x) - np.min(x))*1000/len(x)

print "expected ISI: %.3f" % y

#isi = np.log(np.diff(data[analysisChan]['data']))
isi = np.diff(data[analysisChan]['data'])

print "mean ISI: %.3f" % np.mean(isi)
print "\n"

## Plot multiple return mappings for comparison as offset size increases
k = 9
#factor = pow(len(data[analysisChan]['data']), 1./3)
factor = 3
ks = np.arange(k)+1

dim = np.ceil(np.sqrt(k))


# Definition for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]
rect_text = [left_h, bottom_h, 0.2, 0.2]

# Make figure
figHist = plt.figure(figsize=(15,15))


# Make axes
axScatter = figHist.add_axes(rect_scatter)
axHistx = figHist.add_axes(rect_histx, sharex=axScatter)
axHisty = figHist.add_axes(rect_histy, sharey=axScatter)
axScatter.set_xlabel(filename)

axText = figHist.add_axes(rect_text)
axText.set_xticks([])
axText.set_yticks([])

# Format axes
#axHistx.xaxis.set_ticklabels([])
#axHisty.yaxis.set_ticklabels([])
plt.setp(axHistx.get_xticklabels()+axHisty.get_yticklabels(), visible=False)


print "Making histogram..."
# Make histograms
xN, xBins, xPatches = axHistx.hist(isi[:-1], bins=100)
yN, yBins, yPatches = axHisty.hist(isi[1:], bins=100, orientation='horizontal')


csum = np.array([np.sum(xN[:ix]) for ix in range(len(xN)+1)])
cma = np.array([np.sum(xN[:ix])/(ix+1) for ix in range(len(xN)+1)])
xm = np.argmax(cma)

skew = stats.skew(xN)

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
        
    
x1 = np.max(cma)*alph1
x2 = np.max(cma)*alph2

ixBurstThresh = np.where(np.diff(np.sign(cma-x1)) < 0)[0]
if len(ixBurstThresh) > 0:
    burstThresh = xBins[ixBurstThresh[-1]]
else:
    burstThresh = xBins[-1]


burstSpikes = np.where((isi[:-1] < burstThresh) & (isi[1:] < burstThresh))[0] + 1
onSpikes = np.where((isi[:-1] > burstThresh) & (isi[1:] < burstThresh))[0] + 1
offSpikes = np.where((isi[:-1] < burstThresh) & (isi[1:] > burstThresh))[0] + 1
nonBurstSpikes = np.where((isi[:-1] > burstThresh) & (isi[1:] > burstThresh))[0] + 1

isiRatio = stats.kurtosis(isi)/stats.skew(isi)


print "Plotting scatter..."
axScatter.scatter(isi[burstSpikes[:-1]-1], isi[burstSpikes[:-1]], marker='.', color='b')
axScatter.scatter(isi[onSpikes[1:]-1], isi[onSpikes[1:]], marker='.', color='g')
axScatter.scatter(isi[offSpikes[:-1]-1], isi[offSpikes[:-1]], marker='.', color='r')
axScatter.scatter(isi[nonBurstSpikes[:-1]-1], isi[nonBurstSpikes[:-1]], marker='.', color='m')
#axScatter.scatter(isi[randSpikes-1], isi[randSpikes], marker='.', color='c')

axScatter.plot(np.mean(isi[:-1]), np.mean(isi[1:]), 'ko', markersize=8)

print "Plotting stats..."
axHistx.plot(xBins, csum*np.max(xN)/np.max(csum), 'm-', linewidth=2)
axHistx.plot(xBins, cma, 'c-', linewidth=2)

#axHistx.axvline(xBins[xm], color='c', linewidth=1)
axHistx.axvline(burstThresh, color='r', linewidth=1)
#axHisty.axhline(xBins[xm], color='c', linewidth=2)
#axHistx.axvline(np.mean(isi), color='g')
#axHistx.axvline(np.mean(isi)+np.std(isi), color='g', linestyle='--')


axHistx.axhline(x1, color='r')
#axHistx.axhline(x2, color='r', linestyle='--')

axScatter.axvline(burstThresh, color='r', linewidth=1)
axScatter.axhline(burstThresh, color='r', linewidth=1)



print "Drawing charts..."
fig = plt.figure(figsize=(18,9))
fig.suptitle(filename)

ax_cbJoint = fig.add_subplot(211)
ax_levTon = fig.add_subplot(212, sharex=ax_cbJoint)


print "Plotting data..."
ax_cbJoint.plot(data['Time']['data'], data['CB_joint']['data'], 'g-')

colorCoded = True
if colorCoded:
    ax_levTon.vlines(data[analysisChan]['data'][burstSpikes], ymin=0.2, ymax=0.8, color='b')
    ax_levTon.vlines(data[analysisChan]['data'][onSpikes], ymin=0.2, ymax=0.8, color='g', linewidth=2)
    ax_levTon.vlines(data[analysisChan]['data'][offSpikes], ymin=0.2, ymax=0.8, color='r', linewidth=2)
    ax_levTon.vlines(data[analysisChan]['data'][nonBurstSpikes], ymin=0.2, ymax=0.8, color='m')
else:
    ax_levTon.vlines(data[analysisChan]['data'], ymin=0.2, ymax=0.8, color='b')

ax_levTon.set_ylim([0,1])

txtStr = "N = %i\nalpha = %.1f\n\nISI Avg = %.3f\nISI STD = %.3f\nISI Skew = %.3f\nISI Kurt = %.3f\nISI Ratio = %.3f\n\nBins Skew = %.3f\nBins Kurt = %.3f\nBins Ratio = %.3f" \
    % (len(isi)+1, alph1, np.mean(isi), np.std(isi), stats.skew(isi), stats.kurtosis(isi), stats.kurtosis(isi)/stats.skew(isi), stats.skew(xN), stats.kurtosis(xN), stats.kurtosis(xN)/stats.skew(xN))
axText.annotate(txtStr, xy=(0.1,0.1), xycoords='axes fraction', fontsize=8)

#print "CMA Factor: %.3f" % alph1
#print "ISI Ratio: %.3f" % isiRatio

#print "Plotting distance histogram..."
#plt.figure()
#plt.title(filename)
xs = np.sqrt((isi[:-1]-np.mean(isi[:-1]))**2 + (isi[1:]-np.mean(isi[1:]))**2)
#plt.hist(xs, 10)
#plt.hist(xs, 20)
#plt.hist(xs, 50)
#plt.yscale('log')


print "Plotting 3d ISI scatter..."
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(isi[:-2], isi[1:-1], isi[2:], 'b.')

#axScatter.axvline(np.mean(isi), color='g', linestyle='-')
#axScatter.axvline(np.mean(isi)+np.std(isi), color='g', linestyle='--')
#axScatter.axvline(np.mean(isi)+2*np.std(isi), color='g', linestyle='--')
#axScatter.axvline(np.mean(isi)+np.std(xs), color='g', linestyle='-.')
#axScatter.axvline(np.mean(isi)+2*np.std(xs), color='g', linestyle='-.')


print "\n"
print "CMA Peak Bin: %.3f" % xBins[xm]
print "CMA Peak: %.3f" % cma[xm]
print "CMA Alpha: %.3f" % alph1
print "Threshold: %.3f" % burstThresh
print "Smoothness: %.3f" % (np.std(np.diff(xN))/np.abs(np.mean(xN)))

halfPeak = np.where(np.diff(np.sign(cma - np.max(cma)*0.5)) < 0)[0][-1]
quartPeak = np.where(np.diff(np.sign(cma - np.max(cma)*0.25)) < 0)[0][-1]

print "\n"
print "CMA 1/2 Peak: %.3f" % cma[halfPeak]
print "Val 1/2 Peak: %.3f" % xBins[halfPeak]


print "CMA 1/4 Peak: %.3f" % cma[quartPeak]
print "Val 1/2 Peak: %.3f" % xBins[quartPeak]
#axScatter.axvline(xBins[quartPeak], color='c')
#axScatter.axhline(xBins[quartPeak], color='c')
#axHistx.axvline(xBins[quartPeak], color='c')
#axHistx.axhline(cma[quartPeak], color='c')


## Plot bursts as bars along bottom of raster axis

if onSpikes[0] > offSpikes[0] and len(onSpikes) == len(offSpikes):
    bursts = np.array([np.array(data[analysisChan]['data'])[onSpikes[:-1]], np.array(data[analysisChan]['data'])[offSpikes[1:]]]).T
elif len(onSpikes) == len(offSpikes):
    bursts = np.array([np.array(data[analysisChan]['data'])[onSpikes], np.array(data[analysisChan]['data'])[offSpikes]]).T
elif len(onSpikes) <> len(offSpikes):
    if onSpikes[0] > offSpikes[0]:
        try:
            bursts = np.array([np.array(data[analysisChan]['data'])[onSpikes], np.array(data[analysisChan]['data'])[offSpikes[1:]]]).T
        except:
            print "TOO MUCH ERROR TO HANDLE!!"
            
    
    
for burst in bursts:
    ax_levTon.plot(burst, np.array([1., 1.])*0.1, color='b', linewidth=4)

