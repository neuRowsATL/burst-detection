The Extended Hill-Valley Method is a spike train classification algorithm that will detect bursts and tonic spiking in neural activity.

_do_analysis.py
This is the script that was used to analyze a sample set of spike trains that were used to illustrate the performance of the algorithm. The script loads data from data files and analyzes the data using three methods: 1) Extended Hill-Valley, 2) Poisson Surprise, and 3) Cumulative Moving Average. After analyzing the spike trains with the three methods, the script uses the Jaccard index to compare the performance of each method to a set of burst and tonic spiking events that were detected by visual inspection.

lib_final_ehv.py
The main function used to run the Extended Hill-Valley analysis method is do_ehv(data, **kwargs). Briefly, the algorithm generates a smoothed, history-dependent analysis signal that results in hills and valleys as the frequency of neural activity increases and decreases, respectively. The algorithm is designed to accentuate differences between low, moderate, and high firing frequencies. Bursts and tonic spiking activity is detected based on the characteristics of sequential hills and valleys based on a ratio of their height to width.

ALGORITHM PARAMETERS
Convolution & Smoothing Parameters
widthConv                 Width of exponential decay convolution function
tau                       Rate of decay of convolution function (See note below)
amp                       Amplitude of convolution function

widthGauss                Width of Gaussian smoothing kernel

Note: By default, the convolution function uses a hard-coded decay rate that is based on the convWidth parameter. This guarantees that the kernel reaches 0. You can change this behavior within the function using commenting so that the tau parameter is used instead. Note that using tau may result in a convolution kernal that does not reach 0 within the relevant window.


Burst Detection
burst_peak_ratio          Minimum ratio used to qualify height-to-width ratio as burst
burst_exclusion           Threshold used to exclude burst as analysis signal fluctuates between hills & valleys
burst_perc_peak           Lower threshold used to determine whether burst event can be terminated


Tonic Spiking Detection
tonic_perc_peak           Minimum ratio used to qualify initial height-to-width ratio as tonic spiking
tonic_min_perc            Minimum value of analysis signal used to initiate tracking of tonic spiking event
tonic_peak_ratio          Lower threshold used to determine whether tonic spiking event can be terminated
tonic_std_max             Threshold on variability (measured by standard deviation) of successive peaks and troughs to determine whether
                          to terminate tonic spiking event.


Other Parameters
peaks_adj_thresh          Threshold used to determine whether to scale peaks and troughs if amplitude of analysis signal is too big


Other libraries in the repository include a script to do burst detection using the Cumulative Moving Average method and the Poisson Surprise method.
