from gtg import gammatonegram
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

"""
Example of using python gammatonegram code 
mcusi@mit.edu, July 30 2018
"""

## Convert sound into gammatonegram  
log_constant = 1e-80
dB_threshold = -50
sampling_rate, sound = wf.read('sample.wav')
sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate/2.))
sxx[sxx == 0] = log_constant
sxx = 20.*np.log10(sxx) #convert to dB
sxx[sxx < dB_threshold] = dB_threshold  

##Computing likelihood
"""
from scipy.stats import norm as ssn
observation, sampling_rate = wf.read('observation.wav') --> convert to gammatonegram --> sxx_observation
likelihood_weighting = 5.0 #free parameter!
loglikelihood = likelihood_weighting*np.sum(ssn.logpdf(sxx_observation, loc=sxx, scale=1))
"""

##Plot gammatonegram 
fig, ax = plt.subplots(1,1)

time_per_pixel = len(sound)/(1.*sampling_rate*sxx.shape[1])
t = time_per_pixel * np.arange(sxx.shape[1])
t_space = 50; f_space = 10;

plt.pcolormesh(sxx,vmin=dB_threshold, vmax=10, cmap='Blues')
ax.set_ylabel('Frequency (Hz)', fontsize=16)
ax.set_xlabel('Time (s)',fontsize=16)
ax.xaxis.set_ticks(range(sxx.shape[1])[::t_space])
ax.xaxis.set_ticklabels((t[::t_space]*100.).astype(int)/100.,fontsize=16)
ax.set_xbound(0,sxx.shape[1])
ax.yaxis.set_ticks(range(len(center_frequencies))[::f_space])
ax.yaxis.set_ticklabels(center_frequencies.astype(int)[::f_space],fontsize=16)
ax.set_ybound(0,len(center_frequencies))
cbar = plt.colorbar(pad=0.01)
cbar.ax.tick_params(labelsize=16)
cbar.ax.set_ylabel('Amplitude (dB)', rotation=270, labelpad=15,fontsize=16)
plt.show()
plt.close()