import numpy as n
import matplotlib.pyplot as plt
import goldcodes as gc
import scipy.signal.windows as sw
import scipy as s
import os

def reduce_max(arr, N):
    # Trim length so it's divisible by N
    L = arr.shape[0] - (arr.shape[0] % N)
    arr = arr[:L]

    # Reshape to group along time axis
    new_shape = (L // N, N) + arr.shape[1:]
    arr_reshaped = arr.reshape(new_shape)

    # Take max along the chunk axis
    return arr_reshaped.max(axis=1)
#
# Cold start GPS satellite detector.
# Find doppler, prn, and delay of all 32 satellites in the constellation
#
# this is the sample-rate that the receiver is running at
sr=10e6
# this is how the delay is decimated. we don't need super high resolution delay to start with
dec=10

# low pass filter for digital signal (signal processing course)
def lpf(fc=0.8e6,sr=10e6,N=500):
    om0 = 2*n.pi*fc/sr
    m=n.arange(-N,N)+1e-6
    return(n.array(sw.hann(len(m))*n.sin(m*om0)/(n.pi*m),dtype=n.float32))


# 10 MHz sample-rate data in complex64
f=open("gps_data_2025_09_29T1459.dat","rb")

# +/- 5 kHz
# 20 code repeats are averaged together in power for the delay doppler search
n_repeats=20
n_samples=n_repeats*10000

# get doppler bins
fk=n.fft.fftshift(n.fft.fftfreq(n_samples,d=1/10e6))

# which dopplers to search for. 
# only look for doppler shifts less than the maximum expected line of sight velocity
# between gps satellite and ground station
max_vel = 6e3
dops=fk[n.where(n.abs(fk)<=max_vel)[0]]

# how many doppler bins
n_dops=len(dops)

# doppler vectors to mix signal to baseband
# multiplication by a complex exponential is a shift in frequency!
dop_vecs=n.zeros([n_dops,n_samples],dtype=n.complex64)
for di in range(n_dops):
    dop_vecs[di,:]=n.exp(1j*2*n.pi*dops[di]*n.arange(n_samples)/sr)

# get all 32 codes, FFT'd so that they can be used for convolution in frequency domain
GCM=gc.get_gps_codes(n_samples)
n_sats=GCM.shape[0]

# read and discard garbage from the start produced when the airspy radio is starting up, 
# the first 10000 seems to be more than enough
z=n.fromfile(f,count=10000,dtype=n.complex64)
# read good data
z=n.fromfile(f,count=n_samples,dtype=n.complex64)

# low pass filter
w=lpf()
W=s.fft.fft(w,n_samples)

# this is how long the code is (in 10 MHz samples)
code_length = 10e6*1023/1.023e6
# code length as an integer
code_lengthi = int(n.ceil(10e6*1023/1.023e6))
clidx=n.arange(code_lengthi,dtype=int)

# whiten in time
scale_amp=False
# remove spikes
notch_spikes=False

fo=open("results2.txt","w")

# matrix to hold matched filter outputs for all dopplers and all prns
MFI=n.zeros([n_sats,n_dops,code_lengthi],dtype=n.float32)

time_idx=0
import time
while True:
    # go through all of the data
    # average delay doppler matched filter outputs together for all of the data
    # keep updating the plots every n_samples.
    # 
    # make delay doppler plots for all 32 satellites every n_samples samples
    # this is a lot of data, so this will take a while
    t0=time.time()
    z=n.fromfile(f,count=n_samples,dtype=n.complex64)
    # how many code repeats can fit in the data
    n_reps=int(n.floor(n_samples/code_lengthi-1))


    # remove DC 
    z=z-n.mean(z)
    # inverse weight average based on noise power
    wgt=1/n.sum(n.abs(z)**2.0)
    
    if notch_spikes:
        # remove RFI that is narrow in frequency domain
        # whiten
        Z=s.fft.fft(z)
        ZP=n.convolve(n.repeat(1/100,100),Z*n.conj(Z),mode="same")
        zpt=n.median(ZP)
    
        if False:
            plt.plot(ZP)
            plt.axhline(zpt*3,color="red")
            plt.show()
        
        # notch
        Z[ZP>(3*zpt)]=0
        z=s.fft.ifft(Z)

    if scale_amp:
        # normalize amplitude. this helps if there is a lot of amplitude variation
        zp=z*n.conj(z)    
        zp=n.convolve(n.repeat(1/100,100),zp,mode="same")
        z=z/n.sqrt(zp)

    if False:
        # plot the raw voltage. useful for debugging and identifying problems with the signal
        plt.plot(z.real)
        plt.plot(z.imag)
        plt.show()
    if len(z)!=n_samples:
        break
    
    # lpf the signal. the C/A code is BPSK at 1.023 MHz, so a low pass filter at 0.8 MHz is good
    # this also helps to reduce high frequency noise that would hurt the correlation
    z=s.fft.ifft(s.fft.fft(z)*n.conj(W))

    for di in range(n_dops):
        # for all doppler shifts, remove doppler shift
        zdc=z*dop_vecs[di,:]
        # fft the signal mixed to dc
        ZDC=s.fft.fft(zdc)
        # scipy can paralelize fft now!
        # multiply with frequency domain representations of all gps codes (multiplication in frequency domain is convolution in time domain)
        # and ifft to get convolution in time domain
        # parallelize fft with workers=os.cpu_count()
        # this convolution will search for all possible delays (circular convolution)
        CC=s.fft.ifft(ZDC[None,:]*GCM,axis=1,workers=os.cpu_count())

        # power
        PWR=n.real(CC*n.conj(CC))*wgt

        # incoherently integrate N repetitions of the code
        for ri in range(n_reps):
            MFI[:,di,:]+=PWR[:,clidx+int(ri*code_length)]
                            
    # marginalize in code and time to get noise floor of each doppler bin
    nfloor=n.mean(MFI,axis=(0,2))

    # plot delay doppler search results for each satellite
    # a peak in this plot indicates a satellite is present
    # the peak value is proportional to the signal strength
    # the doppler and delay of the peak indicate the doppler shift and code delay of the satellite
    # the code delay can be used to determine the pseudorange to the satellite
    # the doppler can be used to determine the line of sight velocity between the satellite and
    # the receiver
    for ci in range(n_sats):
        dopi,deli=n.unravel_index(n.argmax(MFI[ci,:,:]/nfloor[:,None]),MFI[ci,:,:].shape)
        peak_cn=10.0*n.log10( (1000)*(MFI[ci,dopi,deli]-nfloor[dopi])/nfloor[dopi])
        print("PRN %d C/N %1.2f (dB-Hz) dop %1.2f delay %d"%(ci,peak_cn,dops[dopi],deli))

        fo.write("%1.2f %1.2f %d "%(10.0*n.log10((1000)*(MFI[ci,dopi,deli]-nfloor[dopi])/nfloor[dopi]),dops[dopi],deli))
        if True:
            plt.figure(figsize=(16,9))
            # the code has 1 kHz bandwidth. Normalize the power by the noise floor to get C/N0 in dB-Hz
            snr=(1e3*MFI[ci,::-1,:]-nfloor[::-1,None])/nfloor[::-1,None]
            #dop_est,tau_est=n.unravel_index(n.argmax(snr.flatten()),snr.shape)
            # to make plots smaller, decimate the delay axis by a factor of dec
            # take maximum of each 10 samples in time
            snr_dec = reduce_max(snr, N=10)
            plt.imshow(10.0*n.log10( snr_dec ) ,aspect="auto",extent=[0,code_lengthi/10,n.min(dops),n.max(dops)])
            cb=plt.colorbar()
            plt.title(r"PRN %d C/N %1.0f (dB) delay %d $\mu$s doppler %1.2f km/s"%(ci+1,peak_cn,deli/10,dops[dopi]))
            plt.plot(deli/10,dops[dopi],"x",color="red")
            cb.set_label("C/N (dB-Hz)")
            plt.xlabel(r"Delay (1 $\mu$s samples)")
            plt.ylabel(r"Doppler (Hz)")
            #plt.title("PRN %d"%(ci+1))
            plt.tight_layout()
            print("saving prn-mf-%03d"%(ci))
            plt.savefig("prn-mf-%03d.png"%(ci))
            plt.close()
    fo.write("\n")
    fo.flush()
    t1=time.time()
    print("Processing speed: %1.2f x realtime."%((t1-t0)/(n_samples/sr)))
#    time_idx+=n_samples
fo.close()
