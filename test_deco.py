import numpy as n
import matplotlib.pyplot as plt
import goldcodes as gc
import scipy.signal.windows as sw
import scipy as s
import os


def reduce_max(arr,N):
    len2=int(arr.shape[1]/N)
    A2=n.zeros([arr.shape[0],len2])
    idx=n.arange(len2)*N
    for i in range(len2):
        A2[:,i]=n.max(arr[:,(i*N):(i*N+N)],axis=1)
    return(A2)

#
# Cold start GPS satellite detector (figure out what satellites are visible without any prior information)
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
fname="gps_data_2025_09_29T1459.dat"
f=open(fname,"rb")

# the code is this many samples
# this is how long the code is (in 10 MHz samples)
code_length = 10000

# process this many repetitions of the code. this is how many are averaged
# together in power. 
n_repeats=100
# this is how many samples we need to read
n_samples=n_repeats*code_length

# make sure that we have enough data
# complex float64 is 8 bytes per sample
# 10000 samples extra removed from start to avoid junk at the start
n_samples_in_file = int(n.floor(os.path.getsize(fname)/8))-10000

print("%d samples in file"%(n_samples_in_file))
if n_samples_in_file < n_samples:
    print("not enough samples in file. reducing number of samples to process to %d"%(n_samples_in_file))
    n_samples=n_samples_in_file

if n_samples < code_length:
    print("not enough samples in file")

# get doppler bins to determine a sufficient doppler search
# factor of 10 oversample 
fk=n.fft.fftshift(n.fft.fftfreq(code_length*10,d=1/10e6))

# which dopplers to search for. 
# only look for doppler shifts less than the maximum expected line of sight velocity
# between gps satellite and ground station
max_vel = 6e3
dops=fk[n.where(n.abs(fk)<=max_vel)[0]]

# how many doppler bins
n_dops=len(dops)

# doppler vectors to mix signal to baseband
# multiplication by a complex exponential is a shift in frequency!
dop_vecs=n.zeros([n_dops,code_length],dtype=n.complex64)
for di in range(n_dops):
    dop_vecs[di,:]=n.exp(1j*2*n.pi*dops[di]*n.arange(code_length)/sr)

# get all 32 codes, FFT'd so that they can be used for convolution in frequency domain
GCM=gc.get_gps_codes(code_length)
n_sats=GCM.shape[0]

# read and discard garbage from the start produced when the airspy radio is starting up, 
# the first 10000 seems to be more than enough
z=n.fromfile(f,count=10000,dtype=n.complex64)

# read good data
z=n.fromfile(f,count=n_samples,dtype=n.complex64)

# low pass filter
w=lpf()
W=s.fft.fft(w,code_length)

# code length as an integer
code_lengthi = int(code_length)
clidx=n.arange(code_lengthi,dtype=int)

# matrix to hold matched filter outputs for all dopplers and all prns
MFI=n.zeros([n_sats,n_dops,code_lengthi],dtype=n.float32)

time_idx=0
import time
for i in range(n_repeats):
    # go through all of the data
    # average delay doppler matched filter outputs together for all of the data
    # keep updating the plots every n_samples.
    # 
    # make delay doppler plots for all 32 satellites every n_samples samples
    # this is a lot of data, so this will take a while
    t0=time.time()
    z=n.fromfile(f,count=code_length,dtype=n.complex64)

    # remove DC 
    z=z-n.mean(z)
    # inverse weight average based on noise power
    wgt=1/n.sum(n.abs(z)**2.0)
        
    # lpf the signal. the C/A code is BPSK at 1.023 MHz, so a low pass filter at 0.8 MHz is good
    # this also helps to reduce high frequency noise that would hurt the correlation
    z=s.fft.ifft(s.fft.fft(z)*n.conj(W))

    for di in range(n_dops):
        # for all doppler shifts, remove doppler shift
        zdc=z*dop_vecs[di,:]
        # fft the signal mixed to dc for convolution in freq doain
        ZDC=s.fft.fft(zdc)
        # scipy can paralelize fft now!
        # multiply with frequency domain representations of all gps codes (multiplication in frequency domain is convolution in time domain)
        # and ifft to get convolution in time domain
        # parallelize fft with workers=os.cpu_count()
        # this convolution will search for all possible delays (circular convolution)
        CC=s.fft.ifft(ZDC[None,:]*GCM,axis=1,workers=os.cpu_count())

        # power
        PWR=n.real(CC*n.conj(CC))

        # add to averaged delay doppler matched filter output
        MFI[:,di,:]+=wgt*PWR[:,:]
                            
    # marginalize in code and time to get noise floor of each doppler bin
    nfloor=n.mean(MFI,axis=(0,2))
    t1=time.time()
    print("Processed %d/%d C/A code cycles. Processing speed: %1.2f x realtime."%(i+1,n_repeats,(t1-t0)/(code_length/sr)))


# plot delay doppler search results for each satellite
# a peak in this plot indicates a satellite is present
# the peak value is proportional to the signal strength
# the doppler and delay of the peak indicate the doppler shift and code delay of the satellite
# the code delay can be used to determine the pseudorange to the satellite
# the doppler can be used to determine the line of sight velocity between the satellite and
# the receiver
for ci in range(n_sats):

    snr=(1e3*MFI[ci,:,:]-nfloor[:,None])/nfloor[:,None]

    dopi,deli=n.unravel_index(n.argmax(snr),snr.shape)
    peak_cn=10.0*n.log10( snr[dopi,deli])
    print("PRN %d C/N %1.2f (dB-Hz) dop %1.2f delay %d"%(ci,peak_cn,dops[dopi],deli))

    # plot snr
    plt.figure(figsize=(16,9))
    # the code has 1 kHz bandwidth. Normalize the power by the noise floor to get C/N0 in dB-Hz
    #dop_est,tau_est=n.unravel_index(n.argmax(snr.flatten()),snr.shape)
    # to make plots smaller, decimate the delay axis by a factor of dec
    # take maximum of each 10 samples in time
    snr_dec = reduce_max(snr, N=10)
    plt.imshow(10.0*n.log10( snr_dec[::-1,:] ) ,aspect="auto",extent=[0,code_lengthi/10,n.min(dops),n.max(dops)])
    cb=plt.colorbar()
    plt.title(r"PRN %d C/N %1.1f (dB) delay %d $\mu$s doppler %1.2f km/s n_avg=%d"%(ci+1,peak_cn,deli/10,dops[dopi],i+1))
    plt.plot(deli/10,dops[dopi],"x",color="red")
    cb.set_label("C/N (dB-Hz)")
    plt.xlabel(r"Delay (1 $\mu$s samples)")
    plt.ylabel(r"Doppler (Hz)")
    #plt.title("PRN %d"%(ci+1))
    plt.tight_layout()
    print("saving prn-mf-%03d"%(ci))
    plt.savefig("prn-mf-%03d.png"%(ci))
    plt.close()

#    time_idx+=n_samples
