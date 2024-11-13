import numpy as n
import matplotlib.pyplot as plt
import goldcodes as gc
import scipy.signal.windows as sw
import scipy as s

#
# Cold start GPS satellite detector.
# Find doppler, prn, and delay of all 32 satellites in the constellation
#

# this is the sample-rate that the receiver is running at
sr=10e6
# this is how the delay is decimated
dec=10

# low pass filter for digital signal
def lpf(fc=0.8e6,sr=10e6,N=500):
    om0 = 2*n.pi*fc/sr
    m=n.arange(-N,N)+1e-6
    return(n.array(sw.hann(len(m))*n.sin(m*om0)/(n.pi*m),dtype=n.float32))


# 10 MHz sample-rate data in complex64
f=open("signal_source.dat","rb")
# usrp capture
#f=open("usrp.dat","rb")
#f=open("airspy2.dat","rb")

# +/- 5 kHz
n_samples=20*10000
fk=n.fft.fftshift(n.fft.fftfreq(n_samples,d=1/10e6))
dops=fk[n.where(n.abs(fk)<=5e3)[0]]

n_dops=len(dops)

dop_vecs=n.zeros([n_dops,n_samples],dtype=n.complex64)
for di in range(n_dops):
    dop_vecs[di,:]=n.exp(1j*2*n.pi*dops[di]*n.arange(n_samples)/sr)

n_coh=4
GCM=gc.get_gps_codes(n_samples,n_coh=n_coh)
n_sats=GCM.shape[0]
#n_sats=len(sats)

# read and discard garbage from the start
z=n.fromfile(f,count=n_samples,dtype=n.complex64)

w=lpf()
W=s.fft.fft(w,n_samples)

# this is how long the code is
code_length = 10e6*1023/1.023e6
code_lengthi = int(n.ceil(10e6*1023/1.023e6))
clidx=n.arange(code_lengthi,dtype=int)

# whiten in time
scale_amp=True
# remove spikes
notch_spikes=True

fo=open("results2.txt","w")

import time
while True:
    t0=time.time()
    z=n.fromfile(f,count=n_samples,dtype=n.complex64)
    n_reps=int(n.floor(n_samples/code_lengthi-1))
    
    # remove DC 
    z=z-n.mean(z)
    
    if notch_spikes:
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
        zp=z*n.conj(z)    
        zp=n.convolve(n.repeat(1/100,100),zp,mode="same")
        z=z/n.sqrt(zp)

    if False:
        plt.plot(z.real)
        plt.plot(z.imag)
        plt.show()
    if len(z)!=n_samples:
        break
    
    # lpf
    z=s.fft.ifft(s.fft.fft(z)*n.conj(W))

    #    plt.plot(z[0:1000].real)
    #    plt.show()
             
#    n_sats=len(sats)
    MFI=n.zeros([n_sats,n_dops,code_lengthi],dtype=n.float32)
    
    for di in range(n_dops):
        # for all doppler shifts, remove doppler shift
        zdc=z*dop_vecs[di,:]
        ZDC=s.fft.fft(zdc)
        import os
        # scipy can paralelize fft now!
        CC=s.fft.ifft(ZDC[None,:]*GCM,axis=1,workers=os.cpu_count())
        PWR=n.real(CC*n.conj(CC))

        for ri in range(n_reps):
            MFI[:,di,:]+=PWR[:,clidx+int(ri*code_length)]
        
        if False:
            for ci in range(n_sats):
                #print("doppler %d code %d"%(di,ci))
                # for all prns, concolve code with signal vector at DC.

                # convolution (time domain return)
                cc=n.fft.ifft(ZDC*sats[ci]["GC"])
                pwr=(cc*n.conj(cc)).real

                # incoherently integrate N repetitions of the code
                for ri in range(n_reps):
                    MFI[ci,di,:]+=pwr[clidx + int(ri*code_length)]
                    
    # marginalize in code and time to get noise floor of each doppler bin
    nfloor=n.mean(MFI,axis=(0,2))

    for ci in range(n_sats):
        dopi,deli=n.unravel_index(n.argmax(MFI[ci,:,:]/nfloor[:,None]),MFI[ci,:,:].shape)
        print("PRN %d C/N %1.2f (dB-Hz) dop %1.2f delay %d"%(ci,10.0*n.log10( (1000/n_coh)*(MFI[ci,dopi,deli]-nfloor[dopi])/nfloor[dopi]),dops[dopi],deli))

        fo.write("%1.2f %1.2f %d "%(10.0*n.log10((1000/n_coh)*(MFI[ci,dopi,deli]-nfloor[dopi])/nfloor[dopi]),dops[dopi],deli))
        if False:
            plt.imshow(10.0*n.log10( (1e3*MFI[ci,::-1,:]-nfloor[::-1,None])/nfloor[::-1,None]),aspect="auto",extent=[0,code_lengthi,n.min(dops),n.max(dops)])
            plt.xlabel(r"Delay (0.1 $\mu$s samples)")
            plt.ylabel(r"Doppler (Hz)")
            plt.title(ci)
            plt.show()
    fo.write("\n")
    fo.flush()
    t1=time.time()
    print("Processing speed: %1.2f x realtime"%((t1-t0)/(n_samples/sr)))
fo.close()
