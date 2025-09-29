
import numpy as n
# from:
# https://natronics.github.io/blag/2014/gps-prn/


def shift(register, feedback, output):
    """GPS Shift Register
    
    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:
    
    """
    
    # calculate output
    out = [register[i-1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]
        
    # modulo 2 add feedback
    fb = sum([register[i-1] for i in feedback]) % 2
    
    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i+1] = register[i]
        
    # put feedback in position 1
    register[0] = fb
    
    return out

# example:
#print shift(G1, [3,10], [10])
#print G1


SV = {
   1: [2,6],
   2: [3,7],
   3: [4,8],
   4: [5,9],
   5: [1,9],
   6: [2,10],
   7: [1,8],
   8: [2,9],
   9: [3,10],
  10: [2,3],
  11: [3,4],
  12: [5,6],
  13: [6,7],
  14: [7,8],
  15: [8,9],
  16: [9,10],
  17: [1,4],
  18: [2,5],
  19: [3,6],
  20: [4,7],
  21: [5,8],
  22: [6,9],
  23: [1,3],
  24: [4,6],
  25: [5,7],
  26: [6,8],
  27: [7,9],
  28: [8,10],
  29: [1,6],
  30: [2,7],
  31: [3,8],
  32: [4,9],
}


def PRN(sv):
    """Build the CA code (PRN) for a given satellite ID
    
    :param int sv: satellite code (1-32)
    :returns list: ca code for chosen satellite
    
    """
    
    # init registers
    G1 = [1 for i in range(10)]
    G2 = [1 for i in range(10)]

    ca = [] # stuff output in here
    
    # create sequence
    for i in range(1023):
        g1 = shift(G1, [3,10], [10])
        g2 = shift(G2, [2,3,6,8,9,10], SV[sv]) # <- sat chosen here from table
        
        # modulo 2 add and append to the code
        ca.append((g1 + g2) % 2)

    # return C/A code!
    return n.array(ca)

def interp(code,sr=10e6):
    # interpolate code from 1.023 MHz to sr
    T_max=len(code)/1.023e6
    n_out=T_max*sr
    tout=n.arange(n_out)/sr
    idx=n.array(n.mod(n.round(tout*1.023e6),1023),dtype=int)
    return(code[idx])

def get_gps_codes(n_samples,n_coh=1):
    """
    Get all 32 GPS C/A codes, repeated n_coh times and FFT'd
    n_coh=1 is one code cycle. this is recommended if you do not know the exact timing
    n_coh>1 is recommended if you know the code timing to improve SNR
    FFT is perfomed, as the codes will be convolved in frequency domain
    """
    sats=[]
    GCM=n.zeros([32,n_samples],dtype=n.complex64)
    for i in range(32):
        gcz=PRN(i+1)

        # n_coh repeats
        gczt=n.tile(gcz,n_coh)
        # get correct dithering of code this way
        gcit=interp(gczt)
        if len(gcit) > n_samples:        
            print("error, too long code!")
        GCM[i,:]=n.conj(n.fft.fft(gcit,n_samples))
#        sats.append({"prn":i+1,"GC":GC,"gci":gci})
    return(GCM)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    code=PRN(14)
    
    plt.plot(code)
    plt.title("GPS code %d"%(14))
    plt.show()
    print(len(code))
