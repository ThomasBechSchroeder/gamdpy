import numpy as np 
import gamdpy as gp 
from numba import jit
import math
import cmath

@jit(nopython=True) 
def __store__(dk, dmk, jk, jmk, sampletype, mass, pos, vel, ptypes, npart, index, nwaves, lbox):
   
    I = complex(0.0, 1.0)

    for k in range(nwaves):
        dk[index, k], dmk[index, k] = complex(0.0, 0.0), complex(0.0, 0.0)
        jk[index, k], jmk[index, k] = complex(0.0, 0.0), complex(0.0, 0.0)

        kwave = 2.0*math.pi*(k+1)/lbox

        for n in range(npart):
            if ptypes[n] == sampletype or sampletype == -1:
                kfac = cmath.exp(I*kwave*pos[n])
                mkfac = cmath.exp(-I*kwave*pos[n])

                dk[index, k] = dk[index, k] + mass[n]*kfac
                dmk[index, k] = dmk[index, k] + mass[n]*mkfac

                jk[index, k] = jk[index, k] + mass[n]*vel[n]*kfac
                jmk[index, k] = jmk[index, k] + mass[n]*vel[n]*mkfac


@jit(nopython=True)
def __calccorr__(dacf, jacf, dk, dmk, jk, jmk, nwaves, lvec):

    for k in range(nwaves):
        for n in range(lvec):
            for nn in range(lvec-n):
                dacf[n, k] = dacf[n,k] + dk[nn,k]*dmk[nn+n, k] 
                jacf[n, k] = jacf[n,k] + jk[nn,k]*jmk[nn+n, k]



class CalculatorHydrodynamicCorrelations:
    """
        Calculates the tranverse current autocorrelation function (jacf) and 
        the longitudinal density autocorrelation function (dacf). 

        Example: See hydrocorr.py

        Initialization variables
        - configuration: Instance of the configuration class
        - dtsample: Time between sampling (= integrator time step * steps_per_timeblock)
        - ptype: The particle type for which the correlations are calculated (default 0)   
        - nwaves: Number of wavevectors (default 10)
        - lvec: Length of the time vector - sample time span then dtsample*lvec (defulat 100) 
        - verbose: Write a bit to screen (default False)

        Output: 
            With the method read() data is written to files and returned to user 
    """
    
    def __init__(self, configuration, dtsample, ptype=0, nwaves=10, lvec=100, verbose=False):
        self.conf = configuration
        self.ptype = ptype
        self.nwaves = nwaves
        self.lvec = lvec
        self.dt = dtsample

        self.nsample = 0
        self.index = 0
        
        self.lbox = configuration.simbox.lengths[1]; 

        # Storage arrays
        self.dk = np.zeros( (lvec, nwaves), dtype=np.complex64)  
        self.dmk = np.zeros( (lvec, nwaves), dtype=np.complex64)  
        
        self.jk = np.zeros( (lvec, nwaves), dtype=np.complex64)  
        self.jmk = np.zeros( (lvec, nwaves), dtype=np.complex64)  

        # Correlation array - density (longitudinal)  
        self.dacf = np.zeros( (lvec, nwaves), dtype=np.complex64)    

        # Current density acf (transverse)
        self.jacf = np.zeros( (lvec, nwaves), dtype=np.complex64)

        
        if verbose:
            print(f"Types {self.ptype}, no. wavevectors {self.nwaves}, vector length {self.lvec}")




    def update(self):
   
         __store__(self.dk, self.dmk, self.jk, self.jmk, self.ptype,       
                   self.conf['m'][:], self.conf['r'][:,0], self.conf['v'][:,1],
                   self.conf.ptype[:], self.conf.N, self.index, self.nwaves, self.lbox)

         self.index = self.index + 1

         if self.index == self.lvec:
             __calccorr__(self.dacf, self.jacf, self.dk, self.dmk, self.jk, self.jmk, self.nwaves, self.lvec)

             self.nsample = self.nsample + 1
             self.index = 0




    def read(self, save=True, fname_dacf="dacf.dat", fname_jacf="jacf.dat"):

        if self.nsample > 0:
        
            volume = self.conf.simbox.lengths[0]*self.conf.simbox.lengths[1]*self.conf.simbox.lengths[2]

            out_dacf = np.zeros( (self.lvec, self.nwaves) )
            out_jacf = np.zeros( (self.lvec, self.nwaves) )

            file_dacf = open(fname_dacf, "w")
            file_jacf = open(fname_jacf, "w")

            for n in range(self.lvec):
                
                fac = 1.0/(self.nsample*(self.lvec-n)*volume)

                file_dacf.write("%f " % (self.dt*n))
                file_jacf.write("%f " % (self.dt*n))

                for k in range(self.nwaves):
                    out_dacf[n,k] = self.dacf[n, k].real*fac
                    out_jacf[n,k] = self.jacf[n, k].real*fac
                    file_dacf.write("%f " % (out_dacf[n,k])) 
                    file_jacf.write("%f " % (out_jacf[n,k]))

                file_dacf.write("\n")
                file_jacf.write("\n")

            file_dacf.close()
            file_jacf.close()

            return (out_dacf, out_jacf)

        else:
            return (0,0)
