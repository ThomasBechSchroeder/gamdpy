import numpy as np 
import gamdpy as gp 
from numba import jit

@jit(nopython=True) 
def __dosample__(cpart, temp, momc, pos, vel, velT, mass, ptype, thistype, npart, wbin, hbox):
    
    for n in range(npart):
        if ptype[n]==thistype:
            
            idx = int((pos[n]+hbox)/wbin)
            
            cpart[idx] = cpart[idx] + 1
            momc[idx] = momc[idx] + mass*vel[n]
            temp[idx] = temp[idx] + mass*(velT[n,0]*velT[n,0] + velT[n,1]*velT[n,1])


class CalculatorHydrodynamicProfile:
    """
    Calculates the density, streaming velocity, and kinetic temperature profiles. 
    
    Example: See atomistic_wall example
 
    Initialization variables:
    - configuration: Instance of the configuration class
    - ptype: The particle type for which the profile is calculated
    - bins: Number of bins to use in the profiles; more bins higher resolution. (default 100)
    - profdir: Profile spatial direction (default 2 - or z - direction)
    - veldir: The streaming velocity component used (default 0 - or x - direction)
    - verbose: If true print some information (default True)

    Output: 
     With the method read() you can retrieve the current data. By default a data file is printed with the same 
    """
    
    def __init__(self, configuration, ptype, bins=100, profdir=2, veldir=0, verbose=True):
        self.conf = configuration
        self.ptype = ptype
        self.bins = bins
        self.pdir = profdir
        self.vdir = veldir

        self.cpart = np.zeros(bins, dtype=np.int64)
        self.dens = np.zeros(bins, dtype=np.float64)
        self.temp = np.zeros(bins, dtype=np.float64)
        self.momc = np.zeros(bins, dtype=np.float64)

        self.volbin = configuration.get_volume()/bins
        self.widthbin = configuration.simbox.lengths[profdir]/bins
        self.hbox =  configuration.simbox.lengths[profdir]*0.5
        self.nsample = 0
        
        self.Tdir = [0, 1, 2]
        self.Tdir.remove(veldir)

        for n in range(configuration.N):
            if configuration.ptype[n] == self.ptype:
                self.mass = configuration['m'][n]
                break

        if verbose:
            print(f"Types {self.ptype}, no. bin {self.bins}, profile dir {self.pdir}, vel dir {self.vdir}")

    def update(self):
        
        __dosample__(self.cpart, self.temp, self.momc, 
                     self.conf['r'][:,self.pdir], self.conf['v'][:,self.vdir], self.conf['v'][:,self.Tdir],
                     self.mass, self.conf.ptype, self.ptype, self.conf.N, self.widthbin, self.hbox)

        self.nsample = self.nsample + 1

    def read(self, save=True): 

        density = np.zeros(self.bins)
        svel = np.zeros(self.bins)
        temp = np.zeros(self.bins)
        x = np.zeros(self.bins)

        fac = 1.0/(self.nsample*self.volbin)
        for n in range(self.bins):
            x[n] = (n+0.5)*self.widthbin
            if self.cpart[n] > 0:
                density[n] = self.mass*self.cpart[n]*fac                
                svel[n] = self.momc[n]*fac/density[n]
                temp[n] = self.temp[n]/(2.0*self.cpart[n])

        if save:
            fp =  open("HydrodynamicProfile.dat","w")
            for n in range(self.bins):
                fp.write("%f %f %f %f\n" % (x[n], density[n], svel[n], temp[n]))
                
            fp.close()
        
        return (x, density, svel, temp)
