import numpy as np
import numba
import math
from numba import cuda

from ..simulation.get_default_compute_plan import get_default_compute_plan

#############################################################
#### Radial Distribution Function  
#############################################################

class CalculatorRadialDistribution():
    """ Calculator class for the radial distribution function, g(r)

    Parameters
    ----------

    configuration : gamdpy.Configuration
        The configuration object for which the radial distribution function is calculated.

    bins : int
        The number of bins in the radial distribution function.

    compute_plan : dict
    
    ptype : optional
        If specified, array ptypes used to calculate g(r). If not specfified default is configuration.ptype

    Example
    -------

    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()
    >>> calc_rdf = gp.CalculatorRadialDistribution(sim.configuration, bins=1000)
    >>> for _ in sim.run_timeblocks():
    ...     calc_rdf.update()      # Current configuration to rdf
    >>> rdf_data = calc_rdf.read() # Read the rdf data as a dictionary
    >>> r = rdf_data['distances']  # Pair distances
    >>> rdf = rdf_data['rdf']      # Radial distribution function
    """

    def __init__(self, configuration, bins, compute_plan=None, ptype=None) -> None:
        self.configuration = configuration
        self.d_ptype = cuda.to_device(ptype) if ptype is not None else None
        self.bins = bins
        self.count = 0  # How many times have statistics been added to?

        self.compute_plan = compute_plan
        if self.compute_plan is None:
            self.compute_plan = get_default_compute_plan(configuration=configuration)

            # Allocate space for statistics
        self.rdf_list = []
        nptypes = int(np.max(ptype if ptype is not None else configuration.ptype)) + 1
        self.gr_bins = np.zeros((nptypes, nptypes, self.bins), dtype=np.float64)
        self.d_gr_bins = cuda.to_device(self.gr_bins)
        self.host_array_zeros = np.zeros(self.d_gr_bins.shape, dtype=self.d_gr_bins.dtype)

        # Make kernel for updating statistics
        self.update_kernel = self.make_updater_kernel(configuration, self.compute_plan)

    def make_updater_kernel(self, configuration, compute_plan, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        num_blocks = (num_part - 1) // pb + 1

        # Unpack indices for scalars to be compiled into kernel  
        r_id, = [configuration.vectors.indices[key] for key in ['r', ]]

        # Prepare user-specified functions for inclusion in kernel(s)
        ptype_function = numba.njit(configuration.ptype_function)
        #params_function = numba.njit(pair_potential.params_function)
        dist_sq_function = numba.njit(configuration.simbox.get_dist_sq_function())

        def rdf_calculator_full(vectors, sim_box, ptype, d_gr_bins):
            """ Calculate g(r) fresh
            Kernel configuration: [num_blocks, (pb, tp)]
        """

            bins = d_gr_bins.shape[2]  # reading number of bins from size of the device array
            min_box_dim = min(sim_box[:D])
            bin_width = (min_box_dim / 2) / bins  # TODO: Chose more directly!

            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block * pb + local_id
            my_t = cuda.threadIdx.y

            if global_id < num_part:
                ptype1 = ptype[global_id]
                for i in range(0, num_part, pb * tp):
                    for j in range(pb):
                        other_global_id = j + i + my_t * pb
                        ptype2 = ptype[other_global_id]
                        if other_global_id != global_id and other_global_id < num_part:
                            dist_sq = dist_sq_function(vectors[r_id][other_global_id], vectors[r_id][global_id],
                                                       sim_box)

                            # Calculate g(r)
                            if dist_sq < (min_box_dim / 2) ** 2:
                                dist = math.sqrt(dist_sq)
                                if dist < min_box_dim / 2:
                                    bin_index = int(dist / bin_width)
                                    cuda.atomic.add(d_gr_bins, (ptype1, ptype2, bin_index), 1)

            return

        return cuda.jit(device=0)(rdf_calculator_full)[num_blocks, (pb, tp)]

    def update(self):
        """ Update the radial distribution function with the current configuration. """
        self.count += 1
        self.update_kernel(self.configuration.d_vectors,
                           self.configuration.simbox.d_data,
                           self.d_ptype if self.d_ptype is not None else self.configuration.d_ptype,
                           self.d_gr_bins)
        self.rdf_list.append(self.d_gr_bins.copy_to_host())
        self.d_gr_bins = cuda.to_device(self.host_array_zeros)

    def read(self):
        """ Read the radial distribution function

        Returns
        -------

        dict
            A dictionary containing the distances and the radial distribution function.
        """
        bins = self.rdf_list[0].shape[2]
        min_box_dim = np.min(self.configuration.simbox.get_lengths())
        bin_width = (min_box_dim / 2) / bins
        rdf_ptype = np.array(self.rdf_list)

        # Normalize the g(r) lengths # Compute in setup and normalize om the fly 
        rho = self.configuration.N / self.configuration.simbox.get_volume()
        for i in range(bins):  # Normalize one bin/distance at a time
            r_outer = (i + 1) * bin_width
            r_inner = i * bin_width
            D = self.configuration.D
            if D % 2 == 0:
                n = D//2
                unit_hypersphere_volume = math.pi**n/math.factorial(n)
            else:
                n = (D - 1) // 2
                unit_hypersphere_volume = ( 2**D * math.pi**n * math.factorial(n) ) / math.factorial(D)
            shell_volume = unit_hypersphere_volume * (r_outer**D - r_inner**D)
            expected_num = rho * shell_volume
            rdf_ptype[:, :, :, i] /= (expected_num * self.configuration.N)
        rdf = rdf_ptype.sum(axis=1).sum(axis=1)
        ptype = self.d_ptype.copy_to_host() if self.d_ptype is not None else self.configuration.d_ptype.copy_to_host()
        for j in range(rdf_ptype.shape[1]):
            n_j = np.sum(ptype == j) / len(ptype)
            rdf_ptype[:, j, :, :] /= n_j
        for k in range(rdf_ptype.shape[2]):
            n_k = np.sum(ptype == k) / len(ptype)
            rdf_ptype[:, :, k, :] /= n_k
        distances = np.arange(0, bins) * bin_width
        return {'distances': distances, 'rdf': rdf, 'rdf_ptype': rdf_ptype, "ptype": ptype}

    def save_average(self, output_filename="rdf.dat", save_ptype=False) -> None:
        """ Save the average radial distribution function to a file

        Parameters
        ----------

        output_filename : str
            The name of the file to which the radial distribution function is saved.

        save_ptype : bool
            Save the type of each particle in a file name ptype_* (default False)

        """

        rdf_dict = self.read()
        rdf_total = np.mean(rdf_dict['rdf'], axis=0)
        rdf_ij = []
        header_ij = " "
        for i in range(rdf_dict["rdf_ptype"].shape[1]):
            for j in range(rdf_dict["rdf_ptype"].shape[2]):
                rdf_ij.append(np.mean(rdf_dict['rdf_ptype'][:, i, j, :], axis=0))
                header_ij += f"g[{i}-{j}](r) "
        np.savetxt(output_filename, np.array([rdf_dict['distances'], rdf_total, *rdf_ij]).T, header=f"r g(r) {header_ij} ptype")
        np.savetxt(f"ptype_{output_filename}", np.array(rdf_dict["ptype"]).T, header="ptype")

