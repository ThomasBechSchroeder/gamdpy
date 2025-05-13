import itertools

import numpy as np

import gamdpy as gp
import numba


class CalculatorStructureFactor:
    """ Calculator class for the static structure factor, S(q).
    The calculation is done for several :math:`{\\bf q}` vectors given by

    .. math::

        {\\bf q} = (2\\pi n_x/L_x, 2\\pi n_y/L_y, ...)

    where :math:`n=(n_x, n_y, ...)` is a D-dimensional vector of integers and
    :math:`L_x`, :math:`L_y`, ... are the box lengths in the :math:`x`, :math:`y`, ... directions, respectively.
    Note that box length are assumed to be constant during the simulation (as in a NVT simulation).
    The collective density :math:`\\rho_{\\bf q}` is calculated as

    .. math::

        \\rho_{\\bf q} = \\frac{1}{\\sqrt{N}} \\sum_{n} f_n \\exp(-i {\\bf q}\\cdot {\\bf r}_n)

    where :math:`x_n` is the position of particle :math:`n`, and :math:`f_n` is the atomic form factor for that particle
    (one by default). The normalization constant is

    .. math::

        N = \\sum_{n} f_n.

    From this, the static structure factor is defined as

    .. math::

        S({\\bf q}) = |\\rho_{\\bf q}|^2.

    The method :meth:`~gamdpy.calculators.CalculatorStructureFactor.update`
    updates the structure factor with the current configuration.
    The method :meth:`~gamdpy.calculators.CalculatorStructureFactor.read` returns the structure factor for the q vectors in the q_direction.

    Parameters
    ----------

    configuration : gamdpy.Configuration
        The configuration object to calculate the structure factor for.

    q_max : float or None
        The maximum value of the q vectors.

    n_vectors : numpy.ndarray or None
        n-vectors defining q-vectors.
        The shape of n_vectors, if specified, must be (N, D)
        where N is the number of q vectors and D is the number of dimensions.
        If None, then use generate_q_vectors method.

    atomic_form_factors : numpy.ndarray or None
        The atomic form factors, :math:`f_n`. If None (default), then the atomic form factors are set to 1.
        Can be given as an array of floats, one for each atom.

    backend : str
        The backend to use for the calculation. Either 'CPU multi core' or 'CPU single core'.

    See also
    --------

    :class:`~gamdpy.CalculatorRadialDistribution`

    """

    BACKENDS = ['CPU multi core', 'CPU single core', 'GPU']

    def __init__(self, 
                 configuration: gp.Configuration, 
                 n_vectors: np.ndarray = None,
                 atomic_form_factors: np.ndarray = None,
                 backend='CPU multi core') -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f'Unknown backend, {backend}. The known backends are {self.BACKENDS}.')
        self.update_count = 0
        self.configuration = configuration
        self.L = self.configuration.simbox.lengths.copy()  # Copy box lengths so that it is not changed

        if n_vectors is not None:
            # n_vectors = [[0, 0, 1], [0, 0, 2], ..., [0, 1, 0], [0, 1, 1] ..., [18, 18, 18], ...]
            self.n_vectors = np.array(n_vectors)
            dimension_of_space = self.configuration.D
            if self.n_vectors.shape[1] != dimension_of_space:
                raise ValueError('n_vectors must have the same number of columns as the number of dimensions.')
            self.q_vectors = np.array(2 * np.pi * self.n_vectors / self.L, dtype=np.float32)
            self.q_lengths = np.linalg.norm(self.q_vectors, axis=1)
            self.sum_S_q = np.zeros_like(self.q_lengths)

        self.atomic_form_factors = atomic_form_factors
        if atomic_form_factors is None:
            number_of_atoms = self.configuration.N
            self.atomic_form_factors = np.ones(number_of_atoms, dtype=np.float32)

        # List for storing data
        self.list_of_rho_q = []
        self.list_of_rho_S_q = []

        self.wallclock_times = []

        # 3 first letters is CPU or GPU
        if backend[:3] == 'CPU':
            self._compute_rho_q = self._generate_compute_rho_q(backend)
        if backend == 'GPU':
            self.nthreads = 64
            self.update_kernel = self._make_update_kernel()
            self._compute_rho_q = self._compute_rho_q_gpu

    def generate_q_vectors(self, q_max:float):
        """ Generate q-vectors inside a sphere of radius q_max """
        dimension_of_space = self.configuration.D
        if q_max<0.0:
            raise ValueError(f'{q_max=} must be positive')
        n_max = int(np.ceil(q_max * max(self.L) / (2 * np.pi)))
        n_vectors = np.array(list(itertools.product(range(n_max), repeat=dimension_of_space)), dtype=int)
        n_vectors = n_vectors[1:]  # Remove the first vector [0, 0, 0]
        self.q_vectors = np.array(2 * np.pi * n_vectors / self.L, dtype=np.float32)

        # Remove q_vectors where the length is greater than q_max
        selection = np.linalg.norm(self.q_vectors, axis=1) < q_max
        self.q_vectors = self.q_vectors[selection]
        self.n_vectors = n_vectors[selection]
        self.q_lengths = np.linalg.norm(self.q_vectors, axis=1)
        self.sum_S_q = np.zeros_like(self.q_lengths)

    def _make_update_kernel(self):
        import math
        from numba import cuda
        from numba import float32 as flt

        def kernel(r_vectors, q_vectors, form_factors, rho_q_real, rho_q_imag):
            num = q_vectors.shape[0]  # Number of q vectors
            tid = cuda.grid(1)

            if tid < num:
                this_q = q_vectors[tid]
                real, imag = flt(0.0), flt(0.0)
                N = flt(0.0)
                for n in range(r_vectors.shape[0]):
                    pos = r_vectors[n]
                    dot = flt(0.0)
                    for d in range(r_vectors.shape[1]):
                        dot += pos[d] * this_q[d]
                    real += form_factors[n]*math.cos(dot)
                    imag += form_factors[n]*math.sin(dot)
                    N += form_factors[n]
                NN = flt(1.0)/math.sqrt(N)
                rho_q_real[tid] =  NN*real
                rho_q_imag[tid] =  NN*imag
            return
        return cuda.jit(device=0)(kernel)

    def _compute_rho_q_gpu(self, r_vectors, q_vectors, form_factors):
        from numba import cuda

        # Copy data to device
        d_r_vectors = cuda.to_device(r_vectors) #, q_vectors, form_factors, rho_q_real, rho_q_imag
        d_q_vectors = cuda.to_device(q_vectors)
        d_form_factors = cuda.to_device(form_factors)
        rho_q_real = np.zeros(q_vectors.shape[0], dtype=np.float32)
        d_rho_q_real = cuda.to_device(rho_q_real)
        d_rho_q_imag = cuda.device_array_like(d_rho_q_real)

        # Compute using GPU kernel
        start = numba.cuda.event()
        end = numba.cuda.event()
        num = self.q_vectors.shape[0]
        nblocks = (num // self.nthreads) + 1
        start.record()
        self.update_kernel[nblocks, self.nthreads](d_r_vectors, d_q_vectors, d_form_factors, d_rho_q_real, d_rho_q_imag)
        end.record()
        end.synchronize()
        self.wallclock_times.append(start.elapsed_time(end))
        rho_q_real = d_rho_q_real.copy_to_host()
        rho_q_imag = d_rho_q_imag.copy_to_host()

        return rho_q_real + 1j*rho_q_imag


    @staticmethod
    def _generate_compute_rho_q(backend):
        if backend == 'CPU multi core':  # May raise "ImportError: scipy 0.16+ is required for linear algebra" if scipy is not installed
            def func(r_vec: np.ndarray, q_vec: np.ndarray, form_factors: np.ndarray) -> np.ndarray:
                N = np.sum(form_factors)
                number_of_q_vectors = q_vec.shape[0]
                rho_q = np.zeros(number_of_q_vectors, dtype=np.complex64)
                for i in numba.prange(number_of_q_vectors):
                    r_dot_q = np.dot(r_vec, q_vec[i])
                    rho_q[i] = np.sum(form_factors*np.exp(1j * r_dot_q))*N**-0.5
                return rho_q
            return numba.njit(parallel=True)(func)
        elif backend == 'CPU single core':
            def func(r_vec: np.ndarray, q_vec: np.ndarray, form_factors: np.ndarray) -> np.ndarray:
                N = np.sum(form_factors)
                r_dot_q = np.dot(r_vec, q_vec.T)
                rho_q = np.sum(form_factors[:, np.newaxis]*np.exp(1j * r_dot_q), axis=0)*N**-0.5
                return rho_q
            return numba.njit(func)

    def update(self) -> None:
        """ Update the structure factor with the current configuration. """
        if not np.allclose(self.L, self.configuration.simbox.lengths):
            raise ValueError('Box length has changed. Recreate the S(q) object.')
        this_rho_q = self._compute_rho_q(self.configuration['r'], self.q_vectors, self.atomic_form_factors)
        self.list_of_rho_q.append(this_rho_q)
        self.list_of_rho_S_q.append(np.abs(this_rho_q)**2)
        self.sum_S_q += np.abs(this_rho_q) ** 2
        self.update_count += 1

    #def read(self, bins: int | None) -> dict:
    def read(self, bins) -> dict:
        """ Return the structure factor S(q) for the q vectors in the q_direction.

        Parameters
        ----------

        bins : int | None
            If bins is an integer, the data is binned (ready to be plotted).
            If bins is None, the raw S(q) data is returned.

        Returns
        -------

        dict
            A dictionary containing the q vectors, the q lengths, the structure factor S(q),
            the collective density rho_q, and the number of q vectors in each bin. Output depends on the value of
            the bins parameter.
        """
        if isinstance(bins, int):
            q_bins = np.linspace(0, np.max(self.q_lengths), bins+1)
            q_binned = np.zeros(bins, dtype=np.float32)
            S_q_binned = np.zeros(bins, dtype=np.float32)
            q_vectors_in_bin = np.zeros(bins, dtype=int)
            for i in range(0, bins):
                mask = (q_bins[i] <= self.q_lengths) & (self.q_lengths < q_bins[i+1])
                if np.sum(mask) == 0:  # No q vectors in this bin
                    continue
                q_binned[i] = np.mean(self.q_lengths[mask])
                # Use self.list_of_rho_S_q
                S_q_binned[i] = np.mean(self.sum_S_q[mask] / self.update_count)
                q_vectors_in_bin[i] = np.sum(mask)
            # Remove bins with no q vectors
            mask = q_vectors_in_bin > 0
            q_binned = q_binned[mask]
            S_q_binned = S_q_binned[mask]
            q_vectors_in_bin = q_vectors_in_bin[mask]
            return {
                '|q|': q_binned,
                'S(|q|)': S_q_binned,
                'q_vectors_in_bin': q_vectors_in_bin
            }
        elif bins is None:
            # Return (un-binned) raw data
            return {
                'q': self.q_vectors,
                '|q|': self.q_lengths,
                'S(q)': self.sum_S_q/self.update_count,
                'rho_q': np.array(self.list_of_rho_q),
                'n_vectors': self.n_vectors,
                'atomic_form_factors': self.atomic_form_factors
            }
        else:
            raise ValueError('bins must be an integer.')

    #def save_average(self, bins: int=None, output_filename: str="sq.dat") -> None:
    def save_average(self, bins=None, output_filename="sq.dat"):
        """ Save average structure factors to a file. """
        if bins is None: bins=100
        sq_dict = self.read(bins)
        np.savetxt(output_filename, np.c_[sq_dict['|q|'], sq_dict['S(|q|)']], header="|q| S(|q|)")

