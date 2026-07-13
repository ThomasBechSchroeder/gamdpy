from typing import Callable

import numba
import numpy as np
from numba import cuda

from gamdpy import Configuration
from . import Interaction
from .make_fixed_interactions import make_fixed_interactions

class Planar(Interaction):
    r""" Planar interactions such as smooth walls, gravity, or an electric field.

    Consider a plane with the normal vector :math:`{\bf n}` going though the point :math:`{\bf p}`.
    For a given particle, let :math:`{\bf r}` be the distance to the nearest point in the plane.
    Then the planar force is

    .. math::

        {\bf F} = s(r) {\bf n}

    Where :math:`s(r)=-u'(r)/r` is the force multiplier of a given potential function.

    Note: The planer interaction is considered an "external force",
    and will not contribute particles scalar energies, virials etc.

    Parameters
    ----------

    potential : Callable
        Potential function for planer interactions
        See :func:gamdpy.potential_functions.harmonic_repulsion for an example.

    params : list[list[float]]
        A list of parameters for each plane type.
        Each entry is a list of parameters for the potential function (see above).

    indices : list[list[int]]
        A list of lists, each containing the indices of a particle involved, and the planer interactions of relevance.

    normal_vectors : list[list[float]]
        A list of lists, each containing a normal vector for a given plane.

    points : list[list[float]]
        A list of lists, each containing a point on a given plane.

    Example
    -------

    Planar interactions can be used to add smooth walls and gravity to a simulation (be aware of periodic boundary conditions).

    >>> import gamdpy as gp
    >>>
    >>> # Replace the below with your own simulation
    >>> sim = gp.get_default_sim()
    >>> box_length = sim.configuration.simbox.get_lengths()[1]  # Box-length in y-direction
    >>> N = sim.configuration.N
    >>>
    >>> # Two smooth walls
    >>> wall_distance = box_length/2
    >>> walls = gp.interactions.Planar(
    ...     potential=gp.harmonic_repulsion,
    ...     params=[[100.0, 1.0], [100.0, 1.0]],
    ...     indices=[[n, 0] for n in range(N)] + [[n, 1] for n in range(N)],  # All particles feel both walls
    ...     normal_vectors=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    ...     points=[[0.0, -wall_distance/2, 0], [0.0, wall_distance/2, 0]]
    ... )
    >>>
    >>> # Gravity
    >>> mg = 0.0005
    >>> potential_gravity = gp.make_IPL_n(-1)
    >>> gravity = gp.interactions.Planar(
    ...     potential=potential_gravity,
    ...     params= [[mg, 10*wall_distance]],
    ...     indices= [[n, 0] for n in range(N)],   # All particles feel the gravity
    ...     normal_vectors= [[0,1,0], ],
    ...     points= [[0, -wall_distance/2.0, 0] ]  # Defining 0 for potential energy on the lower wall
    ... )
    >>> interactions = [..., walls, gravity]

    """

    def __init__(self,
                 potential: Callable,
                 params: list[list[float]],
                 indices: list[list[int]],
                 normal_vectors: list[list[float]],
                 points: list[list[float]]):

        # User set variables
        self.potential = potential
        self.params = params
        self.indices = indices  # list with "particle index", "planer interaction type"
        self.normal_vectors = normal_vectors
        self.points = points

        # Derived variables
        self.potential_njit = numba.njit(potential)
        self.num_types = len(self.params)

        # Set by get_kernal method
        self.d_indices = None
        self.d_params = None


    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool]) -> Callable:
        """ Get a kernel that implements calculation of the interaction """
        D = configuration.D
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
        dist_sq_function = numba.njit(configuration.simbox.get_dist_sq_function())
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        pot_func = self.potential_njit

        def calculator(vectors, scalars, ptype, sim_box, indices, values):
            particle = indices[0]
            interaction_type = indices[1]
            point = values[interaction_type][0:D]  # Point on plane
            normal_vector = values[interaction_type][D:2 * D]  # Normal vector defining plane
            potential_params = values[interaction_type][2 * D:]

            # Calculating displacement vector to plane
            dr = cuda.local.array(shape=D, dtype=numba.float32)
            dist_sq = dist_sq_dr_function(vectors[r_id][particle], point, sim_box, dr)
            dist = numba.float32(0.0)
            for k in range(D):
                dist += dr[k] * normal_vector[k]

            if particle==0 and interaction_type==1:
                #print(interaction_type)
                #print(point[0],point[1], point[2] )
                #print(normal_vector[0],normal_vector[1],normal_vector[2] )
                #print(dist_sq)
                #print(dist, values[interaction_type][-1])
                #print(potential_params[0], potential_params[1], potential_params[2], )
                pass

            if abs(dist) < values[interaction_type][-1]:  # Last index is the cut-off
                u, s, umm = pot_func(dist, potential_params)
                #print(particle, vectors[r_id][particle][1], interaction_type, dist, s, normal_vector[0],normal_vector[1],normal_vector[2])
                for k in range(D):
                    cuda.atomic.add(vectors, (f_id, particle, k), normal_vector[k] * dist * s)  # Force

            return

        return make_fixed_interactions(configuration, calculator, compute_plan, verbose=False)

    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple:
        """ Get a tuple with the parameters expected by the associated kernel """
        values = []
        for point, vector, potenetial_params in zip(self.points, self.normal_vectors, self.params):
            values.append(point + vector + potenetial_params)

        self.d_indices = cuda.to_device(self.indices)
        self.d_values = cuda.to_device(values)
        return self.d_indices, self.d_values
