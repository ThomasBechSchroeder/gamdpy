#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:49:08 2025

@author: nbailey
"""

import numpy as np
import numba
from numba import cuda
import math
from .simulationbox import SimulationBox


class LeesEdwards(SimulationBox):
    """ Simulation box class with LeesEdwards bondary conditions

    Example
    -------

    >>> import gamdpy as gp
    >>> import numpy as np
    >>> simbox = gp.LeesEdwards(D=3, lengths=np.array([3,4,5]), box_shift=1.0)

    """
    def __init__(self, D, lengths, box_shift=0., box_shift_image=0):
        if D < 2:
            raise ValueError("Cannot use LeesEdwards with dimension smaller than 2")
        self.D = D
        self.box_shift = box_shift
        self.box_shift_image = np.float32(box_shift_image)
        self.data_array = np.zeros(D+2,  dtype=np.float32)
        self.data_array[:D] = np.array(lengths, dtype=np.float32)
        self.data_array[D] = self.box_shift
        self.data_array[D+1] = self.box_shift_image
        self.len_sim_box_data = D+2

        return

    def get_name(self):
        return "LeesEdwards"

    def copy_to_device(self):
        self.d_data = cuda.to_device(self.data_array)


    def copy_to_host(self):
        D = self.D
        self.data_array =  self.d_data.copy_to_host()
        self.box_shift = self.data_array[D]
        self.boxshift_image = self.data_array[D+1]

    def get_volume_function(self):
        D = self.D
        def volume(sim_box):
            ''' Returns volume of the rectangular box '''
            vol = sim_box[0]
            for i in range(1,D):
                vol *= sim_box[i]
            return vol
        return volume

    def get_lengths(self):
        return self.data_array[:self.D].copy()

    def get_volume(self):
        #self.copy_to_host()
        return self.get_volume_function()(self.data_array[:self.D])

    def scale(self, scale_factor):
        self.data_array[:self.D] *= scale_factor


    def get_dist_sq_dr_function(self):
        """Generates function dist_sq_dr which computes displacement and distance for one neighbor """
        D = self.D

        def dist_sq_dr_function(ri, rj, sim_box, dr):  
            box_shift = sim_box[D]
            for k in range(D):
                dr[k] = ri[k] - rj[k]

            dist_sq = numba.float32(0.0)
            box_1 = sim_box[1]
            dr[0] += (-box_shift if numba.float32(2.0) * dr[1] > +box_1 else
                      (+box_shift if numba.float32(2.0) * dr[1] < -box_1 else
                        numba.float32(0.0)))

            for k in range(D):
                box_k = sim_box[k]
                dr[k] += (-box_k if numba.float32(2.0) * dr[k] > +box_k else
                          (+box_k if numba.float32(2.0) * dr[k] < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr[k] * dr[k]
            return dist_sq

        return dist_sq_dr_function

    def get_dist_sq_function(self):

        D = self.D
        def dist_sq_function(ri, rj, sim_box):  
            box_shift = sim_box[D]
            dist_sq = numba.float32(0.0)

            # first shift the x-component depending on whether the y-component is wrapped
            dr1 = ri[1] - rj[1]
            box_1 = sim_box[1]
            x_shift = (-box_shift if numba.float32(2.0) * dr1 > box_1 else
                      (+box_shift if numba.float32(2.0) * dr1 < -box_1 else
                        numba.float32(0.0)))
            # then wrap as usual for all components
            for k in range(D):
                dr_k = ri[k] - rj[k]
                if k == 0:
                    dr_k += x_shift
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr_k * dr_k
            return dist_sq
        return dist_sq_function

    def get_apply_PBC(self):
        D = self.D
        def apply_PBC(r, image, sim_box):

            # first shift the x-component depending on whether the y-component is outside the box
            box_shift, bs_image = sim_box[D], int(sim_box[D+1])
            box1_half = sim_box[1] * numba.float32(0.5)
            if r[1] > + box1_half:
                r[0] -= box_shift
                image[0] -= bs_image
            if r[1] < -box1_half:
                r[0] += box_shift
                image[0] += bs_image
            # then put everything back in the box as usual
            for k in range(D):
                if r[k] * numba.float32(2.0) > +sim_box[k]:
                    r[k] -= sim_box[k]
                    image[k] += 1
                if r[k] * numba.float32(2.0) < -sim_box[k]:
                    r[k] += sim_box[k]
                    image[k] -= 1
            return
        return apply_PBC

    def get_update_box_shift(self):
        D = self.D
        def update_box_shift(sim_box, shift):
            # carry out the addition in double precision
            sim_box[D] = numba.float32(sim_box[D] + numba.float64(shift))
            Lx = sim_box[0]
            Lx_half = Lx*numba.float32(0.5)
            if sim_box[D] > +Lx_half:
                sim_box[D] -= Lx
                sim_box[D+1] += 1
            if sim_box[D] < -Lx_half:
                sim_box[D] += Lx
                sim_box[D+1] -= 1
            return
        return update_box_shift

    def get_dist_moved_sq_function(self):
        D = self.D
        def dist_moved_sq_function(r_current, r_last, sim_box, sim_box_last):
            """ See Chattoraj PhD thesis for criterion for neighbor list checking under shear https://pastel.hal.science/pastel-00664392/"""
            zero = numba.float32(0.)
            half = numba.float32(0.5)
            one = numba.float32(1.0)
            box_shift = sim_box[D]
            dist_moved_sq = zero


            strain_change = sim_box[D] - sim_box_last[D] # change in box-shift
            strain_change += (sim_box[D+1] - sim_box_last[D+1]) * sim_box[0] # add contribution from box_shift_image
            strain_change /= sim_box[1] # convert to (xy) strain


            # we will shift the x-component when the y-component is 'wrapped'
            dr1 = r_current[1] - r_last[1]
            box_1 = sim_box[1]
            y_wrap = (one if dr1 > half*box_1 else
                      -one if dr1 < -half*box_1 else zero)

            x_shift = y_wrap * box_shift + (r_current[1] -
                                            y_wrap*box_1) * strain_change
            # see the expression in Chatoraj Ph.D. thesis. Adjusted here to
            # take into account BC wrapping (otherwise would use the images
            # ie unwrapped positions)

            for k in range(D):
                dr_k = r_current[k] - r_last[k]
                if k == 0:
                    dr_k -= x_shift
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))
                dist_moved_sq = dist_moved_sq + dr_k * dr_k


            return dist_moved_sq
        return dist_moved_sq_function

    def get_dist_moved_exceeds_limit_function(self):
        D = self.D

        def dist_moved_exceeds_limit_function(r_current, r_last, sim_box, sim_box_last, skin, cut):
            zero = numba.float32(0.)
            half = numba.float32(0.5)
            one = numba.float32(1.0)
            box_shift = sim_box[D]
            dist_moved_sq = zero


            strain_change = sim_box[D] - sim_box_last[D] # change in box-shift
            strain_change += (sim_box[D+1] - sim_box_last[D+1]) * sim_box[0] # add contribution from box_shift_image
            strain_change /= sim_box[1] # convert to (xy) strain

            # we will shift the x-component when the y-component is 'wrapped'
            dr1 = r_current[1] - r_last[1]
            box_1 = sim_box[1]
            y_wrap = (one if dr1 > half*box_1 else
                      -one if dr1 < -half*box_1 else zero)

            x_shift = y_wrap * box_shift + (r_current[1] -
                                            y_wrap*box_1) * strain_change
            # see the expression in Chatoraj Ph.D. thesis. Adjusted here to
            # take into account BC wrapping (otherwise would use the images
            # ie unwrapped positions)

            for k in range(D):
                dr_k = r_current[k] - r_last[k]
                if k == 0:
                    dr_k -= x_shift
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))
                dist_moved_sq = dist_moved_sq + dr_k * dr_k

            skin_corrected = skin - abs(strain_change)*cut
            if skin_corrected < zero:
                skin_corrected = zero

            return dist_moved_sq > skin_corrected*skin_corrected*numba.float32(0.25)

        return dist_moved_exceeds_limit_function

    def get_loop_x_addition(self):
        return 1

    def get_loop_x_shift_function(self):
        D = self.D
        def loop_x_shift_function(sim_box, cell_length_x):
            box_shift = sim_box[D]
            return -int(math.ceil(box_shift/cell_length_x))

        return loop_x_shift_function
