import numpy as np
from collections.abc import Iterable

def thermodynamics_NpT(
        N: int,
        dof: int,
        U: Iterable[float],
        W: Iterable[float],
        K: Iterable[float],
        Vol: Iterable[float],
        k_B=1.0,
        T_ext: float=None,
        p_ext: float=None,
        per_particle=True
) -> dict:
    r"""
    Calculate thermodynamic response functions from equilibrium fluctuations in an isotropic NpT simulation.

    The implementation analyses time series of thermodynamic observables from a constant :math:`NpT` (isothermal–isobaric) ensemble,
    specifically
    potential energy :math:`U`,
    configurational virial :math:`W= \left. \frac{\partial U\!\left(\lambda\mathbf{r}^N\right)}{\partial \lambda} \right|_{\lambda=1}`,
    kinetic energy :math:`K`, and volume :math:`V`, and computes thermodynamic response functions.
    The implementation assumes:

    - An isotropic :math:`NpT` ensemble at equilibrium.
    - If ``per_particle=True`` (default), the inputs ``U``, ``W``, ``K``, ``Vol`` are *per-particle*
      values and if ``per_particle=False``, the inputs are treated as values for the entire system. ``N`` is used in the conversion.
    - The external temperature, :math:`T`, defaults to :math:`\langle T_\mathrm{inst}\rangle` if not provided, where
      the instantaneous temperature is computed from equipartition,
      :math:`T_\mathrm{inst} = \frac{2K}{k_B\, N_d}` and :math:`\mathrm{N_d}` is the number of degrees of freedom (``dof``).
    - The external pressure, :math:`p`, defaults to :math:`\langle p_\mathrm{inst}\rangle` if not provided, where
      the instantaneous pressure follows from,
      :math:`p_\mathrm{inst} = \frac{Nk_B T_\mathrm{inst} + W}{V}`.

    The isobaric heat capacity (per particle), :math:`c_p=\frac{1}{N}\left(\frac{\partial H}{\partial T}\right)_p`,
    the isothermal compressibility, :math:`\kappa_T=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_T`, and
    the isobaric expansion coefficient, :math:`\alpha_p = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_p`
    are computed using fluctuation-disipation formulas of the NpT ensemble.
    If :math:`H = U + K + p V` is the instantaneous enthalpy, then

    .. math::

        c_p &= \frac{\mathrm{Var}(H)}{N\, k_B\, T^2}, \\
        \kappa_T &= \frac{\mathrm{Var}(V)}{\langle V\rangle\, k_B\, T}, \\
        \alpha_p &= \frac{\mathrm{Cov}(V,H)}{\langle V\rangle\, k_B\, T^2}.

    From these, other thermodynamic identities are computed (see Returns below).

    Parameters
    ----------
    N : int
        Number of particles, :math:`N`.
    dof : int
        Total number of degrees of freedom of the system, :math:`N_d`.
    U : Iterable[float]
        Time series of potential energy, :math:`U`.
    W : Iterable[float]
        Time series of configurational virial, :math:`W`.
    K : Iterable[float]
        Time series of kinetic energy, :math:`K`.
    Vol : Iterable[float]
        Time series of volume, :math:`V`.
    k_B : float, optional
        Boltzmann constant of the unit system (default 1.0), :math:`k_B`.
    T_ext : float, optional
        External (bath) temperature, :math:`T`. If ``None``, then :math:`\langle T_\mathrm{inst}\rangle` is used.
    p_ext : float, optional
        External (bath) pressure, :math:`p`. If ``None``, then :math:`\langle p_\mathrm{inst}\rangle` is used.
    per_particle : bool, optional
        If ``True`` (default), the ``U``, ``W``, ``K``, ``Vol`` inputs are interpreted as *per-particle* values.
        If ``False``, they are treated as extensive.

    Returns
    -------
    dict
        A dictionary with scalar properties and response functions reported as intensive.

        - ``density``: :math:`\rho = N \langle 1/V \rangle`.
        - ``specific_volume``: :math:`v = \langle V \rangle/N`.
        - ``jensen_bias_density_volume``: :math:`\rho v - 1`
        - ``potential_energy``: :math:`\langle U \rangle / N`.
        - ``kinetic_energy``: :math:`\langle K \rangle / N`.
        - ``internal_energy``: :math:`\langle E \rangle / N` where :math:`E=U+K`.
        - ``kinetic_temperature``: :math:`\langle T_\mathrm{inst} \rangle`, with :math:`T_\mathrm{inst} = 2K/k_B N_d`.
        - ``external_temperature``: Provided external :math:`T` (if given).
        - ``pressure``: :math:`\langle p_\mathrm{inst} \rangle`, with :math:`p_\mathrm{inst} = (N k_B T_\mathrm{inst} + W)/V`.
        - ``external_pressure``: Provided external :math:`p` (if given).
        - ``compressibility_factor``: :math:`Z = p/\rho\, k_B T`.
        - ``enthalpy``: :math:`\langle H \rangle / N`, with :math:`H = U + K + p V`.
        - ``isobaric_heat_capacity``: :math:`c_p = \mathrm{Var}(H)/N k_B T^2`.
        - ``isothermal_compressibility``: :math:`\kappa_T = \mathrm{Var}(V) / \langle V \rangle k_B T`.
        - ``isothermal_bulk_modulus``: :math:`K_T = 1/\kappa_T`.
        - ``isobaric_expansion_coefficient``: :math:`\alpha_p = \mathrm{Cov}(V,H) / \langle V \rangle k_B T^2`.
        - ``isochoric_heat_capacity``: :math:`c_V = \frac{1}{N}\left(\frac{\partial E}{\partial T}\right)_V =  c_p - \dfrac{T\, \alpha_p^2}{\rho\, \kappa_T}`.
        - ``isochoric_heat_capacity_excess``: :math:`c_V^{\mathrm{ex}} = \frac{1}{N}\left(\frac{\partial U}{\partial T}\right)_V = c_V - c_V^{\mathrm{id}}`, with :math:`c_V^{\mathrm{id}} = \frac{1}{N}\left(\frac{\partial K}{\partial T}\right)_V  = \dfrac{N_d}{2N} k_B`.
        - ``adiabatic_index``: :math:`\gamma = \dfrac{c_p}{c_V}`.
        - ``adiabatic_compressibility``: :math:`\kappa_S = -\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_S = \kappa_T \dfrac{c_V}{c_p}`.
        - ``adiabatic_bulk_modulus``: :math:`K_S = 1/\kappa_S`.
        - ``isochoric_pressure_coefficient``: :math:`\beta_v = \left(\dfrac{\partial p}{\partial T}\right)_V = \dfrac{\alpha_p}{\kappa_T}`.
        - ``isochoric_pressure_coefficient_excess``: :math:`\beta_v^{\mathrm{ex}} = \left(\dfrac{\partial W}{\partial T}\right)_V / V =\beta_v-\rho k_B`.
        - ``adiabatic_pressure_coefficient``: :math:`\beta_s = \left(\dfrac{\partial p}{\partial T}\right)_S = \dfrac{\rho\, c_p}{T\, \alpha_p}`.
        - ``adiabatic_expansion_coefficient``: :math:`\alpha_s = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_s = \alpha_p - \kappa_T\, \beta_s`.
        - ``thermodynamic_gruneisen_parameter``: :math:`\gamma_G = V\left(\frac{\partial p}{\partial E}\right)_V = \left(\frac{\partial \ln T}{\partial \ln \rho}\right)_S = \dfrac{\alpha_p}{\rho \, \kappa_T \, c_V}`.
        - ``configurational_adiabatic_scaling_exponent``: :math:`\gamma = \left(\dfrac{\partial \ln T}{\partial \ln \rho}\right)_{S_\textrm{ex}} = \beta_v^{\mathrm{ex}} / \rho c_V^{\mathrm{ex}}`.`
        - ``joule_thomson_coefficient``: :math:`\mu_{JT} = \left(\dfrac{\partial T}{\partial p}\right)_H = \dfrac{T\, \alpha_p - 1}{\rho\, c_p}`.

    """

    if not all(np.isscalar(x) for x in (N, dof, k_B)):
        raise TypeError("N, dof and k_B must be scalars")
    for name, value in (("Volumes, Vol", Vol), ("Potential energies, U", U), ("Virials, W", W), ("Kinetic energies, K", K)):
        if np.isscalar(value):
            raise TypeError(f"{name} must be array-like, not a scalar")
        try:
            np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} must be array-like") from exc

    # Convert to numpy arrays
    if type(U) is not np.ndarray:
        U = np.array(U)
    if type(W) is not np.ndarray:
        W = np.array(W)
    if type(K) is not np.ndarray:
        K = np.array(K)
    if type(Vol) is not np.ndarray:
        Vol = np.array(Vol)

    # Scale values to per particle if given as intensive
    if per_particle:
        U = U * N
        W = W * N
        K = K * N
        Vol = Vol * N

    # Dictionary that is returned by this function
    output = {}

    # Density
    V = Vol
    mV = float(np.mean(V))
    mV_inv = float(np.mean(1.0/V))  # Mean inverse volume
    rho = N*mV_inv
    output.update(dict(density=float(rho)))
    output.update(dict(specific_volume=float(mV/N)))

    # Relative volume/density discrepancy due to Jensen’s inequality, ref: https://en.wikipedia.org/wiki/Jensen%27s_inequality
    # <1/V>  >  1/<V>
    jensen_rho_v = float(rho*mV/N-1)
    output.update(dict(jensen_bias_density_volume=float(jensen_rho_v)))

    # Energies
    mU = float(np.mean(U))
    output.update(dict(potential_energy=float(mU / N)))
    mK = float(np.mean(K))
    output.update(dict(kinetic_energy=float(mK / N)))
    E = U + K
    mE = float(np.mean(E))
    output.update(dict(internal_energy=float(mE / N)))

    # Temperature
    T_inst = 2.0 * K / (k_B * dof)
    mT = float(np.mean(T_inst))
    if T_ext is None:
        T_ext = mT  # Use ensemble temperature to estimate external temperature (if not provided)
    else:
        output.update(dict(external_temperature=float(T_ext)))
    output.update(dict(kinetic_temperature=float(mT)))
    T = T_ext  # To simplify below formulas

    # Pressure
    P = (N * k_B * T_inst + W) / V  # Instantaneous pressure
    mP = float(np.mean(P))
    if p_ext is None:
        p_ext = mP  # Use ensemble pressure to estimate external pressure (if not provided)
    else:
        output.update(dict(external_pressure=float(p_ext)))
    output.update(dict(pressure=float(mP)))

    # Compressibility Factor
    Z = p_ext/(rho*k_B*T)
    output.update(dict(compressibility_factor=float(Z)))

    # Enthalpy
    H = U + K + p_ext*V
    mH = float(np.mean(H))
    output.update(dict(enthalpy=float( mH / N )))

    # Isobaric heat capacity,
    dHH = float(np.var(H, ddof=1))
    c_p = dHH / ( N * k_B * T ** 2)
    output.update(dict(isobaric_heat_capacity=float(c_p)))

    # Isothermal compressibility, κ_T = -(∂V/∂p)_T / V
    dVV = float(np.var(V, ddof=1))
    kappa_T = dVV / ( mV * k_B * T )
    output.update(dict(isothermal_compressibility=float(kappa_T)))
    output.update(dict(isothermal_bulk_modulus=float(1/kappa_T)))

    # Isobaric expansion coefficient, αₚ = (δV/δT)ₚ / V
    cov_VH = np.cov(V, H, ddof=1)[0, 1]
    alpha_p = cov_VH/(k_B * T**2 * mV)
    output.update(dict(isobaric_expansion_coefficient=float(alpha_p)))

    # Isochoric heat capacity,
    c_V = c_p - T * alpha_p**2 / ( rho * kappa_T )
    output.update(dict(isochoric_heat_capacity=float(c_V)))

    # Ideal gas heat capacity, c_V_ex = c_V - c_V_id
    c_V_id = 0.5 * k_B * dof / N
    c_V_ex = c_V - c_V_id
    output.update(dict(isochoric_heat_capacity_excess=float(c_V_ex)))

    # Adiabatic index,
    adiabatic_index = c_p/c_V
    output.update(dict(adiabatic_index=float(adiabatic_index)))

    # Adiabatic compressibility, κₛ = -(∂V/∂p)ₛ / V
    kappa_s = kappa_T*c_V/c_p
    output.update(dict(adiabatic_compressibility=float(kappa_s)))
    output.update(dict(adiabatic_bulk_modulus=float(1/kappa_s)))

    # Isochoric pressure coefficient: βᵥ = (∂P/∂T)ᵥ
    beta_v = alpha_p/kappa_T
    output.update(dict(isochoric_pressure_coefficient=float(beta_v)))

    # Excess Isochoric pressure coefficient: β_v^ex = (∂W/∂T)ᵥ/V = (∂p/∂T)_V - ρ k_B
    beta_v_ex = beta_v - rho*k_B
    output.update(dict(isochoric_pressure_coefficient_excess=float(beta_v_ex)))

    # Adiabatic pressure coefficient: βₛ = (∂P/∂T)ₛ
    beta_s = rho*c_p/(T*alpha_p)
    output.update(dict(adiabatic_pressure_coefficient=float(beta_s)))

    # Adiabatic expansion coefficient, αₛ = (δV/δT)ₛ/V
    alpha_s = alpha_p - kappa_T * beta_s
    output.update(dict(adiabatic_expansion_coefficient=float(alpha_s)))

    # Thermodynamic Grüneisen parameter (dimensionless)
    gamma_G = beta_v/(c_V*rho)
    output.update(dict(thermodynamic_gruneisen_parameter=float(gamma_G)))

    # Configurational adiabatic scaling exponent,
    gamma = beta_v_ex/(rho*c_V_ex)
    output.update(dict(configurational_adiabatic_scaling_exponent=float(gamma)))

    # Joule–Thomson coefficient, μ_JT = (δT/δp)_H
    mu_JT = (alpha_p*T-1.0)/(c_p*rho)
    output.update(dict(joule_thomson_coefficient=float(mu_JT)))

    return output


def thermodynamics_NVT(
        N: int,
        dof: int,
        V: float,
        U: Iterable[float],
        W: Iterable[float],
        K: Iterable[float],
        k_B=1.0,
        T_ext: float=None,
        per_particle=True
) -> dict:
    r""" Calculate thermodynamic response functions from equilibrium fluctuations in an isotropic NVT simulation.

    The implementation analyzes time series of thermodynamic observables from a constant :math:`NVT` (canonical) ensemble,
    specifically potential energy :math:`U`,
    configurational virial :math:`W= \left. \frac{\partial U\!\left(\lambda\mathbf{r}^N\right)}{\partial \lambda} \right|_{\lambda=1}`,
    kinetic energy :math:`K`, and computes thermodynamic response functions.
    The implementation assumes:

    - An isotropic :math:`NVT` ensemble at equilibrium.
    - If ``per_particle=True`` (default), the inputs ``V``, ``U``, ``W``, ``K``  are *per-particle*
      values and if ``per_particle=False``, the inputs are treated as values for the entire system. ``N`` is used in the conversion.
    - The external temperature (``T_ext``), :math:`T`, defaults to :math:`\langle T_\mathrm{inst}\rangle` if not provided, where
      the instantaneous temperature is computed from equipartition,
      :math:`T_\mathrm{inst} = \frac{2K}{k_B\, N_d}` and :math:`\mathrm{N_d}` is the number of degrees of freedom (``dof``).

    The isochoric heat capacity (per particle), :math:`c_v=\frac{1}{N}\left(\frac{\partial E}{\partial T}\right)_V`,
    and thermal pressure coefficient :math:`\beta_v = \left(\frac{\partial p}{\partial T}\right)_V` are computed as

    .. math::

        c_v &= \frac{\mathrm{Var}(E)}{N\, k_B\, T^2}, \\
        \beta_V &= \frac{\mathrm{Cov}(p,E)}{k_B T^2}.

    respectively, where the internal energy is :math:`E = U + K` and the instantaneous pressure is :math:`p = (N k_B T + W) / V`

    Parameters
    ----------
    N : int
        Number of particles, :math:`N`.
    dof : int
        Total number of degrees of freedom of the system, :math:`N_d`.
    V : float
        Volume of simulation box, :math:`V`.
    U : Iterable[float]
        Time series of potential energy, :math:`U`.
    W : Iterable[float]
        Time series of configurational virial, :math:`W`.
    K : Iterable[float]
        Time series of kinetic energy, :math:`K`.
    k_B : float, optional
        Boltzmann constant of the unit system (default 1.0), :math:`k_B`.
    T_ext : float, optional
        External (bath) temperature, :math:`T`. If ``None``, then :math:`\langle T_\mathrm{inst}\rangle` is used.
    per_particle : bool, optional
        If ``True`` (default), the ``V``, ``U``, ``W``, ``K`` inputs are interpreted as *per-particle* values.
        If ``False``, they are treated as extensive.

    Returns
    -------
    dict
        A dictionary with scalar properties and response functions reported as intensive.

        - ``density``: :math:`\rho = N /V`.
        - ``specific_volume``: :math:`v = V / N`.
        - ``potential_energy``: Average potential energy per particle, :math:`\langle U\rangle/N`.
        - ``kinetic_energy``: Average kinetic energy per particle, :math:`\langle K\rangle/N`.
        - ``internal_energy``: Average internal energy per particle, :math:`\langle E\rangle/N` where :math:`E = U + K`.
        - ``kinetic_temperature``: Instantaneous kinetic temperature, :math:`\langle T_\textrm{inst}\rangle/N` where :math:`T_\textrm{inst} = 2K/k_B\, N_d`.
        - ``external_temperature``: External bath temperature (if provided), :math:`T`.
        - ``pressure``: Average instantaneous pressure, :math:`\langle p\rangle`, where :math:`p = (N k_B T_\textrm{inst} + W) / V`.
        - ``isochoric_heat_capacity``: Isochoric heat capacity per particle, :math:`c_V = \frac{1}{N}\left(\frac{\partial E}{\partial T}\right)_V = \textrm{Var}(E)/(N k_B T^2)`.
        - ``isochoric_heat_capacity_excess``: Excess (configurational) heat capacity per particle, :math:`c_V^\textrm{ex} = \frac{1}{N}\left(\frac{\partial U}{\partial T}\right)_V = \textrm{Var}(U)/(N k_B T^2)`.
        - ``thermal_pressure_coefficient``: Thermal pressure coefficient :math:`\beta_V = \left(\frac{\partial p}{\partial T}\right)_V = \textrm{Cov}(p,E)/(k_B T^2)`.
        - ``thermal_pressure_coefficient_excess``: Excess thermal pressure coefficient :math:`\beta_V^\textrm{ex} = \frac{1}{V} \left(\frac{\partial W}{\partial T}\right)_V = \textrm{Cov}(W,U)/(V k_B T^2)`.
        - ``thermodynamic_gruneisen_parameter``: :math:`\gamma_G = V\left(\frac{\partial p}{\partial E}\right)_V = \left(\frac{\partial \ln T}{\partial \ln \rho}\right)_S = \dfrac{\beta_V}{\rho \, c_V}`.
        - ``configurational_adiabatic_scaling_exponent``: :math:`\gamma = \left(\dfrac{\partial \ln T}{\partial \ln \rho}\right)_{s_\textrm{ex}} = \textrm{Cov}(W,U)/\textrm{Var}(U)` where :math:`s_\textrm{ex}` is the excess entropy.
        - ``canonical_virial_energy_correlation``: Pearson correlation coefficient between :math:`W` and :math:`U`, :math:`R=\textrm{Cov}(W,U)/\sqrt{\textrm{Var}(W)\textrm{Var}(U)}`.

    """
    if not all(np.isscalar(x) for x in (N, dof , V, k_B)):
        raise TypeError("N, dof, V and k_B must be scalars")
    for name, value in (("Potential energies, U", U), ("Virials, W", W), ("Kinetic energies, K", K)):
        if np.isscalar(value):
            raise TypeError(f"{name} must be array-like, not a scalar")
        try:
            np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} must be array-like") from exc

    # Convert to numpy arrays
    if type(U) is not np.ndarray:
        U = np.array(U)
    if type(W) is not np.ndarray:
        W = np.array(W)
    if type(K) is not np.ndarray:
        K = np.array(K)

    # Scale values to per particle if given as intensive
    if per_particle:
        V = V * N
        U = U * N
        W = W * N
        K = K * N

    # Dictionary to store output
    output = {}

    rho = N / V
    output.update(dict(density=float(rho)))
    output.update(dict(specific_volume=float( V / N )))

    mU = np.mean(U)
    output.update(dict(potential_energy=float( mU / N )))

    mK = np.mean(K)
    output.update(dict(kinetic_energy=float( mK / N )))

    E = U + K
    mE = np.mean(E)
    output.update(dict(internal_energy=float( mE / N )))

    T_inst = 2.0 * K / k_B / dof  # Instantaneous kinetic temperature
    mT_kin = np.mean(T_inst)
    output.update(dict(kinetic_temperature=float(mT_kin)))

    # Fall back to sample estimate if external temperature is not provided
    T = T_ext if T_ext else np.mean(T_inst)
    if T_ext:
        output.update(dict(external_temperature=float(T)))

    p = rho * k_B * T_inst + W / V  # Instantaneous pressure
    output.update(dict(pressure=float(np.mean(p))))

    var_EE = np.var(E, ddof=1)
    c_V = var_EE / (k_B * T ** 2 * N)
    output.update(dict(isochoric_heat_capacity=float(c_V)))

    var_UU = np.var(U, ddof=1)
    c_V_ex = var_UU / (k_B * T ** 2 * N)
    output.update(dict(isochoric_heat_capacity_excess=float(c_V_ex)))

    cov_pE = np.cov(p, E, ddof=1)[0, 1]
    beta_V = cov_pE / k_B / T ** 2  # Thermal pressure coefficient: βᵥ = (∂p/∂T)ᵥ
    output.update(dict(thermal_pressure_coefficient=float(beta_V)))

    cov_WU = np.cov(W, U, ddof=1)[0, 1]
    beta_V_ex = cov_WU / k_B / T ** 2 / V
    output.update(dict(thermal_pressure_coefficient_excess=float(beta_V_ex)))

    gruneisen = beta_V / rho / c_V
    output.update(dict(thermodynamic_gruneisen_parameter=float(gruneisen)))


    gamma = cov_WU / var_UU
    output.update(dict(configurational_adiabatic_scaling_exponent=float(gamma)))

    var_WW = np.var(W, ddof=1)
    R_WU = cov_WU / (var_WW * var_UU)**0.5
    output.update(dict(canonical_virial_energy_correlation=float(R_WU)))

    return output
