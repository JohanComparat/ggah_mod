# The energy closure: its AGN coupling and its options

The energy closure of `HotGasDPM.feedback_budget` turns the energy the supernovae and the active nuclei released in a halo into the fraction of its baryons it retains.
This page gives the default AGN coupling (since version 1.7.0) and the options of the closure: a single coupling, an AGN energy fitted to simulations, a gas shape tied to the energy released, and the bound gas cut at $R_\Delta$.

| Option | Switch | Parameters | Default |
|---|---|---|---|
| AGN energy | `EnergeticsParams.agn_energy` | `"eps_dpl"`: `log10_eps_agn_p`, `log10_m_eps_p`, `a_eps_lo`, `a_eps_hi`; `"soltan"`: `log10_eps_agn`; `"dpl"`: `log10_e_agn_p`, `log10_m_agn_p`, `a_agn_lo`, `a_agn_hi` | `"eps_dpl"` |
| Supernova coupling | `EnergeticsParams.eps_sn` | | 0.0286 |
| Gas shape | `HotGasDPM.closure_shape` | `log10_q_shape` | not applied |
| Bound gas | `HotGasDPM.bound_uk` | none | not applied |

## The AGN coupling as a double power law

By default the AGN energy is the energy the black holes released, $E_1 = \epsilon_{\rm r}[\langle M_{\rm BH}\rangle + M_{\rm BH}^{\rm sat}]c^2$, times a coupling that depends on halo mass (`eps_agn_dpl`):

$$
\epsilon_{\rm AGN}(M) = \frac{2\epsilon_{\rm p}}{(M/M_{\rm p})^{-a_{\rm lo}} + (M/M_{\rm p})^{-a_{\rm hi}}},
$$ (eq-eps-dpl)

with $M = M_{200{\rm m}}$ in $h^{-1}{\rm M}_\odot$, held fixed in redshift.

The parameters of `EnergeticsParams`, with their bounds:

```{include} _generated/params_energetics.md
```

The coupling is 0.0024, 0.042, 0.26 and 0.76 at $10^{12}$, $10^{13}$, $10^{14}$ and $10^{15}\,h^{-1}{\rm M}_\odot$.
The defaults are the joint fit of $\epsilon_{\rm SN}$ and $\epsilon_{\rm AGN}(M)$ to the energy that expels, through the closure's soft saturation and at $\eta_{\rm ej} = 2$, the baryons that the hot-gas profile within $R_\Delta$, the stars and the neutral gas do not hold, from $10^{11.5}$ to $10^{15}\,h^{-1}{\rm M}_\odot$ at $z = 0$.
The fit is `energy_budget.py` of the technical paper; a power law in mass gives $\chi^2 = 54.7$ for 15 masses, this form 2.5.
Against the gas fractions within $R_{500{\rm c}}$ of Lovisari et al. (2015, 2020) and Mantz et al. (2016), the closure at the defaults gives $\chi^2 = 32.2$ for 10 points, against 94.3 for version 1.6.0.

The closure of version 1.6.0, one coupling $\epsilon_{\rm AGN} = 0.010$ with $\epsilon_{\rm SN} = 0.1$, is `EnergeticsParams(agn_energy="soltan", eps_sn=0.1)`.
In the default mode `f_retained_energy` refuses a `log10_eps_agn` that is passed, set or traced.

## The AGN energy as a law in halo mass

With `agn_energy = "dpl"`, the energy the nuclei released in a halo is

$$
E_{\rm AGN}(M) = \frac{2E_{\rm p}}{(M/M_{\rm p})^{-a_{\rm lo}} + (M/M_{\rm p})^{-a_{\rm hi}}},
$$ (eq-eagn-dpl)

in erg, with $M = M_{200{\rm m}}$ in $h^{-1}{\rm M}_\odot$ and $E_{\rm p}$ the energy at the break mass $M_{\rm p}$ (`e_agn_dpl`).
It replaces $\epsilon_{\rm AGN}E_1$, so the closure needs no black hole.
The simulations inject several times the energy the census requires, so at these defaults the closure expels most of the gas of groups.
The law is held fixed in mass at every redshift.

The defaults are the law fitted to the mean energy per halo of four CAMELS models (TNG, SIMBA, Astrid and EAGLE on SWIFT) at their fiducial astrophysics, at $z = 0.21$, from $10^{11}$ to $10^{14.6}\,h^{-1}{\rm M}_\odot$:

Its four parameters are in the table above, `log10_e_agn_p` to `a_agn_hi`.

The box holds the fit to each model and the amplitudes of its 1P boxes.
Each model is fitted to 0.01–0.14 dex rms, against 0.17–0.79 dex for the `"soltan"` mode at its best constant $\epsilon_{\rm AGN}$.

## The gas shape tied to the energy released

`closure_shape(field, params, agn_params, galaxies_params)` returns the gas parameters with the density shape (the concentration ratio and the three slopes) moved towards those of an NFW profile cut at $R_\Delta$:

$$
x' = x_{\rm DM} + \frac{q}{q + q_0}\,(x - x_{\rm DM}),
\qquad
q = \frac{E_{\rm AGN} + E_{\rm SN}}{f_{\rm b}E_{\rm bind}},
$$ (eq-closure-shape)

with $q_0 = 10^{\texttt{log10\_q\_shape}}$.
A halo that released little energy against the binding energy of its baryons holds its gas on its dark matter; one that released much keeps the sector's shape.
The default $\log_{10}q_0 = -2.5$ (box $-5$ to $1$) is provisional.
`f_retained_energy(..., return_ratio=True)` returns $q$ with the retained fraction.

## The bound gas

`bound_uk` is `mass_uk` with the profile cut at $R_\Delta$ (`r_max_over_rdelta = 1`).
The profile of the hot-gas sector is fitted to X-ray and SZ data and extends to $2.54\,R_\Delta$, so it already holds part of the displaced gas.
A matter field that adds the ejected shell (`u_ej`) on top of the full profile counts that gas twice; with `bound_uk` the displaced gas is carried by the shell alone.

On the census's baryon split at $z = 0$ and the 1.7.0 defaults, with the shell at $\eta_{\rm ej} = 2$, $S(k) = P_{\rm mm}/P_{\rm mm}^{\rm DMO}$ is

| Hot gas | $k = 1$ | $k = 2$ | $k = 10\,h\,{\rm Mpc}^{-1}$ |
|---|---|---|---|
| full profile (`mass_uk`) | 0.863 | 0.798 | 0.749 |
| bound (`bound_uk`) | 0.911 | 0.872 | 0.780 |

A matter field takes the bound gas by passing `bound_uk` as its `u_gas`; the gas tracers (X-ray, SZ) keep `mass_uk`.

## Related pages

- {doc}`layer3_galaxies`: the stellar masses that enter $E_{\rm SN}$.
- {doc}`layer3_hot_gas`: the gas profile and the closure it rescales.
- {doc}`api/sectors`: the API of `ggah_mod.sectors.energetics` and `ggah_mod.sectors.gas`.
