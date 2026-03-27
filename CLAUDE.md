# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TLE Tailor generates Two-Line Element Sets (TLEs) for Earth-orbiting satellites through differential correction (non-linear least squares optimization). TLEs encode mean orbital elements for the SGP4 propagation model — there is no direct analytical conversion from state vectors to TLEs, so the approach is to sample a trajectory and optimize TLE elements to best fit that trajectory.

## Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

For GMAT integration (optional), follow `GMAT API SETUP.md`: download GMAT from SourceForge and add its Python API path to `sys.path` in the relevant scripts.

SpaceTrack access requires environment variables:
```bash
export SPACETRACK_USER=your_username
export SPACETRACK_PWD=your_password
```

## Running Scripts

Scripts are run directly — there is no build step:
```bash
python tlefit_equinoctial_fd.py
python tlefit_equinoctial_jax.py
python tlefit_coe_fd.py
python tlefit_coe_jax.py
python tlefit_equinoctial_eph_fd.py   # fits TLE to ephemeris (FD)
python tlefit_equinoctial_eph_jax.py  # fits TLE to ephemeris (JAX)
```

Notebooks can be run interactively in Jupyter. There are no automated tests.

## Architecture

### Core Workflows

1. **TLE → TLE** (proof-of-concept): propagates an existing TLE, then re-fits a new TLE to the sampled trajectory; validates algorithm correctness.
2. **Ephemeris → TLE** (primary use): fits a TLE to GPS/precision ephemeris data (ICESat and GPS satellite examples in `ephemeris/`).
3. **OPM → TLE** (via GMAT): propagates a launch provider's state vector through GMAT to create ephemeris, then fits a TLE.

### Two Jacobian Methods

- **Finite Difference (FD)** (`*_fd.py`): perturbs each element numerically; more general, works with any black-box propagator.
- **JAX autodiff** (`*_jax.py`): analytically computes the Jacobian via JIT compilation; faster for batch processing but has upfront JIT cost.

The `sgp4_jax/` directory contains a full port of the SGP4 algorithm to JAX (enabling `jax.jacfwd`/`jax.jacrev`). This is the key custom component — the standard `sgp4` library does not support autodiff.

### Two Element Parametrizations

- **Classical Orbital Elements (COE)**: semi-major axis, eccentricity, inclination, RAAN, argument of perigee, mean anomaly. Singular at circular/equatorial orbits.
- **Equinoctial Elements**: avoids singularities; preferred for production fits.

### Optimization

All fitting uses **Levenberg-Marquardt** regularized least squares with adaptive damping, element update limiting to prevent divergence, and configurable fit span and iteration limits.

### Module Roles

| File | Role |
|------|------|
| `common.py` | SGP4 satellite creation, COE↔Equinoctial conversions, residual calculations |
| `coarse_fit.py` | Initial guess generator for TLE fitting |
| `common_coe_fd.py` / `common_coe_jax.py` | Jacobian computation for COE-based fitting |
| `common_jax.py` | Shared JAX utilities for equinoctial elements |
| `coord_astropy.py` / `coord_skyfield.py` | ITRF↔TEME coordinate conversions (two library alternatives) |
| `sgp4_jax/propagation.py` | Full SGP4 propagation ported to JAX (the core custom implementation) |
| `ephemeris/` | Sample feather-format binary ephemeris files (ICESat, GPS) |
| `tles/` | TLE text data files |
| `gmat/` | GMAT script for pre-launch OPM propagation |

## Key References

- Vallado & Crawford 2008 AIAA paper: "SGP4 Orbit Determination" (full PDF in `docs/`)
- Vallado: *Fundamentals of Astrodynamics and Applications*
