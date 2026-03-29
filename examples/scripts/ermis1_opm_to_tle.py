import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import warnings

import numpy as np

# Load the GMAT Python API
sys.path.insert(0, r'C:\GMAT_R2025a\api')
from load_gmat import *

from skyfield.api import load

import pandas as pd

import coord_skyfield
from coord_skyfield import ITRF2TEME

from tlefit_equinoctial_eph_fd import *

ts = load.timescale()

# --- ERMIS1 OPM Input ---
# Epoch in UTC. r in km, v in km/s, in the rotating ITRF (ECEF) frame.
# Source: launcher OPM dated 2026-03-30.

epochs = ["30 Mar 2026 11:14:20.948"]

states = [([4773.943352, 4614.259857, -1862.329830], [2.506364, 0.326790, 7.257768])]

# --- GMAT propagation ---
# Propagate each OPM forward 3 days with a high-fidelity SP propagator
# to obtain an ephemeris for TLE fitting.

for idx, (t, state) in enumerate(zip(epochs, states)):
    print(idx)

    r, v = state[0], state[1]

    gmat.LoadScript(str(REPO_ROOT / 'gmat' / 'prelaunch_opm.script'))

    sat = gmat.GetObject("Sat")
    sat.SetField("Epoch", t)
    sat.SetField("X", r[0])
    sat.SetField("Y", r[1])
    sat.SetField("Z", r[2])
    sat.SetField("VX", v[0])
    sat.SetField("VY", v[1])
    sat.SetField("VZ", v[2])

    # ERMIS1: DragArea and DryMass from mission specification
    sat.SetField("DragArea", 0.111)  # m^2
    sat.SetField("DryMass", 8.5)    # kg

    eph = gmat.GetObject("EphemerisFile1")
    eph.SetField("Filename", str(OUTPUT_DIR / f'EphemerisFile_Sat{idx}.e'))

    gmat.RunScript()

# --- Read GMAT ephemeris files ---

for idx, t in enumerate(epochs):

    df = pd.read_fwf(OUTPUT_DIR / f'EphemerisFile_Sat{idx}.e', widths=(21, 24, 24, 24, 24, 24, 24), names=('time', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot'), skiprows=15, skipfooter=4)
    df['timestamp'] = pd.Timestamp(epochs[idx], tz='UTC') + pd.to_timedelta(df.time, unit='s')
    df.to_feather(OUTPUT_DIR / f'sat{idx}.fth')

# --- Fit TLE ---

for idx in range(len(states)):

    df = pd.read_feather(OUTPUT_DIR / f'sat{idx}.fth')

    # Convert state vectors from ECEF to TEME
    t = df.timestamp
    ephemeris = [((row['x'], row['y'], row['z']), (row['xdot'], row['ydot'], row['zdot'])) for idx, row in df.iterrows()]

    ephemeris_teme = ITRF2TEME(t, ephemeris)
    ephemeris = ephemeris_teme

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = np.array([_t.to_pydatetime() for _t in t])

    last_obs = 4320
    obs_stride = 1
    epoch_obs = 0
    lamda = 1e-3 * 0 + 1
    rms_epsilon = 0.002
    iterations, solve_sat, elements_coe, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
    test_tle_fit_normalized_equinoctial(t, ephemeris, last_obs=last_obs, obs_stride=obs_stride, epoch_obs=epoch_obs, lamda=lamda, rms_epsilon=rms_epsilon, debug=False)

    tt = t[::obs_stride]
    if last_obs:
        tt = tt[:last_obs]

    print(f'Epoch: {tt[epoch_obs]}\n')
    print('\n'.join(exporter.export_tle(solve_sat.model)))
    print('\n')
