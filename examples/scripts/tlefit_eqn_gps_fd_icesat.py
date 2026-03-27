import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tlefit_equinoctial_eph_fd import *

import coord_skyfield
from coord_skyfield import ITRF2TEME

df = pd.read_feather(REPO_ROOT / 'ephemeris' / 'icesat.fth')

t = df.timestamp
ephemeris = [((row['x'], row['y'], row['z']), (row['xdot'], row['ydot'], row['zdot'])) for idx, row in df.iterrows()]

ephemeris_teme = ITRF2TEME(t, ephemeris)
ephemeris = ephemeris_teme
t = np.array([_t.to_pydatetime() for _t in t])

last_obs = 2500
obs_stride = 2
lamda = 1e-3 * 0 + 1 # Interesting.The smaller number works, but diverges. This is better
iterations, solve_sat, elements_coe, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = \
test_tle_fit_normalized_equinoctial(t, ephemeris, central_diff=True, last_obs=last_obs, obs_stride=obs_stride, lamda=lamda, rms_epsilon=0.0001, debug=True)

# Split the data into in-sample and out-of-sample

# Optionally thin the observations
t_is = tt = t[::obs_stride]
eph_is = teph = ephemeris[::obs_stride]

if last_obs:
    t_os = t_is[last_obs:]
    t_is = t_is[:last_obs]
    eph_os = eph_is[last_obs:]
    eph_is = eph_is[:last_obs]

jd, jdf = solve_sat.model.jdsatepoch, solve_sat.model.jdsatepochF
print(sat_epoch_datetime(solve_sat.model))

exporter.export_tle(solve_sat.model)

plt.semilogy(range(len(sigmas)), sigmas)
plt.show()

plt.figure(figsize=(15,3))
plt.semilogy(bs)
plt.show()

def solution_residuals(t, ephemeris, solve_sat):

    bs = []

    offset_idxs = range(len(t))

    for offset_idx in offset_idxs:

        # Obs - Nom
        jd, jdf = jday_datetime(t[offset_idx])
        res = np.array(ephemeris[offset_idx]) - np.array(solve_sat.model.sgp4(jd, jdf)[1:])

        b = np.concatenate((res[0], res[1]))

        #bs.append(np.linalg.norm(b.T @ W @ b))
        bs.append(np.linalg.norm(b.T @ b))

    return np.array(bs)

plt.plot(solution_residuals(t_is, eph_is, solve_sat) * 1000, c='g')
plt.plot(range(last_obs, len(tt)), solution_residuals(t_os, eph_os, solve_sat) * 1000, c='r')

plt.xlabel('Observations')
plt.ylabel('Residuals (m)')
plt.grid()
plt.show()

plt.semilogy(solution_residuals(t_is, eph_is, solve_sat) * 1000, c='g')
plt.semilogy(range(last_obs, len(tt)), solution_residuals(t_os, eph_os, solve_sat) * 1000, c='r')
plt.show()

# --- Find the distance to TLEs with the same launch and proximate epoch ---

import datetime as dt

import json

from sgp4.conveniences import sat_epoch_datetime, jday_datetime

from spacetrack import SpaceTrackClient
import spacetrack.operators as op

st = SpaceTrackClient(os.environ['SPACETRACK_USER'], os.environ['SPACETRACK_PWD'])

launch_objects = json.loads(st.gp(object_id=op.like('2003-002~~'), orderby='TLE_LINE1', format='json'))

tles = []

sat_ids = [sat['NORAD_CAT_ID'] for sat in launch_objects]

sat_gps = json.loads(st.gp_history(norad_cat_id=','.join(sat_ids),  epoch=op.inclusive_range((sat_epoch_datetime(solve_sat.model) - dt.timedelta(days=1)).date(), (sat_epoch_datetime(solve_sat.model) + dt.timedelta(days=+2)).date()), orderby='TLE_LINE1', format='json'))

def solution_residuals(t, solve_sat, comp_sat):

    bs = []

    offset_idxs = range(len(t))

    for offset_idx in offset_idxs:

        # Obs - Nom
        jd, jdf = jday_datetime(t[offset_idx])
        b = np.ravel(np.array(np.array(solve_sat.sgp4(jd, jdf)[1:] - np.array(comp_sat.sgp4(jd, jdf)[1:]))))[:3]

        bs.append(b.T @ b)

    return np.array(bs)

candidates = []

for sat in sat_gps:
    candidate_sat = EarthSatellite(sat['TLE_LINE1'], sat['TLE_LINE2'], sat['TLE_LINE0'], ts)

    res = solution_residuals(t_is, solve_sat.model, candidate_sat.model)
    res_epoch = np.sqrt(res[-1])
    res = np.sqrt(np.mean(res))

    candidates.append((sat["NORAD_CAT_ID"], sat["OBJECT_ID"], sat["EPOCH"], sat["TLE_LINE0"], sat["TLE_LINE1"], sat["TLE_LINE2"], res, res_epoch))

df_candidates = pd.DataFrame(candidates, columns=['norad_cat_id', 'object_id', 'epoch', 'tle_line0', 'tle_line1', 'tle_line2', 'residual', 'residual_epoch'])

print(df_candidates.sort_values('residual').head(20))

print(df_candidates[df_candidates['object_id'] == '2003-002A'].sort_values('residual'))

sat = sat_gps[6]
candidate_sat = EarthSatellite(sat['TLE_LINE1'], sat['TLE_LINE2'], sat['TLE_LINE0'], ts)
print(candidate_sat.model.sgp4(jd, jdf))

print((np.ravel(solve_sat.model.sgp4(jd, jdf)[1:]) - np.ravel(candidate_sat.model.sgp4(jd, jdf)[1:]))[:3])

print(solve_sat.model.sgp4_tsince(0))

print(candidate_sat.model.sgp4_tsince((sat_epoch_datetime(solve_sat.model) - sat_epoch_datetime(candidate_sat.model)).total_seconds() / 60))

print(sat_epoch_datetime(solve_sat.model), sat_epoch_datetime(candidate_sat.model))
print((sat_epoch_datetime(solve_sat.model) - sat_epoch_datetime(candidate_sat.model)).total_seconds())

print(eph_is[0], t_is[0])

print(candidate_sat.model.sgp4(*jday_datetime(t_is[0])))

print(np.linalg.norm(np.ravel(eph_is[0])[:3] - np.ravel(candidate_sat.model.sgp4(*jday_datetime(t_is[0]))[1:])[:3]))

# --- Calculate new TLE at same epoch as Reference / Truth TLE ---
#
# * Grab the the epoch from the reference (truth) TLE
# * Fit all ephemeris so epoch is in the sweet spot of the bathtub
#     * This is probably bullshit, since we wouldn't have the future ephemeris yet
# * Calculate the predicted state vector at the reference epoch using the fitted model
# * add the predicted state vector to our ephemeris
# * Re-fit up to the predicted state vector to get the mean elements for the reference epoch
# * Compare fitted TLE to reference TLE
#
# ### Notes
# * Probably better to interpolate ephemeris or use EKF to reference

ref_tle = candidate_sat

print(sat_epoch_datetime(ref_tle.model))

ajd, ajdf = jday_datetime(sat_epoch_datetime(ref_tle.model))

# We'll do the thinning outside so we can preserve the new epoch

if obs_stride:
    tt = t[::obs_stride]
    et = ephemeris[::obs_stride]

if last_obs:
    tt = tt[:last_obs]
    et = et[:last_obs]

aligned_epoch_obs = np.searchsorted(tt, sat_epoch_datetime(ref_tle.model))
# ii = np.searchsorted(t, sat_epoch_datetime(ref_tle.model))
tt = np.insert(t, aligned_epoch_obs, sat_epoch_datetime(ref_tle.model))
et = ephemeris[:aligned_epoch_obs] + [tuple(np.array(x) for x in ref_tle.model.sgp4(ajd, ajdf)[1:])] + ephemeris[aligned_epoch_obs:]

print(sat_epoch_datetime(ref_tle.model))

print(tt[aligned_epoch_obs])

iterations, aligned_solve_sat, elements_coe, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(tt, et, last_obs=None, obs_stride=None, epoch_obs=aligned_epoch_obs, lamda=1e-3, rms_epsilon=0.0001, debug=False)

aligned_solve_sat.model.intldesg = ref_tle.model.intldesg
# aligned_solve_sat.model.satnum = ref_tle.model.satnum # FIXME: assign this in satrec at creation
aligned_solve_sat.model.classification = ref_tle.model.classification
# aligned_solve_sat.model.ndot = ref_tle.model.ndot # Useless, but pretty
# aligned_solve_sat.model.nddot = ref_tle.model.nddot# Useless, but pretty

exporter.export_tle(aligned_solve_sat.model)

exporter.export_tle(ref_tle.model)

plt.plot(np.sqrt(solution_residuals(t, aligned_solve_sat.model, ref_tle.model)) * 1000)
plt.title("Solution vs. Candidate NORAD Object Residuals")
plt.xlabel("Observation number")
plt.ylabel("Position Residual (m)")
plt.show()
