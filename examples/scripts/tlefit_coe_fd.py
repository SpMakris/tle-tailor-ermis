import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from tlefit_coe_fd import *

import numpy as np

line1 = '1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082'
line2 = '2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473'
satellite = EarthSatellite(line1, line2, 'ISS (ZARYA)', ts)

line1 = '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994'
line2 = '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033'
satellite = EarthSatellite(line1, line2, 'APRIZESAT 10', ts)

fit_method = test_tle_fit_normalized

iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = fit_method(satellite, lamda=1e-3, rms_epsilon=0.0001, debug=True)

plt.semilogy(range(len(sigmas)), sigmas)
plt.show()

plt.semilogy(bs)
plt.show()

# --- List TLE files ---
print(sorted(REPO_ROOT.glob('tles/*.txt')))

tle_filename = REPO_ROOT / 'tles' / 'TWOLINE.TXT'
tle_filename = REPO_ROOT / 'tles' / '22335.txt'

with open(tle_filename) as f:
    print(''.join(f.readlines()[:10]))

with open(tle_filename, 'r') as f:

    tle_lines = f.readlines()

    tles = []

    for cnt, ix in enumerate(range(0, len(tle_lines), 3)):
        #print(cnt, ix, tle_lines[ix])
        tles.append((tle_lines[ix].strip(), tle_lines[ix + 1].strip(), tle_lines[ix + 2].strip()))

if str(tle_filename).endswith('TWOLINE.TXT'):
    with open(tle_filename, 'r') as f:

        tle_lines = f.readlines()

        tles = []
        tle = []

        for line in tle_lines:
            if line.startswith('#'):
                continue

            tle.append(line.strip())

            if line.startswith('2 '):
                tles.append((tle[0][2:9], tle[0], tle[1]))
                tle = []

print(tles[:3])

# --- LM Test ---

limit = 100#9*1000000
failed_tles = []

print(f'{"TLE":24s} {"Iter":>5s} {"Cov (m)":>10s}   {"StdDev (m)":>10s}   {"Res @ Epoch (m)":>13s}   {"Res @ End (m)":>13s}')

for idx, tle in enumerate(tles):

    line1 = tle[1]
    line2 = tle[2]
    satellite = EarthSatellite(line1, line2, tle[0], ts)

    try:
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = fit_method(satellite, fit_span=1, max_iter=25, lamda=1e-3, rms_epsilon=0.0001, debug=False)

        print(f'{tle[0]:24s} {iterations:5d} {np.sqrt(np.diag(P)[0]) * 1000:10.3f} {sigma * 1000:12.2e} {np.linalg.norm(b_new_epoch[0:3]) * 1000:17.2e} {np.linalg.norm(b[0:3]) * 1000:15.2e}')
    except:
        print(f'{tle[0]:24s} Failed')

    if idx == limit:
        break
