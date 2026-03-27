# This works, but
# * There are still some dimension / normalization stuff that doesn't make sense,
#   but works and may work worse if "fixed"

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from tlefit_equinoctial_jax import *

line1 = '1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082'
line2 = '2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473'
satellite = EarthSatellite(line1, line2, 'ISS (ZARYA)', ts)

line1 = '1 40019U 14033K   21064.48089419  .00000027  00000-0  13123-4 0  9994'
line2 = '2 40019  97.7274 245.3630 0083155 314.3836  45.0579 14.67086574359033'
satellite = EarthSatellite(line1, line2, 'APRIZESAT 10', ts)

lamda = 1e-3
hermitian = True # True
dx_limit = False # False
coe_limit = True # True
lm_reg = False # False

iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(satellite, lamda=lamda, rms_epsilon=0.0001, debug=True, hermitian=hermitian, dx_limit=dx_limit, coe_limit=coe_limit, lm_reg=lm_reg)

plt.semilogy(range(len(sigmas)), sigmas)
plt.show()

plt.semilogy(bs)
plt.show()

plt.plot(range(len(lamdas)), lamdas)
plt.show()

legends = ['a', 'e', 'i', 'w', 'argp', 'm', 'b*']

_dxs = np.array(dxs).reshape(-1, 7)

fig, axs = plt.subplots(1, 7)
for x in range(7):
    axs[x].semilogy(_dxs[:, x])
    plt.legend(legends[x])
plt.show()

# --- Play with fit span ---

debug = False

fits = []
its = []
sigs = []

# FIXME: Try timing the function as well

for x in range(1, 5):
    iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(satellite, fit_span=x, lamda=1e-3, rms_epsilon=0.0001, debug=debug)
    if debug:
        print()
        print('#' * 80)
        print()
    fits.append(x)
    its.append(iterations)
    sigs.append(sigma)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.semilogy(fits, sigs)
ax2.plot(fits, its)

ax1.set_xlabel("Fit Span")
ax1.set_ylabel("Sigma (m)")
ax2.set_ylabel("Iterations")
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

print(len(tles))

if str(tle_filename).split('/')[-1] == 'TWOLINE.TXT' or str(tle_filename).split('\\')[-1] == 'TWOLINE.TXT':
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

limit = 100 #9*1000000
failed_tles = []
results = []

print(f'{"TLE":24s} {"Iter":>5s} {"Cov (m)":>10s}   {"StdDev (m)":>10s}   {"Res @ Epoch (m)":>13s}   {"Res @ End (m)":>13s}')

for idx, tle in enumerate(tles):

    line1 = tle[1]
    line2 = tle[2]
    satellite = EarthSatellite(line1, line2, tle[0], ts)

    try:
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(satellite, fit_span=1, max_iter=25, lamda=1e-3, rms_epsilon=0.0001, debug=False)

        results.append((idx, iterations, sigma, lamdas, b_new_epoch, b, P))

        print(f'{tle[0]:24s} {iterations:5d} {np.sqrt(np.diag(P)[0]) * 1000:10.3f} {sigma * 1000:12.2e} {np.linalg.norm(b_new_epoch[0:3]) * 1000:17.2e} {np.linalg.norm(b[0:3]) * 1000:15.2e}')
    except:
        print(f'{tle[0]:24s} Failed')
        failed_tles.append(idx)

    if idx == limit:
        break

import pandas as pd

idx, iterations, sigma, lamdas, b_new_epoch, b, P = results[10]

df = pd.DataFrame(results, columns=['idx', 'iter', 'rms', 'lamdas', 'b_epoch', 'b_end', 'cov'])
print(df)

print(failed_tles)

df['b_epoch_mag'] = df.b_epoch.apply(lambda x: np.linalg.norm(x[:3]))
df['b_end_mag'] = df.b_end.apply(lambda x: np.linalg.norm(x[:3]))
df['cov'] = df['cov'].apply(lambda x: np.sqrt(x[0,0]) * 1000)
df = df.drop(['b_epoch', 'b_end'], axis=1)

# df.to_feather('eqn_jax_results.fth')

df.iter.hist()
plt.show()

df.iter.plot(kind='hist', logy=True)
plt.show()

df.rms.hist()
plt.show()

df.rms.plot(kind='hist', logy=True)
plt.show()

df.b_epoch_mag.hist()
plt.show()

df.b_epoch_mag.plot(kind='hist', logy=True)
plt.show()

df.b_end_mag.hist()
plt.show()

df.b_end_mag.plot(kind='hist', logy=True)
plt.show()

# --- SVD fails and nothing fixes it unless we restart ---

tle = tles[failed_tles[0]]
line1 = tle[1]
line2 = tle[2]

print(tle[0])
print(tle[1])
print(tle[2])
print()

lamda = 1e-3

satellite = EarthSatellite(line1, line2, tle[0], ts)

while True:

    try:
        iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(satellite, fit_span=1, max_iter=25, lamda=lamda, rms_epsilon=0.0001, debug=False)
    except np.linalg.LinAlgError:
        lamda *= 10
        continue

    break

print(f'Converged with starting Lamda: {lamda}\n')

iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(satellite, fit_span=1, max_iter=25, lamda=lamda, rms_epsilon=0.0001, debug=True)

plt.plot([satellite.model.sgp4_tsince(_x)[1][:3] for _x in range(100000)])
plt.show()

lamda = 1e-3 * 0 + 1
hermitian = True
dx_limit = False
coe_limit = True
lm_reg = False

print(f'{"Lamda":7s} {"Iter":>5s} {"Cov (m)":>10s}   {"StdDev (m)":>10s}   {"Res @ Epoch (m)":>13s}   {"Res @ End (m)":>13s}')

for lamda in [1e-3, 1e-2, 1e-1, 1, 10]:
    iterations, sigma, sigmas, dxs, bs, lamdas, b_epoch, b_new_epoch, b, P, A = test_tle_fit_normalized_equinoctial(satellite, lamda=lamda, rms_epsilon=0.0001, debug=False, hermitian=hermitian, dx_limit=dx_limit, coe_limit=coe_limit, lm_reg=lm_reg)

    print(f'{lamda:3.3g} {iterations:5d} {np.sqrt(np.diag(P)[0]) * 1000:10.3f} {sigma * 1000:12.2e} {np.linalg.norm(b_new_epoch[0:3]) * 1000:17.2e} {np.linalg.norm(b[0:3]) * 1000:15.2e}')
