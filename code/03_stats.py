"""Step 3. Group statistics across dyads.

Question 1: is real-partner synchrony higher than trial-shuffled synchrony?
Question 2: does that partner-specific synchrony differ between musical duets and
            constant-pitch sequences?
Tests are paired across dyads (one value per dyad), FDR-corrected across the
16 measure x band combinations. Electrode-level maps use max-T permutation tests
(mne.stats.permutation_t_test; hypyp.stats.statsCond for same-site pairs).
"""
import json
import numpy as np
import mne
from scipy import stats as ss
from hypyp import stats as hstats
from config import *

mne.set_log_level("ERROR")
pairs = [p for p in PAIRS if (SYNC / f"{p}_sync.npz").exists()]
bands = list(BANDS)
R, S, Rc, Sc, SG = ({m: [] for m in MODES} for _ in range(5))
n_ep = []
for p in pairs:
    z = np.load(SYNC / f"{p}_sync.npz", allow_pickle=True)
    cond = z["cond"]
    n_ep.append(len(cond))
    for m in MODES:
        r, s = z[f"real_{m}"], z[f"surr_{m}"]                # (f,e,c,c) ; (shuf,3,f,c,c)
        R[m].append(r.mean(1)); S[m].append(s[:, 0].mean(0))
        Rc[m].append(np.stack([r[:, cond == "duet"].mean(1), r[:, cond == "pitch"].mean(1)]))
        Sc[m].append(s[:, 1:].mean(0))
        SG[m].append(s[:, 0].mean((2, 3)))                   # (shuf, f) global value per shuffle
ch_names = list(z["ch_names"])
R, S, Rc, Sc, SG = ({m: np.array(v) for m, v in d.items()} for d in (R, S, Rc, Sc, SG))


def paired(d):
    t, p = ss.ttest_1samp(d, 0)
    w = ss.wilcoxon(d).pvalue
    return dict(mean=float(d.mean()), t=float(t), p=float(p), p_wilcoxon=float(w),
                dz=float(d.mean() / d.std(ddof=1)), n_pos=int((d > 0).sum()))


res = dict(pairs=pairs, n_epochs=n_ep, bands=BANDS, ch_names=ch_names, real_vs_surrogate=[], duet_vs_pitch=[])
for m in MODES:
    for fi, b in enumerate(bands):
        r, s = R[m][:, fi].mean((1, 2)), S[m][:, fi].mean((1, 2))
        row = dict(measure=m, band=b, real=float(r.mean()), surrogate=float(s.mean()),
                   pct=float(100 * ((r - s) / s).mean()), **paired(r - s))
        row["per_dyad_real"], row["per_dyad_surr"] = r.tolist(), s.tolist()
        row["dyads_above_all_shuffles"] = int((r > SG[m][:, :, fi].max(1)).sum())
        res["real_vs_surrogate"].append(row)
        d = (Rc[m][:, 0, fi] - Sc[m][:, 0, fi]).mean((1, 2)) - (Rc[m][:, 1, fi] - Sc[m][:, 1, fi]).mean((1, 2))
        raw = Rc[m][:, 0, fi].mean((1, 2)) - Rc[m][:, 1, fi].mean((1, 2))
        res["duet_vs_pitch"].append(dict(measure=m, band=b, **paired(d), raw=paired(raw),
                                         per_dyad=d.tolist(), per_dyad_raw=raw.tolist(),
                                         duet=float(Rc[m][:, 0, fi].mean()), pitch=float(Rc[m][:, 1, fi].mean())))
for key in ("real_vs_surrogate", "duet_vs_pitch"):
    rej, q = mne.stats.fdr_correction([r["p"] for r in res[key]])
    for r, qq in zip(res[key], q):
        r["q_fdr"] = float(qq)

# electrode-level maps: real - surrogate, every A-electrode x B-electrode link
info = mne.read_epochs(EPO / f"{pairs[0]}_L-epo.fif").info
epo = mne.read_epochs(EPO / f"{pairs[0]}_L-epo.fif")
maps = {}
for m in MODES:
    for fi, b in enumerate(bands):
        d = R[m][:, fi] - S[m][:, fi]                        # (dyads, chA, chB)
        n = d.shape[1]
        T, p, _ = mne.stats.permutation_t_test(d.reshape(len(d), -1), 5000, tail=0, seed=SEED)
        homol = d[:, np.arange(n), np.arange(n)]             # same electrode on both heads
        sc = hstats.statsCond(homol[:, :, None], epo, 5000, 0.05)
        maps[f"{m}_{b}"] = dict(T=T.reshape(n, n), p=p.reshape(n, n), mean=d.mean(0),
                                homol_T=sc.T_obs, homol_p=sc.p_values, homol_mean=homol.mean(0))
        res.setdefault("electrode_level", []).append(dict(
            measure=m, band=b, n_links_p05_maxT=int((p < .05).sum()), min_p=float(p.min()),
            max_abs_T=float(np.abs(T).max()), homol_sites_p05=[ch_names[i] for i in np.where(sc.p_values < .05)[0]]))
np.savez_compressed(SYNC / "group_maps.npz", **{f"{k}__{kk}": vv for k, v in maps.items() for kk, vv in v.items()},
                    R_plv=R["plv"].mean(0), S_plv=S["plv"].mean(0))
json.dump(res, open(ROOT / "derivatives" / "results.json", "w"), indent=1)

print(f"{len(pairs)} dyads, epochs/dyad median {int(np.median(n_ep))} (min {min(n_ep)})")
print("REAL vs SHUFFLED"); 
for r in res["real_vs_surrogate"]:
    print(f"{r['measure']:14s}{r['band']:6s} real={r['real']:.4f} surr={r['surrogate']:.4f} {r['pct']:+5.2f}% t={r['t']:+5.2f} p={r['p']:.4f} q={r['q_fdr']:.3f} dz={r['dz']:+.2f} pos={r['n_pos']}/{len(pairs)} top={r['dyads_above_all_shuffles']}")
print("DUET vs PITCH (partner-specific | raw)")
for r in res["duet_vs_pitch"]:
    print(f"{r['measure']:14s}{r['band']:6s} d={r['mean']:+.4f} t={r['t']:+5.2f} p={r['p']:.4f} q={r['q_fdr']:.3f} | raw t={r['raw']['t']:+5.2f} p={r['raw']['p']:.4f}")
print("ELECTRODE LEVEL")
for r in res["electrode_level"]:
    print(f"{r['measure']:14s}{r['band']:6s} links p<.05(maxT)={r['n_links_p05_maxT']} min_p={r['min_p']:.3f} maxT={r['max_abs_T']:.2f} homol={r['homol_sites_p05']}")
