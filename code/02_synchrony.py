"""Step 2. Inter-brain synchrony with HyPyP.

For each dyad, HyPyP band-passes + Hilbert-transforms both participants
(analyses.compute_freq_bands) and computes PLV, coherence, circular correlation
and imaginary coherence between every electrode of person A and every electrode
of person B (analyses.compute_sync), one value per 2 s window.

Control: the two players hear the same tones and move at the same tempo, so some
"synchrony" is expected from shared input alone. We therefore re-pair person A's
window with person B's window from a *different trial of the same condition and
the same position in the trial* (N_SHUFFLE times). Real minus shuffled is what
is specific to playing together in that moment.
"""
import sys
import numpy as np
import mne
from hypyp import analyses
from config import *

mne.set_log_level("ERROR")
CROP = slice(int(PAD * SFREQ), int((PAD + WIN) * SFREQ))


def sync(cs, mode):
    """HyPyP connectivity per epoch -> inter-brain block (n_freq, n_epochs, n_ch, n_ch)."""
    n = cs.shape[2]
    con = analyses.compute_sync(cs, mode, epochs_average=False)
    return np.abs(con[:, :, :n, n:]).astype(np.float32)


def main(pair):
    rng = np.random.default_rng(SEED + int(pair[-2:]))
    eL = mne.read_epochs(EPO / f"{pair}_L-epo.fif")
    eR = mne.read_epochs(EPO / f"{pair}_R-epo.fif")
    md = eL.metadata.copy()
    gL, gR = eL.metadata.good.values, eR.metadata.good.values
    data = np.array([eL.get_data(), eR.get_data()])                    # (2, epochs, ch, times)
    cs = analyses.compute_freq_bands(data, SFREQ, BANDS, filter_length="auto",
                                     l_trans_bandwidth=1.0, h_trans_bandwidth=2.0)
    cs = cs[..., CROP]                                                 # drop padded edges
    good = gL & gR
    real = {m: sync(cs[:, good], m) for m in MODES}

    # surrogate pairings: same condition, same window index, different trial
    trial, win, cond = md.trial.values, md.win.values, md.cond.values
    iL = np.where(gL)[0]
    surr = {m: [] for m in MODES}
    for _ in range(N_SHUFFLE):
        partner = np.full(len(md), -1)
        for i in iL:
            cand = np.where(gR & (win == win[i]) & (cond == cond[i]) & (trial != trial[i]))[0]
            if len(cand):
                partner[i] = rng.choice(cand)
        keep = iL[partner[iL] >= 0]
        cs_s = np.array([cs[0, keep], cs[1, partner[keep]]])
        for m in MODES:
            s = sync(cs_s, m)                                          # (freq, epochs, ch, ch)
            c = cond[keep]
            surr[m].append(np.stack([s.mean(1), s[:, c == "duet"].mean(1), s[:, c == "pitch"].mean(1)]))
    out = dict(ch_names=np.array(eL.ch_names, dtype=str), cond=cond[good].astype(str), trial=trial[good], win=win[good])
    for m in MODES:
        out[f"real_{m}"] = real[m]                                     # (freq, epochs, ch, ch)
        out[f"surr_{m}"] = np.array(surr[m])                           # (shuffle, [all,duet,pitch], freq, ch, ch)
    np.savez_compressed(SYNC / f"{pair}_sync.npz", **out)
    r, s = real["plv"].mean((1, 2, 3)), np.array(surr["plv"])[:, 0].mean((0, 2, 3))
    print(pair, f"epochs={good.sum()}", "PLV real", np.round(r, 4), "surr", np.round(s, 4), flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or PAIRS):
        main(p)
