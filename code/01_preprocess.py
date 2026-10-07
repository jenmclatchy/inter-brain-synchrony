"""Step 1. Read each dyad's raw BrainVision binary, split it into two people,
clean it with MNE-Python and cut it into joint-performance epochs.

Per person: linked-mastoid re-reference, 1-45 Hz FIR band-pass, resample to 250 Hz,
ICA with automatic removal of ocular components (EOG channels as reference).
Epochs: 2 s analysis windows (plus 1 s padding each side) tiling the first 8 s of
every joint trial; a window is flagged bad if either the window exceeds 150 uV
peak-to-peak on any scalp channel.
"""
import sys
import numpy as np
import pandas as pd
import mne
from config import *

mne.set_log_level("ERROR")


def joint_trials(pair):
    """Onset, duration and condition of each joint test trial from events.tsv."""
    ev = pd.read_csv(BIDS / pair / "eeg" / f"{pair}_task-jointaction_events.tsv", sep="\t")
    ev["c"] = ev.trial_type.str.replace(" ", "")
    s1 = ev.onset[ev.c == "S1"]                             # start of test section
    t_test = s1.iloc[0] if len(s1) else 0.0
    e = ev[ev.onset >= t_test].reset_index(drop=True)
    starts = e.index[e.c == "S105"].tolist() + [len(e)]
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        seg = e.iloc[a:b]
        left, right = seg.onset[seg.c == "S128"].values, seg.onset[seg.c == "S4"].values
        if len(left) < 10 or len(right) < 5:                # solo trial (only one player)
            continue
        if not ((seg.c == "S2").any() and (seg.c == "S3").any()):
            continue
        block = e.iloc[:a]
        block = block.c[block.c.isin(["S10", "S11"])]
        if not len(block):
            continue
        t0 = left[4] + np.median(np.diff(left[:5]))         # first beat after 5 metronome tones
        t1 = min(seg.onset[seg.c == "S2"].iloc[0], seg.onset[seg.c == "S3"].iloc[0])
        rows.append(dict(t0=t0, dur=t1 - t0, cond="duet" if block.iloc[-1] == "S10" else "pitch"))
    # if the S1 marker is missing, practice trials come first: keep the last 40 (the test trials)
    return pd.DataFrame(rows[-40:] if not len(s1) else rows).reset_index(drop=True)


def load_person(pair, who, tmin, tmax):
    """One participant's 32 channels as an MNE Raw object (volts)."""
    f = RAW / f"{pair}_task-jointaction_eeg.eeg"
    mm = np.memmap(f, dtype="<i2", mode="r").reshape(-1, N_CH_FILE)
    a, b = int(tmin * SFREQ_RAW), min(int(tmax * SFREQ_RAW), len(mm))
    off = 0 if who == "R" else 32
    x = mm[a:b, off:off + 32].T.astype(np.float64) * RESOLUTION
    x = x - 0.5 * x[CH32.index("ref_lm")]                   # left mastoid -> linked mastoids
    types = ["eog" if ("EOG" in c) else "misc" if c == "ref_lm" else "eeg" for c in CH32]
    raw = mne.io.RawArray(x, mne.create_info(CH32, SFREQ_RAW, types))
    raw.drop_channels(["ref_lm"])
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


def clean(raw):
    raw.filter(L_FREQ, H_FREQ, picks=["eeg", "eog"], fir_design="firwin")
    raw.resample(SFREQ)
    ica = mne.preprocessing.ICA(n_components=15, method="fastica", random_state=SEED, max_iter=500)
    ica.fit(raw, picks="eeg", reject=dict(eeg=400e-6), decim=2)
    bad = set()
    for ch in ("VEOG_a", "VEOG_b", "HEOG_l", "HEOG_r"):
        idx, _ = ica.find_bads_eog(raw, ch_name=ch, measure="correlation", threshold=0.5)
        bad.update(idx)
    ica.exclude = sorted(bad)[:4]
    ica.apply(raw)
    raw.pick("eeg")
    return raw, len(ica.exclude)


def main(pair):
    trials = joint_trials(pair)
    tmin, tmax = trials.t0.min() - 20, (trials.t0 + trials.dur).max() + 20
    meta, onsets = [], []
    for ti, tr in trials.iterrows():
        for k in range(N_WIN):
            if (k + 1) * WIN <= tr.dur:
                meta.append(dict(trial=ti, win=k, cond=tr.cond))
                onsets.append(tr.t0 + k * WIN - tmin)
    meta = pd.DataFrame(meta)
    events = np.c_[np.round(np.array(onsets) * SFREQ).astype(int), np.zeros(len(meta), int),
                   (meta.cond == "duet").astype(int) + 1]
    n_ica = {}
    for who in ("L", "R"):
        raw, n_ica[who] = clean(load_person(pair, who, tmin, tmax))
        epo = mne.Epochs(raw, events, dict(pitch=1, duet=2), tmin=-PAD, tmax=WIN + PAD - 1 / SFREQ,
                         baseline=None, preload=True, metadata=meta)
        assert len(epo) == len(meta)
        core = epo.copy().crop(0, WIN - 1 / SFREQ).get_data()
        epo.metadata["good"] = np.ptp(core, axis=2).max(axis=1) < REJECT_PTP
        epo.save(EPO / f"{pair}_{who}-epo.fif", overwrite=True)
        n_ica[who + "_good"] = int(epo.metadata.good.sum())
    print(pair, f"trials={len(trials)} windows={len(meta)}", n_ica, flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or PAIRS):
        main(p)
