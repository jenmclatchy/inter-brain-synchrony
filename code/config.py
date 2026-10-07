"""Shared settings for the inter-brain synchrony pipeline (OpenNeuro ds007471)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIDS = ROOT / "data" / "ds007471"                               # events / behaviour sidecars
RAW = Path(os.environ.get("IBS_RAW", ROOT / "data" / "raw"))    # the .eeg binaries
EPO = ROOT / "derivatives" / "epochs"
SYNC = ROOT / "derivatives" / "sync"
FIG = ROOT / "figures"

# sub-05 has a different channel layout in its header, so it is left out.
PAIRS = [f"sub-{n:02d}" for n in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)]

# Channel order in every BrainVision header: 32 channels per person.
# Header labels the first 32 "_R" and the last 32 "_L".
CH32 = ["VEOG_b", "VEOG_a", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "Cz", "C4", "T8", "HEOG_l", "CP5", "CP1", "CP2", "CP6", "HEOG_r",
        "P7", "P3", "Pz", "P4", "P8", "ref_lm", "O1", "FCz", "O2", "CPz"]
N_CH_FILE = 64
SFREQ_RAW = 1000.0
RESOLUTION = 0.5e-6            # volts per int16 unit

SFREQ = 250.0                  # after resampling
L_FREQ, H_FREQ = 1.0, 45.0
WIN = 2.0                      # analysis window (s)
PAD = 1.0                      # padding each side for band-pass + Hilbert edges
N_WIN = 4                      # windows per trial -> first 8 s of each joint performance
REJECT_PTP = 150e-6            # peak-to-peak rejection threshold on the analysis window

BANDS = {"theta": [4, 7], "alpha": [8, 12], "beta": [13, 30], "gamma": [31, 45]}
MODES = ["plv", "coh", "ccorr", "imaginary_coh"]
N_SHUFFLE = 20
SEED = 7
