"""Step 0. Fetch the raw .eeg binaries for the analysed dyads from OpenNeuro's public bucket
(ds007471, CC0). About 4.2 GB. The small sidecar files are already in data/ds007471."""
import urllib.request
from config import PAIRS, RAW

RAW.mkdir(parents=True, exist_ok=True)
for p in PAIRS:
    out = RAW / f"{p}_task-jointaction_eeg.eeg"
    if not out.exists():
        print("downloading", out.name, flush=True)
        urllib.request.urlretrieve(f"https://s3.amazonaws.com/openneuro.org/ds007471/{p}/eeg/{out.name}", out)
