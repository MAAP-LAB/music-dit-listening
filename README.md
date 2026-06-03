# Music DiT Steering — Single-Listener Listening Pilot

Companion materials for *Probing-Based Test-Time Steering of Music Diffusion Transformers* (ICML 2026 ML for Audio Workshop).

**Take the test:** https://maap-lab.github.io/music-dit-listening/

About 9 minutes, headphones strongly recommended.

## What you do
- 13 trials. For 9 of them, listen to two short clips (A and B) and pick which has more, less, or about the same amount of a target concept (drums, bass, electronic).
- The remaining 4 trials ask you to rate the audio quality of a single clip on a 1–5 scale.
- At the end your answers download as a small CSV. Please email it to `arsol970812@gmail.com`.

No audio leaves your browser. Only the CSV you choose to send is shared.

## What's in this repo
- `index.html` — the survey page
- `stimuli/` — 22 short WAV clips (paired baseline vs steered, plus solo MOS clips)
- `stimuli_metadata.json` — ground-truth mapping for trials (not shown to listeners)
- `analyze.py` — the script used to score returned CSVs

## How the stimuli were generated
Each pair comes from Stable Audio Open's DiT with the same random seed for both clips. The "steered" clip has a learned concept activation vector injected at layer 12 with strength ±15. Trials are chosen so that the addition or removal is musically plausible — for example, removing drums from a rock or hip-hop excerpt, or adding bass to an ambient pad.

## License
Code under MIT. Audio for research / evaluation use only.
