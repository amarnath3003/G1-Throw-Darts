# Darts video collection and review

Batch 01 is ready for human review: 20 body-motion candidates and 4 qualitative
grip/release references, cut from six downloaded sources. The local archive has
119 videos from the dataset listed in the master plan and three YouTube coaching
videos. This is a first review batch, not a claim that collection or motion
validation is complete. No user selections were made during preparation.

Update, 28 October 2026: the user subsequently approved D01–D06 (all Group A)
and R03–R04 (Gary side-view references). These eight clips are listed in
`selected/manifest.json`; the six body clips have 30 FPS exports and frame maps.
The selected-bundle validation passed. Other candidates remain pending.

## Review

From this repository's root (`hackathon/`):

```powershell
python -m pip install -r video_dataset/requirements.txt
python video_dataset/review.py
```

Open http://127.0.0.1:8766. Use the existing `video_to_g1/.venv` interpreter when
working from the parent robotics workspace; FFmpeg and FFprobe must be on PATH.

- **Okay — select:** approve this clip, save its native inspection trim and
  provenance into `selected/<ID>/`, and add it to `selected/manifest.json`.
- Approved **body** clips also receive `gvhmr_30fps.mp4` and `frame_map.csv`.
  The map identifies every output frame's original source frame and encoded time.
- Approved grip/release references stay qualitative and get no GVHMR export.
- **Skip / Reset choice:** remove the clip's eligibility from the manifest.
  Previously exported files are retained as archival files. Downstream consumers
  must read the manifest rather than globbing folders for videos.
- The server binds only to loopback. Choices persist in `catalog/decisions.json`.
  Restarting the server preserves choices. The UI never launches GVHMR.

For conversational approval, the agent can call `review.decide` with the IDs the
human explicitly approved and an actor such as `human_chat`. Do not approve clips
on the user's behalf while inspecting the gallery.

## Initial recommendations and limitations

- D01–D18: preferable body-motion candidates. Fixed camera, visible feet and torso,
  and a clear throw sequence in inspected frames. The frontal/oblique view
  foreshortens the throwing arm. Release is blurred and the dart is small.
- D19–D20: additional appearance group, with throwing-hand visibility near the
  frame boundary requiring careful review. Lower confidence than D01–D18.
- R01: grip demonstration, no body reconstruction value.
- R02–R04: side-on upper-body/slow-motion references. Feet are cropped; timing
  factors are unknown. Do not use their encoded speed as physical throwing speed.
- The dataset card says four participants. Appearance groups A–D are provisional
  labels, not verified identities or session labels. Handedness/mirroring has not
  been verified. Keep the entire HF dataset in a single conservative leakage group
  until identity/session annotation is resolved. Do not split random clips across
  train/test or claim these groups are independent camera sessions.
- Body sources are encoded at **20 FPS**, with unknown capture settings. Thirty FPS
  export uses nearest original frames, including repeats; it creates no new motion
  information. Timing is source-encoded time, not independently verified real time.
  Velocity supervision remains disabled pending timing verification.
- Release windows are coarse visual review cues; they do not establish fingertip
  separation. Targets, impact positions and success labels are unknown.
- Whole-file SHA256 deduplication found no exact duplicates. Overlapping recordings,
  alternate encodes and reused footage still require perceptual/session inspection.
- The current checks establish decoding, trim frames and review/export mechanics.
  They do not establish GVHMR pose quality or robotic tracking feasibility.

## Provenance and permissions

Dataset: [pratikshapai/human-motion-tracking-deeplabcut](https://huggingface.co/datasets/pratikshapai/human-motion-tracking-deeplabcut),
revision `202b971078b1b1be719bab211bfdc7b9848bed71`.
Its card describes 100+ darts videos and four participants. No license is declared
in that card. Original camera operator, participant consent and redistribution
permissions have not been established. Downloaded for the user's local inspection;
do not publish the files as licensed dataset content.

Downloaded coaching sources:

- [England Darts Reviews — How I throw darts in detail and slow motion](https://www.youtube.com/watch?v=KV2XqzIJDgw)
- [Darts Review Channel — Gary Anderson darts throw in slow motion](https://www.youtube.com/watch?v=KdbdYeST2fE)
- [The Darts Box — How to Perfect Your Stance and Throw in Darts](https://www.youtube.com/watch?v=NRcxlA_eyeQ)

These metadata did not establish reuse permission. Their license/redistribution
status stays unverified. The Darts Box source is archived but not in Batch 01:
sampled footage emphasizes cropped stance views, board imagery and edited layouts.

Raw videos, generated previews and selected media are ignored by Git so the repo's
MIT code license does not silently become a media license. Full source metadata
stays in `catalog/sources/`; yt-dlp transport metadata stays beside originals in
`sources/youtube/` and may contain expiring CDN URLs. Do not publish that transport
metadata. The personally recorded data required by the challenge is still pending.

## Files and repeatable commands

```text
sources/                 unchanged downloaded sources (ignored by Git)
catalog/hf_inventory.json pinned revision, file inventory and acquisition metadata
catalog/sources/         provenance, checksums, codec/FPS/resolution/duration
catalog/frame_times/     original presentation times used for exact frame trims
catalog/candidates.json  batch manifest, flags, time/frame mappings
catalog/decisions.json   human choices; created on the first decision
candidates/<ID>/         H.264 native-rate trim, poster, sheet and metadata
selected/manifest.json   sole eligibility list for later GVHMR consumption
selected/<ID>/           approved media, metadata and body-only 30 FPS export
catalog/validation.json  decode and isolated approval/export check report
```

From `hackathon/`:

```powershell
python video_dataset/collect.py hf --all
python -m yt_dlp --no-playlist -f "bestvideo[height<=1080][ext=mp4]" --write-info-json -o "video_dataset/sources/youtube/%(id)s.%(ext)s" https://www.youtube.com/watch?v=KV2XqzIJDgw https://www.youtube.com/watch?v=KdbdYeST2fE https://www.youtube.com/watch?v=NRcxlA_eyeQ
python video_dataset/collect.py youtube-index
python video_dataset/prepare.py
python video_dataset/check_dataset.py
python video_dataset/review.py
```

`prepare.py` defines explicit frame-based trims, preserves source timing, strips
audio in inspection copies, and never approves media. Original files keep their
downloaded codec and resolution; YouTube acquisitions used video-only tracks.
`check_dataset.py` decodes all sources/candidates and exercises approval, rejection,
reset, routing and 30 FPS timing maps **inside an OS temporary directory**.

Next gate: review additional candidates if needed, refine release/phase labels, resolve mirror/session
identity, and run GVHMR/GMR only on approved body clips. The master plan's 20 accepted
throws threshold has not been reached merely by proposing 20 candidates.
