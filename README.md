# EduSense AI

**Multimodal Attendance & Classroom Activity Assistant**

EduSense AI includes a full local analysis app and a shareable hosted demo. The local app analyzes a webcam or video and optional room-audio clip on the computer running it. The hosted presentation demo uses generated scenarios only and has no camera, microphone, or media-upload controls.

It is designed to support teacher judgment. It does not identify students, transcribe speech, infer emotions, grade learners, or evaluate teaching quality. Optional model suggestions use an Ollama instance on the same computer; the app never calls a cloud model.

## Presentation links

- GitHub source: https://github.com/kartikk0211/edusense-ai
- Live demo: https://kartikk0211.github.io/edusense-ai/

The GitHub Pages demo is a static interactive page with generated scenarios only. The full webcam/video/audio workflow remains available in the local app and is not exposed through the public demo.
## What works

| Area | Status | Current behavior |
|---|---|---|
| Video | Implemented locally | OpenCV samples a local webcam or uploaded video and detects frontal faces. |
| Presence | Implemented as an estimate | Anonymous tracks are confirmed after sustained observations; short gaps are tolerated. |
| Audio | Implemented locally | WAV, FLAC, and OGG clips are reduced to one-second loudness and signal-shape features. |
| Signal fusion | Implemented as a heuristic | Motion and speech-like audio components are shown separately and summarized as an activity signal. |
| Teaching prompts | Implemented as rules, with optional local AI | Rule-based prompts work by default. If Ollama and a local model are already installed, a model can create another set from aggregate metrics over loopback only. |
| History | Implemented locally | Saving is explicit. The SQLite store keeps at most 50 aggregate session summaries. |
| Exports | Implemented locally | CSV data and an in-memory PDF summary can be downloaded. |
| Identity recognition | Not implemented | No face embeddings, student names, or identity matching are used. |
| Hosted presentation demo | GitHub Pages at `docs/index.html` | Synthetic scenarios only; no media capture or upload. |

## Architecture

~~~mermaid
flowchart LR
    A[Local webcam or video] --> B[OpenCV face boxes]
    B --> C[Anonymous temporal tracking]
    D[Optional audio clip] --> E[One-second audio features]
    C --> F[Transparent session metrics]
    E --> F
    F --> G[Teacher dashboard and rule-based prompts]
    F --> H[Optional local SQLite history]
    F --> I[CSV and PDF downloads]
~~~

Frames and audio are analyzed locally. Only the last annotated frame remains in the active app session as a preview. Saved history contains metrics and anonymous session-only profile labels.

## Requirements

- Python 3.12 recommended (Python 3.11+ supported)
- A webcam for local camera mode, or a video file
- A microphone or audio clip if you want audio features
- Windows, macOS, or Linux
- Camera and microphone permission in the browser when recording audio

## Install

### Windows PowerShell

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### macOS or Linux

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

## Run

Start the local dashboard from the project directory:

~~~bash
python -m streamlit run app.py
~~~

Open the local URL Streamlit prints in the terminal. The webcam option accesses the camera attached to the computer running the local app. The public presentation demo is hosted from docs/index.html through GitHub Pages and uses synthetic scenarios only. The optional Streamlit sample demo can be run with: python -m streamlit run public_demo/app.py.

For camera mode, select a camera index and set a maximum run duration. For video mode, upload a short clip. Audio can be omitted, uploaded, or recorded in the page. If supplied, an audio clip is aligned from the beginning of the video timeline; the app does not claim sample-accurate audio/video synchronization.

## Tests and checks

Run the included behavioral tests:

~~~bash
python -m unittest discover -s tests -v
~~~

Compile all project modules:

~~~bash
python -m compileall app.py edusense tests
~~~

The tests exercise presence thresholds, brief occlusions, audio-window analysis, feature alignment, recommendation rules, local storage, and PDF creation. They do not validate face detection across real classroom conditions.

## How the activity signal is calculated

For each sampled second:

- Motion signal is the mean centroid movement of active anonymous tracks, scaled and bounded to 0–100.
- Audio signal is a simple speech-likelihood estimate derived from loudness, zero-crossing rate, and spectral flatness.
- Activity signal is the mean of the available motion and audio components. If no audio was supplied, it uses motion alone.

These are broad room-level signals. Stillness can mean focused reading, and speech-like audio can come from many sources. The index is not a validated engagement or learning measure. Rule-based prompts are the default. Optional Ollama prompts send only the aggregate session summary to the local loopback service; inspect suggestions before using them. See the [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md) for endpoint details.

Roster coverage is calculated only when an expected class size is entered. It is the number of anonymous tracks confirmed by the presence threshold divided by that count. Camera misses and duplicate tracks can make it inaccurate; verify attendance with the official roster.

## Privacy and responsible use

- Processing runs locally; no API key or external AI service is required.
- Video frames are not stored in the session database. The most recent annotated preview stays in active app memory until the page is refreshed or closed.
- Audio is reduced to one-second features. EduSense AI does not create transcripts or identify speakers.
- Uploaded media can remain in the active Streamlit session while the page is open. A temporary video copy is removed after analysis.
- Saved data contains aggregate metrics, timestamps, an input-source label, and temporary profile IDs. It does not contain names, face images, embeddings, or raw audio.
- History is stored in a user-local SQLite file and capped at 50 sessions. Use the History page to clear it.
- Do not use the activity signal for student ranking, discipline, grading, diagnosis, or other high-stakes decisions. Provide appropriate notice and follow your institution's privacy rules.

See docs/PRIVACY.md for data locations and limits.

## Project layout

~~~text
app.py                  Streamlit dashboard
edusense/tracking.py     Anonymous sustained-presence tracker
edusense/vision.py       Local webcam/video sampling
edusense/audio.py        One-second audio activity features
edusense/analytics.py    Timestamp alignment and session summaries
edusense/recommendations.py
edusense/storage.py      Local SQLite history
edusense/reporting.py    In-memory PDF generation
tests/                   Offline behavior tests
docs/                    Privacy and demo notes
~~~

No license file is included. The repository is public for viewing; reuse rights remain subject to copyright unless a license is added.
