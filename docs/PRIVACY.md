# Privacy and data handling

EduSense AI is designed for a local classroom demo. Its outputs are aggregate signals for a teacher to review.

## Public hosted demo

The GitHub Pages demo in docs/index.html uses generated scenario values in the visitor’s browser. It has no webcam, microphone, or media-upload controls, and it does not save session history. Scenario and class-size controls render synthetic charts and downloadable synthetic CSV data. No real student media is sent to GitHub Pages by this demo.

The separate app.py local workflow accepts local webcam/video/audio input. Keep real classroom media in that local workflow; do not put student media into public repositories, issues, or demo pages.
## Data flow

1. The app opens a local webcam or a video file selected by the teacher.
2. OpenCV samples frames and returns face rectangles. It does not create biometric templates or compare faces with a roster.
3. A temporary video file is created only when OpenCV needs a path for an uploaded clip. The app removes that file after analysis.
4. Optional audio is decoded in memory and reduced to one-second RMS, dBFS, and signal-shape estimates. No transcript or voice identity is produced.
5. The dashboard can save an aggregate report to SQLite after the teacher presses Save summary locally.

## Stored information

The local database keeps session start time, source label, aggregate session summary, per-sample measurements, and anonymous profile IDs with their presence status. It does not save video frames, face embeddings, names, or raw audio.

The final annotated preview is held in active app memory so it can be shown on screen. Uploaded media may also remain in the Streamlit session until the page is closed or refreshed. Nothing in the app sends these values to a remote API. If you enable the optional Ollama assistant, only aggregate session numbers are sent to the local loopback endpoint on this same computer; no media or profile labels are included.

## Database location

The default file is:

- Windows: %LOCALAPPDATA%\EduSenseAI\sessions.sqlite3
- macOS/Linux: $XDG_DATA_HOME/EduSenseAI/sessions.sqlite3
- If XDG_DATA_HOME is not set: ~/.local/share/EduSenseAI/sessions.sqlite3

Set EDUSENSE_DATA_DIR to choose a different parent directory. The application creates an EduSenseAI folder inside it.

The History page can delete saved sessions. You can also close the app and remove the SQLite file to clear all persisted history.

## Limits

- Face detection can miss faces, especially with side views, low light, distance, or partial obstruction.
- Anonymous tracking may split one person into multiple IDs or join nearby detections.
- Roster coverage is a camera estimate, not verified attendance.
- Audio activity labels can mistake music, noise, or silence for other categories.
- Audio/video timing is approximate when the audio clip was recorded separately.
- The activity signal is not a measure of attention, learning, emotion, or teaching effectiveness.

Use the app only with appropriate classroom notice and consent. Keep teacher judgment in control of all decisions.
