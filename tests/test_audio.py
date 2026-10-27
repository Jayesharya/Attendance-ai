from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
import soundfile as sf

from edusense.audio import analyze_audio_bytes


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    output = BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class AudioAnalysisTests(unittest.TestCase):
    def test_audio_is_reduced_to_one_second_windows(self) -> None:
        sample_rate = 16000
        time_axis = np.arange(sample_rate * 2 + 4000) / sample_rate
        samples = 0.15 * np.sin(2 * np.pi * 220 * time_axis)
        windows = analyze_audio_bytes(wav_bytes(samples))
        self.assertEqual(len(windows), 3)
        self.assertAlmostEqual(windows[0].start_seconds, 0.0)
        self.assertAlmostEqual(windows[1].start_seconds, 1.0)
        self.assertEqual(windows[0].activity, "speech-like")
        self.assertGreater(windows[0].speech_likelihood, 0.4)

    def test_silence_is_reported_as_quiet(self) -> None:
        windows = analyze_audio_bytes(wav_bytes(np.zeros(16000, dtype=np.float32)))
        self.assertEqual(windows[0].activity, "quiet")
        self.assertLess(windows[0].rms_dbfs, -48.0)

    def test_noise_like_signal_is_not_labeled_speech(self) -> None:
        rng = np.random.default_rng(7)
        noise = (rng.standard_normal(16000) * 0.2).astype(np.float32)
        windows = analyze_audio_bytes(wav_bytes(noise))
        self.assertEqual(windows[0].activity, "ambient sound")

    def test_empty_or_invalid_audio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_audio_bytes(b"")
        with self.assertRaises(ValueError):
            analyze_audio_bytes(b"not an audio file")


if __name__ == "__main__":
    unittest.main()
