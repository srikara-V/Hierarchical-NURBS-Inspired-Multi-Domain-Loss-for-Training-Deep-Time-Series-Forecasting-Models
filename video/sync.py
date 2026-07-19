"""Map narration substrings to audio timestamps (per scene)."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")


class Timing:
    def __init__(self, sid, text):
        self.sid = sid
        self.text = text
        meta = json.load(open(os.path.join(AUDIO, f"{sid}.json")))
        self.starts = meta["starts"]
        self.ends = meta["ends"]
        self.aligned = meta["aligned_text"]
        # alignment should match narration text; if not, we map by ratio
        self.exact = (self.aligned == text)
        self.audio_end = self.ends[-1]

    def t(self, substr, off=0.0):
        """Start time (s) of the first occurrence of substr in narration."""
        idx = self.text.find(substr)
        if idx < 0:
            raise KeyError(f"[{self.sid}] anchor not found: {substr!r}")
        if self.exact:
            return max(0.0, self.starts[idx] + off)
        # fallback: find in aligned text, else scale by char ratio
        j = self.aligned.find(substr)
        if j >= 0:
            return max(0.0, self.starts[j] + off)
        frac = idx / max(1, len(self.text))
        return max(0.0, frac * self.audio_end + off)

    def t_end(self, substr, off=0.0):
        idx = self.text.find(substr)
        if idx < 0:
            raise KeyError(f"[{self.sid}] anchor not found: {substr!r}")
        j = idx + len(substr) - 1
        if self.exact:
            return max(0.0, self.ends[j] + off)
        k = self.aligned.find(substr)
        if k >= 0:
            return max(0.0, self.ends[k + len(substr) - 1] + off)
        frac = j / max(1, len(self.text))
        return max(0.0, frac * self.audio_end + off)
