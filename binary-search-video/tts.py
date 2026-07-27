"""ElevenLabs narration synthesis with disk cache."""
import hashlib
import json
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", "cache")
os.makedirs(CACHE, exist_ok=True)


def _key(voice, model, text):
    return hashlib.sha1(f"{voice}|{model}|{text}".encode()).hexdigest()[:20]


def probe_dur(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(out.stdout.strip())


def synth(text, voice, model, path, api_key):
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
           f"?output_format=mp3_44100_128")
    body = {"text": text, "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8,
                               "style": 0.1, "use_speaker_boost": True}}
    for attempt in range(4):
        try:
            r = requests.post(url, headers={"xi-api-key": api_key},
                              json=body, timeout=180)
            if r.status_code == 200 and len(r.content) > 800:
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, path)
                return
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            if attempt == 3:
                raise
            wait = 3 * (attempt + 1)
            print(f"  retry ({e}) in {wait}s", flush=True)
            time.sleep(wait)


def ensure_all(beats, voice, model, fake=False):
    """Returns dict key -> (mp3_path_or_None, duration_seconds)."""
    api_key = os.environ.get("ELEVEN_LABS_API", "")
    out = {}
    todo = []
    for bt in beats:
        k = _key(voice, model, bt.text)
        mp3 = os.path.join(CACHE, k + ".mp3")
        meta = os.path.join(CACHE, k + ".json")
        if os.path.exists(mp3) and os.path.exists(meta):
            out[bt.key] = (mp3, json.load(open(meta))["dur"])
        elif fake:
            out[bt.key] = (None, 0.6 + len(bt.text) / 16.0)
        else:
            todo.append((bt, mp3, meta))
    for i, (bt, mp3, meta) in enumerate(todo):
        print(f"[tts {i + 1}/{len(todo)}] {bt.key} ({len(bt.text)} ch)",
              flush=True)
        synth(bt.text, voice, model, mp3, api_key)
        dur = probe_dur(mp3)
        json.dump({"dur": dur, "beat": bt.key}, open(meta, "w"))
        out[bt.key] = (mp3, dur)
    return out


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import storyboard as sb
    fake = "--fake" in sys.argv
    res = ensure_all(sb.all_beats(), sb.VOICE_ID, sb.MODEL_ID, fake=fake)
    total_speech = sum(d for _, d in res.values())
    total_chars = sum(len(bt.text) for bt in sb.all_beats())
    print(f"beats: {len(res)} | chars: {total_chars} | "
          f"speech: {total_speech / 60:.1f} min")
