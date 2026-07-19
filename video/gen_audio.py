"""Generate narration audio per scene via ElevenLabs with char timestamps."""
import os, json, base64, time, sys
import requests
from narration import SCENES, VOICE_ID, MODEL_ID, VOICE_SETTINGS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "audio")
os.makedirs(OUT, exist_ok=True)

KEY = os.environ["ELEVEN_LABS_API"]
URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"


def synth(text, prev_text=None, next_text=None, tries=4):
    body = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
    }
    if prev_text:
        body["previous_text"] = prev_text[-500:]
    if next_text:
        body["next_text"] = next_text[:500]
    for k in range(tries):
        r = requests.post(URL, headers={"xi-api-key": KEY},
                          json=body, params={"output_format": "mp3_44100_128"},
                          timeout=300)
        if r.ok:
            return r.json()
        wait = 2 ** (k + 1)
        print(f"  retry {k+1} after HTTP {r.status_code}: {r.text[:200]}",
              flush=True)
        time.sleep(wait)
    raise RuntimeError(f"TTS failed: {r.status_code} {r.text[:300]}")


def main():
    only = set(sys.argv[1:])
    for i, sc in enumerate(SCENES):
        sid = sc["id"]
        if only and sid not in only:
            continue
        mp3 = os.path.join(OUT, f"{sid}.mp3")
        meta = os.path.join(OUT, f"{sid}.json")
        if os.path.exists(mp3) and os.path.exists(meta) and not only:
            print(f"{sid}: exists, skip", flush=True)
            continue
        prev_text = SCENES[i - 1]["text"] if i > 0 else None
        next_text = SCENES[i + 1]["text"] if i + 1 < len(SCENES) else None
        print(f"{sid}: synthesizing {len(sc['text'])} chars ...", flush=True)
        d = synth(sc["text"], prev_text, next_text)
        with open(mp3, "wb") as f:
            f.write(base64.b64decode(d["audio_base64"]))
        al = d["alignment"]
        joined = "".join(al["characters"])
        with open(meta, "w") as f:
            json.dump({
                "text": sc["text"],
                "aligned_text": joined,
                "starts": al["character_start_times_seconds"],
                "ends": al["character_end_times_seconds"],
            }, f)
        dur = al["character_end_times_seconds"][-1]
        match = "exact" if joined == sc["text"] else "MISMATCH"
        print(f"{sid}: {dur:.1f}s  align={match}", flush=True)
    total = 0.0
    for sc in SCENES:
        meta = os.path.join(OUT, f"{sc['id']}.json")
        if os.path.exists(meta):
            total += json.load(open(meta))["ends"][-1]
    print(f"TOTAL narration: {total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
