import os 
import subprocess
import torch

import whisper
import json
import os
import time
import torch


# ===================================
# CHECK GPU
# ===================================

print("\nChecking GPU...")

if torch.cuda.is_available():

    device = "cuda"

    print(
        "GPU FOUND →",
        torch.cuda.get_device_name(0)
    )

else:

    device = "cpu"

    print("GPU NOT FOUND → CPU MODE")


# ===================================
# LOAD MODEL
# ===================================

print("\nLoading Whisper...\n")

model = whisper.load_model(
    "turbo"
).to(device)


# ===================================
# CREATE JSON OUTPUT FOLDER
# ===================================

os.makedirs(
    os.path.join(os.path.dirname(__file__), "../../data/chunks"),
    exist_ok=True
)


# ===================================
# GET ALL FILES
# ===================================

audios = sorted(
    os.listdir(os.path.join(os.path.dirname(__file__), "../../data/audios"))
)

total = len(audios)

print(
    f"\nFound {total} files\n"
)


# ===================================
# PROCESS FILES
# ===================================

for i, audio in enumerate(audios, start=1):


    # Skip non mp3
    if not audio.endswith(".mp3"):
        continue


    start_time = time.time()


    # ============================
    # Extract Number + Title
    # ============================

    filename = audio[:-4]

    parts = filename.split(" - ")


    number = parts[0]


    if len(parts) > 1:
        title = parts[1]

    else:
        title = filename


    print(
        f"\n[{i}/{total}] Processing → {title}"
    )


    # ============================
    # TRANSCRIBE
    # ============================

    result = model.transcribe(

        audio=os.path.join(os.path.dirname(__file__), f"../../data/audios/{audio}"),

        language="en",

        task="transcribe",

        word_timestamps=False,

        fp16=(device == "cuda")
    )


    # ============================
    # BUILD CHUNKS
    # ============================

    chunks = []


    for segment in result["segments"]:


        chunks.append({

            "number": number,

            "title": title,

            "start": round(
                segment["start"],
                1
            ),

            "end": round(
                segment["end"],
                1
            ),

            "text": segment["text"].strip()

        })


    # ============================
    # FINAL JSON
    # ============================

    final_data = {

        "chunks": chunks,

        "text": result[
            "text"
        ].strip()

    }


    # ============================
    # SAVE
    # ============================

    save_path = (

        os.path.join(os.path.dirname(__file__), f"../../data/chunks/{audio}.json")

    )


    with open(

        save_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            final_data,

            f,

            indent=4,

            ensure_ascii=False

        )


    elapsed = round(
        time.time() -
        start_time,
        1
    )


    print(
        f"Saved → {save_path}"
    )

    print(
        f"Time → {elapsed}s"
    )


print("\nALL FILES COMPLETED ✅")
