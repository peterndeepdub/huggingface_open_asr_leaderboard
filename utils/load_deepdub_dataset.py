import os
import glob
import pandas as pd
import soundfile as sf
from datasets import Dataset
from datasets.features import Audio


def create_hf_dataset_from_folder(folder, dataset_name="deepdub_dataset"):
    """
    Create a Hugging Face Dataset from a folder containing .wav files and corresponding .csv transcription files.
    Each .csv must have a 'Transcript Text' column. All rows are concatenated for the text.
    Returns a datasets.Dataset object with columns: dataset, audio, text, audio_length_s, id
    """
    data = {"dataset": [], "audio": [], "text": [], "audio_length_s": [], "id": []}
    wav_files = glob.glob(os.path.join(folder, "*.wav"))
    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        csv_path = os.path.join(folder, f"{base}.csv")
        if not os.path.exists(csv_path):
            continue
        # Read and concatenate transcript
        df = pd.read_csv(csv_path)
        transcript = " ".join(df["Transcript Text"].astype(str))
        # Get audio length
        with sf.SoundFile(wav_path) as f:
            audio_length_s = len(f) / f.samplerate
        data["dataset"].append(dataset_name)
        data["audio"].append(wav_path)
        data["text"].append(transcript)
        data["audio_length_s"].append(audio_length_s)
        data["id"].append(base)
    audio_dataset = Dataset.from_dict(data).cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    return audio_dataset


if __name__ == "__main__":
    ds = create_hf_dataset_from_folder("data/deepdub_dataset/")
    print(ds)
