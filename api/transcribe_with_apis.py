import os
import time
from io import BytesIO
from typing import Optional

import assemblyai as aai
import openai
import requests
import soundfile as sf
from deepgram import DeepgramClient, PrerecordedOptions, PrerecordedResponse, FileSource
from elevenlabs.client import ElevenLabs
from httpx import HTTPStatusError
from requests_toolbelt import MultipartEncoder
from rev_ai import apiclient
from rev_ai.models import CustomerUrlData
from speechmatics.batch_client import BatchClient
from speechmatics.models import BatchTranscriptionConfig, ConnectionSettings, FetchData

from utils.constants import (
    ASSEMBLY_LANGUAGE_MAP,
    DEEPGRAM_LANGUAGE_MAP,
    ELEVENLABS_LANGUAGE_MAP,
    OPENAI_LANGUAGE_MAP,
    REVAI_LANGUAGE_MAP,
    SPEECHMATICS_LANGUAGE_MAP,
)
from utils.enums import Languages


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    language: Languages,
    max_retries=10,
    use_url=False,
):
    retries = 0
    while retries <= max_retries:
        try:
            PREFIX = "speechmatics/"
            if model_name.startswith(PREFIX):
                api_key = os.getenv("SPEECHMATICS_API_KEY")
                if not api_key:
                    raise ValueError(
                        "SPEECHMATICS_API_KEY environment variable not set"
                    )

                settings = ConnectionSettings(
                    url="https://asr.api.speechmatics.com/v2", auth_token=api_key
                )
                with BatchClient(settings) as client:
                    config = BatchTranscriptionConfig(
                        language=SPEECHMATICS_LANGUAGE_MAP[language],
                        enable_entities=True,
                        operating_point=model_name[len(PREFIX) :],
                    )

                    job_id = None
                    audio_url = None
                    try:
                        if use_url:
                            audio_url = sample["row"]["audio"][0]["src"]
                            config.fetch_data = FetchData(url=audio_url)
                            multipart_data = MultipartEncoder(
                                fields={"config": config.as_config().encode("utf-8")}
                            )
                            response = client.send_request(
                                "POST",
                                "jobs",
                                data=multipart_data.to_string(),
                                headers={"Content-Type": multipart_data.content_type},
                            )
                            job_id = response.json()["id"]
                        else:
                            job_id = client.submit_job(audio_file_path, config)

                        transcript = client.wait_for_completion(
                            job_id, transcription_format="txt"
                        )
                        return transcript
                    except HTTPStatusError as e:
                        if e.response.status_code == 401:
                            raise ValueError(
                                "Invalid Speechmatics API credentials"
                            ) from e
                        elif e.response.status_code == 400:
                            raise ValueError(
                                f"Speechmatics API responded with 400 Bad request: {e.response.text}"
                            )
                        raise e
                    except Exception as e:
                        if job_id is not None:
                            status = client.check_job_status(job_id)
                            if (
                                audio_url is not None
                                and "job" in status
                                and "errors" in status["job"]
                                and isinstance(status["job"]["errors"], list)
                                and len(status["job"]["errors"]) > 0
                            ):
                                errors = status["job"]["errors"]
                                if (
                                    "message" in errors[-1]
                                    and "failed to fetch file" in errors[-1]["message"]
                                ):
                                    retries = max_retries + 1
                                    raise Exception(
                                        f"could not fetch URL {audio_url}, not retrying"
                                    )

                        raise Exception(
                            f"Speechmatics transcription failed: {str(e)}"
                        ) from e

            elif model_name.startswith("assembly/"):
                aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(
                    speech_model=model_name.split("/")[1],
                    language_code=ASSEMBLY_LANGUAGE_MAP[language],
                )
                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    audio_duration = sample["row"]["audio_length_s"]
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    transcript = transcriber.transcribe(audio_url, config=config)
                else:
                    audio_duration = (
                        len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                    )
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    transcript = transcriber.transcribe(audio_file_path, config=config)

                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(
                        f"AssemblyAI transcription error: {transcript.error}"
                    )
                return transcript.text

            elif model_name.startswith("openai/"):
                if use_url:
                    response = requests.get(sample["row"]["audio"][0]["src"])
                    audio_data = BytesIO(response.content)
                    response = openai.Audio.transcribe(
                        model=model_name.split("/")[1],
                        file=audio_data,
                        response_format="text",
                        language=OPENAI_LANGUAGE_MAP[language],
                        temperature=0.0,
                    )
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        response = openai.Audio.transcribe(
                            model=model_name.split("/")[1],
                            file=audio_file,
                            response_format="text",
                            language=OPENAI_LANGUAGE_MAP[language],
                            temperature=0.0,
                        )
                return response.strip()

            elif model_name.startswith("elevenlabs/"):
                client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                if use_url:
                    response = requests.get(sample["row"]["audio"][0]["src"])
                    audio_data = BytesIO(response.content)
                    transcription = client.speech_to_text.convert(
                        file=audio_data,
                        model_id=model_name.split("/")[1],
                        language_code=ELEVENLABS_LANGUAGE_MAP[language],
                        tag_audio_events=True,
                    )
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        transcription = client.speech_to_text.convert(
                            file=audio_file,
                            model_id=model_name.split("/")[1],
                            language_code=ELEVENLABS_LANGUAGE_MAP[language],
                            tag_audio_events=True,
                        )
                return transcription.text

            elif model_name.startswith("revai/"):
                access_token = os.getenv("REVAI_API_KEY")
                client = apiclient.RevAiAPIClient(access_token)

                if use_url:
                    # Submit job with URL for Rev.ai
                    job = client.submit_job_url(
                        transcriber=model_name.split("/")[1],
                        source_config=CustomerUrlData(sample["row"]["audio"][0]["src"]),
                        metadata="benchmarking_job",
                        language=REVAI_LANGUAGE_MAP[language],
                    )
                else:
                    # Submit job with local file
                    job = client.submit_job_local_file(
                        transcriber=model_name.split("/")[1],
                        filename=audio_file_path,
                        metadata="benchmarking_job",
                        language=REVAI_LANGUAGE_MAP[language],
                    )

                # Polling until job is done
                while True:
                    job_details = client.get_job_details(job.id)
                    if job_details.status.name in ["IN_PROGRESS", "TRANSCRIBING"]:
                        time.sleep(0.1)
                        continue
                    elif job_details.status.name == "FAILED":
                        raise Exception("RevAI transcription failed.")
                    elif job_details.status.name == "TRANSCRIBED":
                        break

                transcript_object = client.get_transcript_object(job.id)

                # Combine all words from all monologues
                transcript_text = []
                for monologue in transcript_object.monologues:
                    for element in monologue.elements:
                        transcript_text.append(element.value)

                return "".join(transcript_text) if transcript_text else ""

            elif model_name.startswith("deepgram/"):
                access_token = os.getenv("DEEPGRAM_API_KEY")
                deepgram_client = DeepgramClient(access_token)

                if use_url:
                    raise ValueError("Not implemented for url")
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        buffer_data = audio_file.read()

                    payload: FileSource = {
                        "buffer": buffer_data,
                    }
                    model = "nova-3"
                    if language in [Languages.JA, Languages.KO]:
                        model = "nova-2"
                    response = deepgram_client.listen.rest.v("1").transcribe_file(
                        source=payload,
                        options=PrerecordedOptions(
                            model=model,
                            language=DEEPGRAM_LANGUAGE_MAP[language],
                        ),  # Apply other options
                    )

                return response.results.channels[0].alternatives[0].transcript

            else:
                raise ValueError(
                    "Invalid model prefix, must start with 'assembly/', 'openai/', 'elevenlabs/' or 'revai/' or 'deepgram/'"
                )

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            if not use_url:
                sf.write(
                    audio_file_path,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
            delay = 1
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)
