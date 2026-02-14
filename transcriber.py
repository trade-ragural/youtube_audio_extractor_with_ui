"""
Unified transcription module supporting:
1. YouTube Transcript API (for YouTube videos) - faster, free, native captions
2. Gemini ASR (for local files) - fallback
"""

import os
import time
import math
from dataclasses import dataclass
from typing import List, Optional, Dict

from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed


# Default API key (override via GEMINI_API_KEY env var)
DEFAULT_API_KEY = "AIzaSyD5gvrqdbSeNGIWW38EHC2GLBsb8lGjr_g"

TRANSCRIPTION_PROMPT = """Transcribe verbatim. Detect language.
Format:
Language: <lang>
---
<text>"""

REFINE_PROMPT = """You are a text refinement tool. Your ONLY job is to add punctuation.
"Dont Summarize" Keep the flow as it is. Each and every line of text is very very important.
You just need to make sure to put the punctuations, commas, full stops etc to the text.
Do NOT change any words or capitalization unless necessary for sentence structure.
Keep the language as {lang}.

Text:
{text}"""


@dataclass
class TranscriptionResult:
    """Result from transcribing a single audio chunk."""
    chunk_index: int
    text: str
    language: str
    start_time: float
    end_time: float


def _get_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    # Simple extraction logic
    if "youtu.be" in url:
        return url.split("/")[-1]
    if "youtube.com" in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return params.get("v", [None])[0]
    return None


def check_transcript_availability(video_id: str) -> bool:
    """Check if a transcript (manual or auto) is available for the video."""
    try:
        api = YouTubeTranscriptApi()
        # list() is fast and doesn't fetch content
        transcript_list = api.list(video_id)
        # If we get a list object, transcripts exist
        return True
    except Exception:
        return False


def fetch_youtube_transcript(video_id: str, chunks: List) -> List[TranscriptionResult]:
    """
    Fetch transcript using youtube_transcript_api and align to chunks.
    Tries multiple languages (en, kn, hi, auto-generated).
    """
    try:
        # Instantiate API
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try to find a transcript in this priority order
        # Manually created > Auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(['en', 'kn', 'hi'])
        except:
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'kn', 'hi'])
            except:
                # Fallback to whatever is available
                transcript = transcript_list.find_transcript(['en', 'kn', 'hi'])
        
        full_transcript = transcript.fetch()
        lang_code = transcript.language_code
        
        # Align transcript entries to our chunks
        results = []
        for chunk in chunks:
            chunk_text = []
            
            # Simple alignment: include entries that start within the chunk
            # or overlap significantly
            for entry in full_transcript:
                # Handle both object attributes (v1.2.3) and dict access (older/newer)
                try:
                    start = entry.start
                    duration = entry.duration
                    text = entry.text
                except AttributeError:
                    start = entry['start']
                    duration = entry['duration']
                    text = entry['text']
                
                end = start + duration
                
                # Check if entry falls within this chunk's time window
                # We include it if its midpoint is within the chunk
                midpoint = start + (duration / 2)
                
                if chunk.start_time <= midpoint < chunk.end_time:
                    chunk_text.append(text)
            
            results.append(TranscriptionResult(
                chunk_index=chunk.chunk_index,
                text=" ".join(chunk_text).strip(),
                language=lang_code,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            ))
            
        return results

    except Exception as e:
        print(f"YouTube transcript fetch failed: {e}")
        return []  # Return empty to trigger fallback or error


def _get_client() -> genai.Client:
    """Create a Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY)
    return genai.Client(api_key=api_key)


def transcribe_chunk_gemini(
    client: genai.Client,
    audio_path: str,
    chunk_index: int,
    start_time: float,
    end_time: float,
) -> TranscriptionResult:
    """
    Transcribe a single audio chunk using Gemini.
    """
    # Upload the audio file to Gemini
    uploaded_file = client.files.upload(file=audio_path)

    # Wait for file to be processed
    max_wait = 30
    waited = 0
    while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
        time.sleep(1)
        waited += 1
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"File upload failed for chunk {chunk_index}")

    # Call Gemini with audio + prompt
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=TRANSCRIPTION_PROMPT),
                ]
            )
        ],
    )

    # Parse the response
    raw_text = response.text.strip()
    language, text = _parse_response(raw_text)

    # Clean up uploaded file
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass  # Non-critical cleanup

    return TranscriptionResult(
        chunk_index=chunk_index,
        text=text,
        language=language,
        start_time=start_time,
        end_time=end_time,
    )


def _parse_response(raw_text: str) -> tuple:
    """
    Parse Gemini response to extract language and transcription text.
    """
    language = "Auto-detected"
    text = raw_text

    if "---" in raw_text:
        parts = raw_text.split("---", 1)
        header = parts[0].strip()
        text = parts[1].strip()

        for line in header.split("\n"):
            line = line.strip()
            if line.lower().startswith("language:"):
                language = line.split(":", 1)[1].strip()
                break
    elif raw_text.lower().startswith("language:"):
        lines = raw_text.split("\n", 1)
        language = lines[0].split(":", 1)[1].strip()
        text = lines[1].strip() if len(lines) > 1 else raw_text

    return language, text


def refine_text_gemini(client: genai.Client, text: str, lang: str) -> str:
    """Uses Gemini to add punctuation and fix capitalization."""
    if not text or not text.strip():
        return text
        
    try:
        # Simple retry logic
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=REFINE_PROMPT.format(lang=lang, text=text)
                )
                return response.text.strip()
            except Exception:
                if attempt == 1: raise
                time.sleep(1)
    except Exception as e:
        # Fallback to original text on failure
        print(f"Refinement failed: {e}")
        return text


def transcribe_all_chunks(
    chunks,
    youtube_url: Optional[str] = None,
    progress_callback=None,
) -> List[TranscriptionResult]:
    """
    Main transcription entry point.
    1. If YouTube URL provided -> Try fetching transcript via API first.
    2. If that fails or no URL -> Fallback to Gemini for each chunk.
    """
    # 1. Try YouTube API if URL exists
    if youtube_url:
        video_id = _get_video_id(youtube_url)
        if video_id:
            if progress_callback:
                progress_callback(0.2, "Fetching YouTube transcript...")
            
            yt_results = fetch_youtube_transcript(video_id, chunks)
            if yt_results:
                if progress_callback:
                    progress_callback(0.3, "Refining transcript with Gemini (adding punctuation)...")
                
                # Refine with Gemini in parallel
                client = _get_client()
                total = len(yt_results)
                completed = 0
                
                def refine_task(result):
                    refined_text = refine_text_gemini(client, result.text, result.language)
                    return result, refined_text

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(refine_task, r) for r in yt_results]
                    
                    for future in as_completed(futures):
                        r, refined_text = future.result()
                        r.text = refined_text
                        completed += 1
                        if progress_callback:
                            prog = 0.3 + (0.7 * completed / total)
                            progress_callback(prog, f"Refining chunk {completed}/{total}...")
                            
                return yt_results

    # 2. Fallback to Gemini
    client = _get_client()
    results = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress = 0.2 + (0.75 * i / total)
            progress_callback(progress, f"Transcribing chunk {i + 1}/{total} (Gemini)...")

        try:
            result = transcribe_chunk_gemini(
                client=client,
                audio_path=chunk.audio_path,
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
            results.append(result)
        except Exception as e:
            results.append(TranscriptionResult(
                chunk_index=chunk.chunk_index,
                text=f"[Transcription error: {str(e)}]",
                language="Unknown",
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            ))

    if progress_callback:
        progress_callback(1.0, "Transcription complete!")

    return results
