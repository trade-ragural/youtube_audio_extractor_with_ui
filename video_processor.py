"""
Video processing module: YouTube download, audio extraction, and chunking.
"""

import os
import tempfile
import subprocess
import shutil
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkInfo:
    """Represents a single video/audio chunk."""
    chunk_index: int
    video_path: str
    audio_path: str
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds

    @property
    def start_timestamp(self) -> str:
        return _format_timestamp(self.start_time)

    @property
    def end_timestamp(self) -> str:
        return _format_timestamp(self.end_time)

    @property
    def label(self) -> str:
        return f"Chunk {self.chunk_index + 1}  —  {self.start_timestamp} → {self.end_timestamp}"


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_youtube_video_metadata(url: str) -> dict:
    """Get video metadata (title, duration) from YouTube without download."""
    try:
        cmd = [
            "yt-dlp",
            "--print", "%(title)s",
            "--print", "%(duration)s",
            "--no-playlist",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        return {
            "title": lines[0],
            "duration": float(lines[1])
        }
    except Exception:
        return {"title": "transcript_export", "duration": 0}


def get_video_title(url: str) -> str:
    # Use generic function
    return get_youtube_video_metadata(url)["title"]



def download_youtube(url: str, output_dir: str, progress_callback=None) -> str:
    """
    Download a YouTube video using yt-dlp.
    Returns the path to the downloaded video file.
    """
    if progress_callback:
        progress_callback(0.05, "Downloading YouTube video...")

    output_template = os.path.join(output_dir, "source_video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("source_video"):
            return os.path.join(output_dir, f)

    raise FileNotFoundError("Downloaded video file not found")


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video as WAV (16kHz mono for best ASR quality)."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # mono
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return output_path


def chunk_video(
    video_path: str,
    chunk_duration_sec: int,
    output_dir: str,
    progress_callback=None,
    is_youtube: bool = False,
    duration: float = None,
) -> List[ChunkInfo]:
    """
    Split a video into chunks of specified duration.
    If is_youtube=True, just metadata chunks (no ffmpeg splitting).
    """
    if is_youtube:
        if duration is None:
            raise ValueError("Duration required for YouTube virtual chunking")
        total_duration = duration
    else:
        total_duration = get_video_duration(video_path)

    num_chunks = math.ceil(total_duration / chunk_duration_sec)

    if progress_callback:
        progress_callback(0.15, f"Creating {num_chunks} chunks of {chunk_duration_sec}s each...")

    chunks = []
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        # Last chunk may be shorter
        end_time = min((i + 1) * chunk_duration_sec, total_duration)
        actual_duration = end_time - start_time

        if actual_duration < 0.5:
            continue

        if is_youtube:
            # Virtual chunk: video_path is just the URL (or kept generic)
            # audio_path is None (transcription must come from YouTube API)
            chunk_info = ChunkInfo(
                chunk_index=i,
                video_path=video_path, # URL
                audio_path="",         # None
                start_time=start_time,
                end_time=end_time,
                duration=actual_duration,
            )
            chunks.append(chunk_info)

        else:
            video_chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}.mp4")
            audio_chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}.wav")

            # Extract video chunk
            vid_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(actual_duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                video_chunk_path
            ]
            subprocess.run(vid_cmd, capture_output=True, text=True, check=True)

            # Extract audio chunk
            aud_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(actual_duration),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_chunk_path
            ]
            subprocess.run(aud_cmd, capture_output=True, text=True, check=True)

            chunks.append(ChunkInfo(
                chunk_index=i,
                video_path=video_chunk_path,
                audio_path=audio_chunk_path,
                start_time=start_time,
                end_time=end_time,
                duration=actual_duration,
            ))

        if progress_callback and not is_youtube:
            chunk_progress = 0.15 + (0.35 * (i + 1) / num_chunks)
            progress_callback(chunk_progress, f"Chunked {i + 1}/{num_chunks}")

    return chunks


def process_video_input(
    source_type: str,
    youtube_url: Optional[str],
    uploaded_file: Optional[str],
    chunk_duration_sec: int,
    work_dir: str,
    progress_callback=None,
    skip_download: bool = False,
) -> List[ChunkInfo]:
    """
    Main entry point. If skip_download=True and YouTube, just extracts metadata.
    """
    os.makedirs(work_dir, exist_ok=True)

    if source_type == "YouTube Link":
        if not youtube_url or not youtube_url.strip():
            raise ValueError("Please provide a YouTube URL")
        
        if skip_download:
            # Get metadata only
            if progress_callback:
                progress_callback(0.05, "Getting video info...")
            meta = get_youtube_video_metadata(youtube_url.strip())
            return chunk_video(
                youtube_url.strip(), 
                chunk_duration_sec, 
                work_dir, 
                progress_callback,
                is_youtube=True,
                duration=meta["duration"]
            )
        
        # Original download path (fallback)
        video_path = download_youtube(youtube_url.strip(), work_dir, progress_callback)
        
    elif source_type == "Upload Video":
        if not uploaded_file:
            raise ValueError("Please upload a video file")
        # Copy uploaded file to work dir
        ext = os.path.splitext(uploaded_file)[1] or ".mp4"
        video_path = os.path.join(work_dir, f"source_video{ext}")
        shutil.copy2(uploaded_file, video_path)
        if progress_callback:
            progress_callback(0.1, "Video loaded")
            
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # Standard local chunking
    return chunk_video(video_path, chunk_duration_sec, work_dir, progress_callback)

