"""
Audio-to-Text Extractor — Gradio UI Application
Extracts speech from YouTube videos or uploaded files, transcribes using Gemini.
"""

import os
import sys
import shutil
import tempfile
import gradio as gr

from video_processor import process_video_input, get_video_title
from transcriber import transcribe_all_chunks, check_transcript_availability, _get_video_id


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
TEMP_BASE = os.path.join(tempfile.gettempdir(), "audio_extractor")
MAX_CHUNKS = 30  # max pre-allocated UI slots

CUSTOM_CSS = """
#app-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
    margin-bottom: 0;
    padding-top: 10px;
}
#app-subtitle {
    text-align: center;
    color: #999;
    margin-top: 2px;
    font-size: 0.95em;
}
.chunk-group {
    border: 1px solid rgba(128,128,255,0.2);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    background: rgba(20, 20, 40, 0.3);
}
    background: rgba(20, 20, 40, 0.3);
}
.transcript-box textarea {
    font-size: 0.95em !important;
    line-height: 1.7 !important;
}
#sticky-player {
    position: sticky;
    top: 0;
    z-index: 100;
    padding-bottom: 10px;
    background-color: white; /* Ensure readability over content */
    margin-bottom: 20px;
    border-bottom: 1px solid #eee;
}
"""


def _clean_work_dir():
    """Remove previous temp files."""
    if os.path.exists(TEMP_BASE):
        shutil.rmtree(TEMP_BASE, ignore_errors=True)


# ---------------------------------------------------------------------------
# Processing Logic
# ---------------------------------------------------------------------------
def process_video(source_type, youtube_url, uploaded_file, chunk_duration, progress=gr.Progress()):
    """
    Main pipeline:
    1. Download / load video
    2. Chunk into segments
    3. Transcribe each chunk via Gemini or YouTube API
    4. Return results for UI display and export file
    """
    _clean_work_dir()
    work_dir = os.path.join(TEMP_BASE, "session")
    os.makedirs(work_dir, exist_ok=True)

    def progress_cb(value, desc):
        progress(value, desc=desc)

    # Step 1: Check YouTube transcript availability first
    skip_download = False
    if source_type == "YouTube Link" and youtube_url:
        vid = _get_video_id(youtube_url)
        if vid and check_transcript_availability(vid):
            skip_download = True
            progress_cb(0.1, "Found native transcript! Skipping download...")

    # Step 2: Process input (download or get metadata)
    try:
        chunks = process_video_input(
            source_type=source_type,
            youtube_url=youtube_url,
            uploaded_file=uploaded_file,
            chunk_duration_sec=int(chunk_duration),
            work_dir=work_dir,
            progress_callback=progress_cb,
            skip_download=skip_download,
        )
    except Exception as e:
        raise gr.Error(f"Video processing failed: {str(e)}")

    if not chunks:
        raise gr.Error("No chunks were created. The video may be too short.")

    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    # Step 3: Transcribe
    try:
        # Pass youtube_url if applicable
        yt_url = youtube_url if source_type == "YouTube Link" else None
        transcriptions = transcribe_all_chunks(
            chunks, 
            youtube_url=yt_url,
            progress_callback=progress_cb
        )
    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")

    progress(1.0, desc="Done!")

    # Step 4: Build output values
    # Step 4: Build output values
    trans_map = {t.chunk_index: t for t in transcriptions}
    
    # Generate full transcript text for export
    full_text_lines = []
    
    results = []
    for chunk in chunks:
        t = trans_map.get(chunk.chunk_index)
        text = t.text if t else "[No transcription]"
        lang = t.language if t else "Unknown"
        
        # Append to full transcript with timestamps
        full_text_lines.append(f"[{chunk.start_timestamp} - {chunk.end_timestamp}] ({lang})")
        full_text_lines.append(text)
        full_text_lines.append("-" * 40)
        
        
        if skip_download and source_type == "YouTube Link":
            # For YouTube, we use the Master Player approach now.
            # So individual chunk 'video' and 'html' are None.
            # We rely on metadata and 'type'='youtube'
            results.append({
                "type": "youtube",
                "video": None,
                "label": chunk.label,
                "language": lang,
                "text": text,
            })
        else:
            # Local video file
            results.append({
                "type": "local",
                "html": None,
                "video": chunk.video_path,
                "label": chunk.label,
                "language": lang,
                "text": text,
            })

    # Create metadata for export
    title = "transcript"
    try:
        if source_type == "YouTube Link" and youtube_url:
            title = get_video_title(youtube_url)
        elif uploaded_file:
            title = os.path.splitext(os.path.basename(uploaded_file))[0]
    except:
        pass
        
    export_meta = {
        "title": title,
        "chunks": [
            {
                "start": chunk.start_time,
                "end": chunk.end_time,
                "language": trans_map.get(chunk.chunk_index).language if trans_map.get(chunk.chunk_index) else "Unknown"
            }
            for chunk in chunks
        ]
    }

    return results, export_meta


def generate_export_file(meta, *text_inputs):
    if not meta:
        return None
        
    title = meta.get("title", "transcript")
    chunks = meta.get("chunks", [])
    
    lines = []
    for i, c in enumerate(chunks):
        if i >= len(text_inputs):
            break
        txt = text_inputs[i]
        lines.append(f"[{c['start']} - {c['end']}] ({c['language']})")
        lines.append(txt)
        lines.append("-" * 40)
        
    # Sanitize filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title[:100] or "transcript"
    
    # Create temp file
    # We place it in a temp dir so Gradio can serve it
    work_dir = os.path.join(TEMP_BASE, "export")
    os.makedirs(work_dir, exist_ok=True)
    export_filename = f"{safe_title}_edited.txt"
    export_path = os.path.join(work_dir, export_filename)
        
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    return export_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def create_ui():
    with gr.Blocks(
        title="Audio-to-Text Extractor",
    ) as app:

        # Header
        gr.HTML(
            '<p id="app-title">🎙️ Audio-to-Text Extractor</p>'
            '<p id="app-subtitle">Multilingual video transcription powered by Gemini&nbsp;·&nbsp;'
            'Supports Kannada, Hindi, English &amp; 100+ languages</p>'
        )

        with gr.Row():
            # ---- Left: Configuration ----
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### ⚙️ Configuration")

                source_type = gr.Radio(
                    choices=["YouTube Link", "Upload Video"],
                    value="YouTube Link",
                    label="Video Source",
                )

                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    visible=True,
                )

                uploaded_file = gr.File(
                    label="Upload Video File",
                    file_types=["video"],
                    visible=False,
                )

                chunk_duration = gr.Slider(
                    minimum=15, maximum=600, value=60, step=15,
                    label="Chunk Duration (seconds)",
                    info="Each chunk will be independently transcribed",
                )

                process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")

                gr.Markdown(
                    "---\n"
                    "**Tips:**\n"
                    "- 30-60s chunks → more granular text\n"
                    "- 120-300s chunks → fewer API calls\n"
                    "- Language is auto-detected per chunk\n"
                    f"- Max {MAX_CHUNKS} chunks per run"
                )

            # ---- Right: Results ----
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Transcription Results")

                
                # Master Player (Sticky)
                master_yt_player = gr.HTML(visible=False, elem_id="sticky-player")
                
                vid_id_state = gr.State("")
                chunk_starts_state = gr.State([])
                
                export_meta_state = gr.State({})
                
                # Export Download Area
                with gr.Row():
                    export_btn = gr.Button("💾 Export Edited Transcript", variant="primary")
                    export_file = gr.File(label="Download", visible=False, interactive=False)
                
                status_md = gr.Markdown("*Paste a YouTube link or upload a video to get started.*")

                # Pre-allocate chunk display slots
                chunk_ui = []
                for i in range(MAX_CHUNKS):
                    with gr.Group(visible=False, elem_classes=["chunk-group"]) as grp:
                        lbl = gr.Markdown(f"**Chunk {i+1}**")
                        with gr.Row():
                            # Play button for YoutSube segments
                            play_btn = gr.Button("▶ Play Segment", variant="secondary", size="sm", visible=False)
                            
                            # Local Video Chunk Player
                            vid = gr.Video(label=None, show_label=False, height=260, scale=1, visible=False)
                            
                            with gr.Column(scale=1):
                                lang = gr.Textbox(label="Language", interactive=False, max_lines=1)
                                txt = gr.Textbox(
                                    label="Transcription",
                                    interactive=True, lines=7,
                                    elem_classes=["transcript-box"],
                                )
                    chunk_ui.append({"grp": grp, "lbl": lbl, "vid": vid, "play": play_btn, "lang": lang, "txt": txt})

        # ---- Interaction Logic ----
        def toggle_source(choice):
            return (
                gr.update(visible=choice == "YouTube Link"),
                gr.update(visible=choice == "Upload Video"),
            )

        source_type.change(toggle_source, [source_type], [youtube_url, uploaded_file])

        def on_process(src, url, ufile, cdur, progress=gr.Progress()):
            results, meta = process_video(src, url, ufile, cdur, progress)
            n = len(results)
            
            # Prepare Master Player content if YouTube
            is_yt_source = (src == "YouTube Link")
            master_html = None
            vid_id = ""
            chunk_starts = []
            
            if is_yt_source and url:
                vid_id = _get_video_id(url)
                if vid_id:
                    # Initialize with start of video (or first chunk)
                    master_html = (
                        f'<iframe width="100%" height="360" '
                        f'src="https://www.youtube.com/embed/{vid_id}?autoplay=0" '
                        f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                        f'allowfullscreen></iframe>'
                    )
                chunk_starts = [c['start'] for c in meta.get('chunks', [])]

            outputs = [
                gr.update(value=f"✅ **{n} chunk(s)** transcribed successfully!"),
                meta, # Update state
                gr.update(visible=False), # Hide previous download button
                gr.update(value=vid_id), # vid_id_state
                gr.update(value=chunk_starts), # chunk_starts_state
                gr.update(value=master_html, visible=True) if master_html else gr.update(visible=False) # master player
            ]
            
            for i in range(MAX_CHUNKS):
                if i < n:
                    r = results[i]
                    is_yt = (r["type"] == "youtube")
                    outputs += [
                        gr.update(visible=True),                # group
                        gr.update(value=f"**{r['label']}**"),   # label
                        gr.update(visible=is_yt),               # play_btn
                        gr.update(visible=not is_yt, value=r["video"]), # video
                        gr.update(value=r["language"]),         # lang
                        gr.update(value=r["text"]),             # text
                    ]
                else:
                    outputs += [
                        gr.update(visible=False),
                        gr.update(),
                        gr.update(visible=False),
                        gr.update(visible=False, value=None),
                        gr.update(value=""),
                        gr.update(value=""),
                    ]
            return outputs

        all_outputs = [status_md, export_meta_state, export_file, vid_id_state, chunk_starts_state, master_yt_player]
        text_inputs = []
        for i, c in enumerate(chunk_ui):
            all_outputs += [c["grp"], c["lbl"], c["play"], c["vid"], c["lang"], c["txt"]]
            text_inputs.append(c["txt"])
            
            # Wire up Play Button for this chunk
            def on_play_segment(idx, vid_id, starts):
                if not vid_id or idx >= len(starts):
                    return gr.update()
                
                start = int(starts[idx])
                # We can also set end time if desired, but user might want to continue watching
                # Let's set end to next chunk start just in case, or leave open?
                # User asked to "go to that exact chunk". Usually implies playing that chunk.
                # Let's try just start.
                return (
                    f'<iframe width="100%" height="360" '
                    f'src="https://www.youtube.com/embed/{vid_id}?start={start}&autoplay=1" '
                    f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                    f'allowfullscreen></iframe>'
                )

            # Important: capture i=i in default arg or use partial, but Gradio event definition is immediate.
            # We must use a wrapper or functools.partial to capture 'i'
            # Or simpler: define logic that takes index state? But index is static per button?
            # Actually, just passing `gr.State(i)` works perfectly as input.
            c["play"].click(
                fn=on_play_segment,
                inputs=[gr.State(i), vid_id_state, chunk_starts_state],
                outputs=[master_yt_player]
            )

        process_btn.click(
            fn=on_process,
            inputs=[source_type, youtube_url, uploaded_file, chunk_duration],
            outputs=all_outputs,
        )
        
        # Export Logic
        def on_export(meta, *texts):
            path = generate_export_file(meta, *texts)
            return gr.update(value=path, visible=True)
            
        export_btn.click(
            fn=on_export,
            inputs=[export_meta_state] + text_inputs,
            outputs=[export_file]
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  🎙️  Audio-to-Text Extractor")
    print("  Powered by Gemini 2.0 Flash")
    print("=" * 60)

    if shutil.which("ffmpeg") is None:
        print("\n❌ ERROR: ffmpeg is not installed or not on PATH.")
        print("   Install with: sudo apt install ffmpeg")
        sys.exit(1)

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink"),
        css=CUSTOM_CSS,
    )
