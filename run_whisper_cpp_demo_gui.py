import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
CONFIG_EXAMPLE_PATH = SCRIPT_DIR / "config.example.json"


def load_config() -> dict:
    """Load configuration from config.json file."""
    if not CONFIG_PATH.exists():
        if CONFIG_EXAMPLE_PATH.exists():
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}\n"
                f"'{CONFIG_EXAMPLE_PATH.name}'을 '{CONFIG_PATH.name}'으로 복사한 후 경로를 수정하세요."
            )
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["whisper_cli", "model_path"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"설정 파일에 '{key}' 항목이 필요합니다.")

    return config


_config = load_config()
WHISPER_CLI = Path(_config["whisper_cli"])
MODEL_PATH = Path(_config["model_path"])
SUPPORTED_TYPES = ("wav", "mp3", "mp4", "m4a")
YOUTUBE_DOWNLOAD_DIR = Path(_config.get("youtube_download_dir", SCRIPT_DIR / "downloaded"))


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device index (or comma-separated list) to expose to whisper-cli.",
    )
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args


def save_uploaded_file(uploaded_file) -> Path:
    YOUTUBE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = YOUTUBE_DOWNLOAD_DIR / uploaded_file.name
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def prepare_audio_file(input_path: Path) -> Path:
    """
    WAV가 아닌 오디오 파일을 WAV로 변환.
    이미 WAV 파일이면 그대로 반환.
    """
    suffix = input_path.suffix.lower()
    if suffix not in {".wav", ".mp3", ".mp4", ".m4a"}:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    # WAV 파일이면 변환 없이 반환
    if suffix == ".wav":
        return input_path

    # 변환된 WAV 파일을 입력 파일과 같은 디렉토리에 저장
    output_path = input_path.with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg 명령을 찾을 수 없습니다. 서버 환경에 설치되어 있는지 확인하세요."
        ) from exc

    if result.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        msg = stderr or stdout or "ffmpeg 변환에 실패했습니다."
        raise RuntimeError(f"ffmpeg 변환 실패: {msg}")

    return output_path


def download_youtube_audio(url: str) -> Path:
    """
    Download audio from YouTube as wav using yt-dlp CLI and return local file path.
    """
    import uuid

    YOUTUBE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # UUID로 파일명 생성
    output_filename = f"youtube_{uuid.uuid4().hex[:8]}.wav"
    output_path = YOUTUBE_DOWNLOAD_DIR / output_filename

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        str(output_path),
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp 명령을 찾을 수 없습니다. 서버 환경에 설치되어 있는지 확인하세요."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            f"yt-dlp 다운로드 실패: {stderr or stdout or '알 수 없는 오류가 발생했습니다.'}"
        )

    if not output_path.exists():
        raise RuntimeError("yt-dlp 다운로드에 성공했지만 wav 파일을 찾지 못했습니다.")

    return output_path


def build_command(audio_path: Path) -> list:
    cmd = [
        str(WHISPER_CLI),
        "-m",
        str(MODEL_PATH),
        "-f",
        str(audio_path),
        "-l",
        "ko",
        "-otxt"
    ]

    return cmd


def run_whisper_cli(
    audio_path: Path,
    cuda_device: str,
) -> Tuple[str, Optional[str]]:
    """
    Run whisper-cli and return transcription text from the generated txt file.
    """

    cmd = build_command(audio_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", env=env)
    
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        error_msg = stderr or stdout or "whisper-cli failed with an unknown error."
        return "", error_msg

    # -otxt 플래그로 생성된 txt 파일 경로: {audio_path}.txt
    txt_path = Path(str(audio_path) + ".txt")

    if not txt_path.exists():
        return "", f"전사 결과 파일을 찾을 수 없습니다: {txt_path}"

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        transcription = f.read().strip()

    return transcription, None


def main() -> None:
    args = parse_cli_args()
    st.set_page_config(page_title="Metabuild STT Demo", layout="wide")

    st.title("Metabuild STT 데모")

    # Initialize session state so widgets don't conflict with later updates.
    st.session_state.setdefault("transcription", "")
    st.session_state.setdefault("last_error", "")

    left, right = st.columns(2)
    with left:
        st.subheader("오디오 입력")
        input_mode = st.radio(
            "입력 방식 선택", ("파일 업로드", "YouTube URL"), horizontal=True
        )

        uploaded = None
        youtube_url = ""
        if input_mode == "파일 업로드":
            uploaded = st.file_uploader(
                f"로컬 오디오/동영상 파일 선택 ({'/'.join(SUPPORTED_TYPES)})",
                type=list(SUPPORTED_TYPES),
            )
        else:
            youtube_url = st.text_input(
                "YouTube 링크 입력", placeholder="https://youtu.be/..."
            )

    with right:
        st.subheader("전사 결과")
        transcription_text = st.session_state["transcription"] if st.session_state["transcription"] else "대기 중..."
        st.code(transcription_text, language=None)
        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])

    start_button = st.button("전사 시작", type="primary", use_container_width=True)

    if start_button:
        # 이전 결과 초기화
        st.session_state["transcription"] = ""
        st.session_state["last_error"] = ""

        # 입력 검증
        if input_mode == "파일 업로드" and not uploaded:
            st.session_state["last_error"] = "먼저 파일을 업로드하세요."
        elif input_mode == "YouTube URL" and not youtube_url.strip():
            st.session_state["last_error"] = "먼저 YouTube 링크를 입력하세요."
        else:
            # 전사 수행
            temp_paths = set()
            try:
                if input_mode == "파일 업로드":
                    source_path = save_uploaded_file(uploaded)
                else:
                    with st.spinner("YouTube 오디오 다운로드 중..."):
                        source_path = download_youtube_audio(youtube_url.strip())

                with st.spinner("오디오 준비 중..."):
                    prepared_audio_path = prepare_audio_file(source_path)

                with st.spinner("Metabuild STT 실행 중..."):
                    transcription, error = run_whisper_cli(
                        prepared_audio_path, args.cuda_device
                    )

                if error:
                    st.session_state["last_error"] = error
                else:
                    st.session_state["transcription"] = str(transcription)
            except Exception as exc:
                st.session_state["last_error"] = str(exc)
            finally:
                for path in temp_paths:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

        # 결과 표시를 위해 페이지 새로고침
        rerun_fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
        if rerun_fn:
            rerun_fn()


if __name__ == "__main__":
    main()
