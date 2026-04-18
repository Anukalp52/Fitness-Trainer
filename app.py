import math
import importlib
import logging
import os
import tempfile
import sys
import threading
import time
import base64
from urllib.parse import quote_plus
from typing import Dict, List, Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# `streamlit-webrtc` runs media processing on worker threads where Streamlit
# context is intentionally unavailable. Silence that known warning noise.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


def load_mediapipe_pose():
    if hasattr(mp, "solutions"):
        solutions = mp.solutions
        return (
            solutions.drawing_utils,
            solutions.pose.POSE_CONNECTIONS,
            solutions.pose.Pose,
        )

    try:
        drawing_utils = importlib.import_module("mediapipe.python.solutions.drawing_utils")
        pose_module = importlib.import_module("mediapipe.python.solutions.pose")
        return drawing_utils, pose_module.POSE_CONNECTIONS, pose_module.Pose
    except ModuleNotFoundError as exc:
        python_path = sys.executable
        raise RuntimeError(
            "MediaPipe is installed, but the current interpreter cannot load its pose modules. "
            f"Run the app and install packages with the same Python executable: {python_path}\n"
            f"Try:\n{python_path} -m pip install mediapipe\n"
            f"{python_path} -m streamlit run app.py"
        ) from exc


mp_drawing, POSE_CONNECTIONS, Pose = load_mediapipe_pose()


st.set_page_config(
    page_title="AI Fitness Trainer",
    page_icon="AI",
    layout="wide",
)

LANGUAGE_OPTIONS = {
    "en": "English",
    "hi": "à¤¹à¤¿à¤‚à¤¦à¥€",
}

TRANSLATIONS = {
    "app_name": {"en": "AI Fitness Trainer", "hi": "à¤à¤†à¤ˆ à¤«à¤¿à¤Ÿà¤¨à¥‡à¤¸ à¤Ÿà¥à¤°à¥‡à¤¨à¤°"},
    "hero_badge": {"en": "Smart Movement Guidance", "hi": "à¤¸à¥à¤®à¤¾à¤°à¥à¤Ÿ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤—à¤¾à¤‡à¤¡à¥‡à¤‚à¤¸"},
    "hero_title": {"en": "Fitness Coach", "hi": "à¤«à¤¿à¤Ÿà¤¨à¥‡à¤¸ à¤•à¥‹à¤š"},
    "hero_text": {
        "en": "Build better form with live posture feedback, guided exercise references, and upload-based review for gym training and yoga practice.",
        "hi": "à¤²à¤¾à¤‡à¤µ à¤ªà¥‹à¤¶à¥à¤šà¤° à¤«à¥€à¤¡à¤¬à¥ˆà¤•, à¤—à¤¾à¤‡à¤¡à¥‡à¤¡ à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ à¤°à¥‡à¤«à¤°à¥‡à¤‚à¤¸, à¤”à¤° à¤…à¤ªà¤²à¥‹à¤¡ à¤°à¤¿à¤µà¥à¤¯à¥‚ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤œà¤¿à¤® à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤”à¤° à¤¯à¥‹à¤— à¤…à¤­à¥à¤¯à¤¾à¤¸ à¤•à¥‹ à¤¬à¥‡à¤¹à¤¤à¤° à¤¬à¤¨à¤¾à¤à¤‚à¥¤",
    },
    "selected_exercise": {"en": "Selected Exercise", "hi": "à¤šà¤¯à¤¨à¤¿à¤¤ à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ"},
    "training_library": {"en": "Training Library", "hi": "à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤²à¤¾à¤‡à¤¬à¥à¤°à¥‡à¤°à¥€"},
    "active_category": {"en": "Active Category", "hi": "à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤¶à¥à¤°à¥‡à¤£à¥€"},
    "gym_count": {"en": "20 Gym + 20 Asana", "hi": "20 à¤œà¤¿à¤® + 20 à¤†à¤¸à¤¨"},
    "gym": {"en": "Gym", "hi": "à¤œà¤¿à¤®"},
    "asana": {"en": "Asana", "hi": "à¤†à¤¸à¤¨"},
    "movement_library": {"en": "Movement Library", "hi": "à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤²à¤¾à¤‡à¤¬à¥à¤°à¥‡à¤°à¥€"},
    "exercise_reference_title": {"en": "Exercise reference and setup guide", "hi": "à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ à¤°à¥‡à¤«à¤°à¥‡à¤‚à¤¸ à¤”à¤° à¤¸à¥‡à¤Ÿà¤…à¤ª à¤—à¤¾à¤‡à¤¡"},
    "exercise_reference_copy": {
        "en": "Review the selected movement before training so the camera angle, setup, and key technique cues are clear.",
        "hi": "à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤šà¤¯à¤¨à¤¿à¤¤ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤•à¥‹ à¤¦à¥‡à¤–à¥‡à¤‚ à¤¤à¤¾à¤•à¤¿ à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤², à¤¸à¥‡à¤Ÿà¤…à¤ª à¤”à¤° à¤®à¥à¤–à¥à¤¯ à¤¤à¤•à¤¨à¥€à¤• à¤¸à¤‚à¤•à¥‡à¤¤ à¤¸à¥à¤ªà¤·à¥à¤Ÿ à¤°à¤¹à¥‡à¤‚à¥¤",
    },
    "live_coach": {"en": "Live Coach", "hi": "à¤²à¤¾à¤‡à¤µ à¤•à¥‹à¤š"},
    "webcam_posture": {"en": "Webcam posture stream", "hi": "à¤µà¥‡à¤¬à¤•à¥ˆà¤® à¤ªà¥‹à¤¶à¥à¤šà¤° à¤¸à¥à¤Ÿà¥à¤°à¥€à¤®"},
    "webcam_posture_copy": {
        "en": "Use the live camera feed for real-time pose feedback, then switch to uploaded media for slower review if needed.",
        "hi": "à¤°à¤¿à¤¯à¤²-à¤Ÿà¤¾à¤‡à¤® à¤ªà¥‹à¥› à¤«à¥€à¤¡à¤¬à¥ˆà¤• à¤•à¥‡ à¤²à¤¿à¤ à¤²à¤¾à¤‡à¤µ à¤•à¥ˆà¤®à¤°à¤¾ à¤«à¥€à¤¡ à¤•à¤¾ à¤‰à¤ªà¤¯à¥‹à¤— à¤•à¤°à¥‡à¤‚, à¤”à¤° à¤œà¤°à¥‚à¤°à¤¤ à¤¹à¥‹à¤¨à¥‡ à¤ªà¤° à¤…à¤ªà¤²à¥‹à¤¡ à¤®à¥€à¤¡à¤¿à¤¯à¤¾ à¤¸à¥‡ à¤§à¥€à¤®à¤¾ à¤°à¤¿à¤µà¥à¤¯à¥‚ à¤•à¤°à¥‡à¤‚à¥¤",
    },
    "upload_analysis": {"en": "Upload Analysis", "hi": "à¤…à¤ªà¤²à¥‹à¤¡ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£"},
    "photo_video_review": {"en": "Photo or video review", "hi": "à¤«à¥‹à¤Ÿà¥‹ à¤¯à¤¾ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤°à¤¿à¤µà¥à¤¯à¥‚"},
    "photo_video_review_copy": {
        "en": "Upload a workout photo for a quick snapshot or a video for frame-by-frame posture review with downloadable output.",
        "hi": "à¤¤à¥‡à¤œà¤¼ à¤¸à¥à¤¨à¥ˆà¤ªà¤¶à¥‰à¤Ÿ à¤•à¥‡ à¤²à¤¿à¤ à¤µà¤°à¥à¤•à¤†à¤‰à¤Ÿ à¤«à¥‹à¤Ÿà¥‹ à¤¯à¤¾ à¤«à¥à¤°à¥‡à¤®-à¤¦à¤°-à¤«à¥à¤°à¥‡à¤® à¤ªà¥‹à¤¶à¥à¤šà¤° à¤°à¤¿à¤µà¥à¤¯à¥‚ à¤•à¥‡ à¤²à¤¿à¤ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚à¥¤",
    },
    "photo_upload": {"en": "Photo Upload", "hi": "à¤«à¥‹à¤Ÿà¥‹ à¤…à¤ªà¤²à¥‹à¤¡"},
    "video_upload": {"en": "Video Upload", "hi": "à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤…à¤ªà¤²à¥‹à¤¡"},
    "upload_workout_photo": {"en": "Upload a workout photo", "hi": "à¤µà¤°à¥à¤•à¤†à¤‰à¤Ÿ à¤«à¥‹à¤Ÿà¥‹ à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚"},
    "upload_workout_video": {"en": "Upload a workout video", "hi": "à¤µà¤°à¥à¤•à¤†à¤‰à¤Ÿ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚"},
    "session_insights": {"en": "Session Insights", "hi": "à¤¸à¥‡à¤¶à¤¨ à¤‡à¤¨à¤¸à¤¾à¤‡à¤Ÿà¥à¤¸"},
    "live_metrics_title": {"en": "Live metrics and coaching", "hi": "à¤²à¤¾à¤‡à¤µ à¤®à¥‡à¤Ÿà¥à¤°à¤¿à¤•à¥à¤¸ à¤”à¤° à¤•à¥‹à¤šà¤¿à¤‚à¤—"},
    "live_metrics_copy": {
        "en": "Monitor your current form quality, hold time, and coaching prompts while you train.",
        "hi": "à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤•à¥‡ à¤¦à¥Œà¤°à¤¾à¤¨ à¤…à¤ªà¤¨à¥‡ à¤«à¥‰à¤°à¥à¤®, à¤¹à¥‹à¤²à¥à¤¡ à¤Ÿà¤¾à¤‡à¤® à¤”à¤° à¤•à¥‹à¤šà¤¿à¤‚à¤— à¤¸à¤‚à¤•à¥‡à¤¤à¥‹à¤‚ à¤ªà¤° à¤¨à¤œà¤¼à¤° à¤°à¤–à¥‡à¤‚à¥¤",
    },
    "session_state": {"en": "Session State", "hi": "à¤¸à¥‡à¤¶à¤¨ à¤¸à¥à¤¥à¤¿à¤¤à¤¿"},
    "exercise": {"en": "Exercise", "hi": "à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ"},
    "reps": {"en": "Reps", "hi": "à¤°à¥‡à¤ªà¥à¤¸"},
    "form_score": {"en": "Form Score", "hi": "à¤«à¥‰à¤°à¥à¤® à¤¸à¥à¤•à¥‹à¤°"},
    "hold_time": {"en": "Hold Time", "hi": "à¤¹à¥‹à¤²à¥à¤¡ à¤Ÿà¤¾à¤‡à¤®"},
    "feedback": {"en": "Feedback", "hi": "à¤«à¥€à¤¡à¤¬à¥ˆà¤•"},
    "exercise_guide": {"en": "Exercise Guide", "hi": "à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ à¤—à¤¾à¤‡à¤¡"},
    "how_to_perform": {"en": "How To Perform", "hi": "à¤•à¥ˆà¤¸à¥‡ à¤•à¤°à¥‡à¤‚"},
    "step_by_step": {"en": "Step by step", "hi": "à¤¸à¥à¤Ÿà¥‡à¤ª à¤¬à¤¾à¤¯ à¤¸à¥à¤Ÿà¥‡à¤ª"},
    "why_it_helps": {"en": "Why It Helps", "hi": "à¤¯à¤¹ à¤•à¥à¤¯à¥‹à¤‚ à¤‰à¤ªà¤¯à¥‹à¤—à¥€ à¤¹à¥ˆ"},
    "benefits": {"en": "Benefits", "hi": "à¤«à¤¾à¤¯à¤¦à¥‡"},
    "trainer_care": {"en": "Trainer Care", "hi": "à¤Ÿà¥à¤°à¥‡à¤¨à¤° à¤•à¥‡à¤¯à¤°"},
    "trainer_care_title": {"en": "Train safely and consistently", "hi": "à¤¸à¥à¤°à¤•à¥à¤·à¤¿à¤¤ à¤”à¤° à¤¨à¤¿à¤¯à¤®à¤¿à¤¤ à¤°à¥‚à¤ª à¤¸à¥‡ à¤Ÿà¥à¤°à¥‡à¤¨ à¤•à¤°à¥‡à¤‚"},
    "trainer_care_copy": {
        "en": "Use the coach as a support tool, but pace your workout like a real training session: prepare first, move with control, and recover between efforts.",
        "hi": "à¤•à¥‹à¤š à¤•à¥‹ à¤¸à¤ªà¥‹à¤°à¥à¤Ÿ à¤Ÿà¥‚à¤² à¤•à¥€ à¤¤à¤°à¤¹ à¤‰à¤ªà¤¯à¥‹à¤— à¤•à¤°à¥‡à¤‚, à¤²à¥‡à¤•à¤¿à¤¨ à¤µà¤°à¥à¤•à¤†à¤‰à¤Ÿ à¤•à¥‹ à¤…à¤¸à¤²à¥€ à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤¸à¥‡à¤¶à¤¨ à¤•à¥€ à¤¤à¤°à¤¹ à¤•à¤°à¥‡à¤‚: à¤ªà¤¹à¤²à¥‡ à¤¤à¥ˆà¤¯à¤¾à¤°à¥€ à¤•à¤°à¥‡à¤‚, à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤®à¥‚à¤µ à¤•à¤°à¥‡à¤‚, à¤”à¤° à¤¬à¥€à¤š à¤®à¥‡à¤‚ à¤°à¤¿à¤•à¤µà¤°à¥€ à¤²à¥‡à¤‚à¥¤",
    },
    "coach_settings": {"en": "Coach Settings", "hi": "à¤•à¥‹à¤š à¤¸à¥‡à¤Ÿà¤¿à¤‚à¤—à¥à¤¸"},
    "language": {"en": "Language", "hi": "à¤­à¤¾à¤·à¤¾"},
    "category": {"en": "Category", "hi": "à¤¶à¥à¤°à¥‡à¤£à¥€"},
    "setup_tips": {"en": "Setup Tips", "hi": "à¤¸à¥‡à¤Ÿà¤…à¤ª à¤Ÿà¤¿à¤ªà¥à¤¸"},
    "notes": {"en": "Notes", "hi": "à¤¨à¥‹à¤Ÿà¥à¤¸"},
    "gym exercises": {"en": "Gym Exercises", "hi": "à¤œà¤¿à¤® à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ"},
    "asana exercises": {"en": "Asana Exercises", "hi": "à¤†à¤¸à¤¨ à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ"},
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get("en", key))


EXERCISE_HI = {
    "squat": {"label": "à¤¸à¥à¤•à¥à¤µà¤¾à¤Ÿ", "title": "à¤¸à¥à¤•à¥à¤µà¤¾à¤Ÿ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤›à¤¾à¤¤à¥€ à¤•à¥‹ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤à¤¡à¤¼à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤¯à¤¦à¤¿ à¤¸à¤‚à¤­à¤µ à¤¹à¥‹ à¤¤à¥‹ à¤œà¤¾à¤‚à¤˜à¥‡à¤‚ à¤¸à¤®à¤¾à¤¨à¤¾à¤‚à¤¤à¤° à¤•à¥‡ à¤•à¤°à¥€à¤¬ à¤†à¤¨à¥‡ à¤¤à¤• à¤¨à¥€à¤šà¥‡ à¤œà¤¾à¤à¤à¥¤", "à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤ªà¤‚à¤œà¥‹à¤‚ à¤•à¥€ à¤¦à¤¿à¤¶à¤¾ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "pushup": {"label": "à¤ªà¥à¤¶-à¤…à¤ª", "title": "à¤ªà¥à¤¶-à¤…à¤ª à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¬à¥‡à¤¹à¤¤à¤° à¤à¤²à¥à¤¬à¥‹ à¤Ÿà¥à¤°à¥ˆà¤•à¤¿à¤‚à¤— à¤•à¥‡ à¤²à¤¿à¤ à¤•à¥ˆà¤®à¤°à¤¾ à¤¸à¤¾à¤‡à¤¡ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¤‚à¤§à¥‡, à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤”à¤° à¤Ÿà¤–à¤¨à¥‡ à¤à¤• à¤¸à¥€à¤§ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¥‹à¤¹à¤¨à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¬à¤¹à¥à¤¤ à¤œà¤¼à¥à¤¯à¤¾à¤¦à¤¾ à¤¬à¤¾à¤¹à¤° à¤¨ à¤«à¥ˆà¤²à¤¾à¤à¤à¥¤"]},
    "plank": {"label": "à¤ªà¥à¤²à¥ˆà¤‚à¤•", "title": "à¤ªà¥à¤²à¥ˆà¤‚à¤• à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¤•à¤° à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤—à¥à¤²à¥‚à¤Ÿà¥à¤¸ à¤•à¥‹ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤•à¤°à¥‡à¤‚à¥¤", "à¤—à¤°à¥à¤¦à¤¨ à¤•à¥‹ à¤¨à¥à¤¯à¥‚à¤Ÿà¥à¤°à¤² à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤¹à¤²à¥à¤•à¤¾ à¤¨à¥€à¤šà¥‡ à¤¦à¥‡à¤–à¥‡à¤‚à¥¤", "à¤•à¤‚à¤§à¥‹à¤‚ à¤¸à¥‡ à¤Ÿà¤–à¤¨à¥‹à¤‚ à¤¤à¤• à¤¸à¥€à¤§à¥€ à¤²à¤¾à¤‡à¤¨ à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "lunge": {"label": "à¤²à¤‚à¤œ", "title": "à¤²à¤‚à¤œ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¸à¤¾à¤®à¤¨à¥‡ à¤µà¤¾à¤²à¥‡ à¤˜à¥à¤Ÿà¤¨à¥‡ à¤•à¥‹ à¤Ÿà¤–à¤¨à¥‡ à¤•à¥‡ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤†à¤—à¥‡ à¤à¥à¤•à¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤¸à¥€à¤§à¤¾ à¤¨à¥€à¤šà¥‡ à¤œà¤¾à¤à¤à¥¤", "à¤›à¤¾à¤¤à¥€ à¤”à¤° à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤"]},
    "deadlift": {"label": "à¤¡à¥‡à¤¡à¤²à¤¿à¤«à¥à¤Ÿ", "title": "à¤¡à¥‡à¤¡à¤²à¤¿à¤«à¥à¤Ÿ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤°à¥€à¤¢à¤¼ à¤¸à¥€à¤§à¥€ à¤°à¤–à¤¤à¥‡ à¤¹à¥à¤ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤¸à¥‡ à¤¹à¤¿à¤‚à¤— à¤•à¤°à¥‡à¤‚à¥¤", "à¤µà¤œà¤¼à¤¨ à¤•à¥‹ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‡ à¤ªà¤¾à¤¸ à¤°à¤–à¥‡à¤‚à¥¤", "à¤¹à¤° à¤°à¥‡à¤ª à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¥‡à¤‚à¥¤"]},
    "shoulder_press": {"label": "à¤¶à¥‹à¤²à¥à¤¡à¤° à¤ªà¥à¤°à¥‡à¤¸", "title": "à¤¶à¥‹à¤²à¥à¤¡à¤° à¤ªà¥à¤°à¥‡à¤¸ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤°à¤¿à¤¬à¥à¤¸ à¤•à¥‹ à¤¨à¥€à¤šà¥‡ à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤•à¤®à¤° à¤•à¥‹ à¤œà¤¼à¥à¤¯à¤¾à¤¦à¤¾ à¤†à¤°à¥à¤š à¤¨ à¤•à¤°à¥‡à¤‚à¥¤", "à¤ªà¥à¤°à¥‡à¤¸ à¤•à¥‹ à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‡ à¤Šà¤ªà¤° à¤¸à¥€à¤§à¥€ à¤²à¤¾à¤‡à¤¨ à¤®à¥‡à¤‚ à¤•à¤°à¥‡à¤‚à¥¤", "à¤¦à¥‹à¤¨à¥‹à¤‚ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤ªà¤° à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "bicep_curl": {"label": "à¤¬à¤¾à¤‡à¤¸à¥‡à¤ª à¤•à¤°à¥à¤²", "title": "à¤¬à¤¾à¤‡à¤¸à¥‡à¤ª à¤•à¤°à¥à¤² à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¥‹à¤¹à¤¨à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¶à¤°à¥€à¤° à¤•à¥‡ à¤ªà¤¾à¤¸ à¤°à¤–à¥‡à¤‚à¥¤", "à¤µà¤œà¤¼à¤¨ à¤‰à¤ à¤¾à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤à¥à¤²à¤¾à¤à¤ à¤¨à¤¹à¥€à¤‚à¥¤", "à¤¨à¥€à¤šà¥‡ à¤²à¤¾à¤¨à¥‡ à¤•à¥€ à¤—à¤¤à¤¿ à¤•à¥‹ à¤§à¥€à¤°à¥‡ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤•à¤°à¥‡à¤‚à¥¤"]},
    "tricep_dip": {"label": "à¤Ÿà¥à¤°à¤¾à¤‡à¤¸à¥‡à¤ª à¤¡à¤¿à¤ª", "title": "à¤Ÿà¥à¤°à¤¾à¤‡à¤¸à¥‡à¤ª à¤¡à¤¿à¤ª à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‹ à¤•à¤¾à¤¨à¥‹à¤‚ à¤¸à¥‡ à¤¦à¥‚à¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¥‹à¤¹à¤¨à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¬à¤¾à¤¹à¤° à¤«à¥ˆà¤²à¤¾à¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤ªà¥€à¤›à¥‡ à¤®à¥‹à¤¡à¤¼à¥‡à¤‚à¥¤", "à¤¦à¤°à¥à¤¦-à¤°à¤¹à¤¿à¤¤ à¤°à¥‡à¤‚à¤œ à¤®à¥‡à¤‚ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤®à¥‚à¤µ à¤•à¤°à¥‡à¤‚à¥¤"]},
    "pullup": {"label": "à¤ªà¥à¤²-à¤…à¤ª", "title": "à¤ªà¥à¤²-à¤…à¤ª à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¥‹à¤° à¤•à¥‹ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤°à¤–à¤¤à¥‡ à¤¹à¥à¤ à¤¸à¥à¤¥à¤¿à¤° à¤¹à¥ˆà¤‚à¤— à¤¸à¥‡ à¤¶à¥à¤°à¥‚ à¤•à¤°à¥‡à¤‚à¥¤", "à¤•à¤‚à¤§à¥‡ à¤šà¤¢à¤¼à¤¾à¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤•à¥‹à¤¹à¤¨à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¨à¥€à¤šà¥‡ à¤–à¥€à¤‚à¤šà¥‡à¤‚à¥¤", "à¤°à¥‡à¤ªà¥à¤¸ à¤•à¥‡ à¤¬à¥€à¤š à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤à¥‚à¤²à¤¨à¥‡ à¤¨ à¤¦à¥‡à¤‚à¥¤"]},
    "bench_press": {"label": "à¤¬à¥‡à¤‚à¤š à¤ªà¥à¤°à¥‡à¤¸", "title": "à¤¬à¥‡à¤‚à¤š à¤ªà¥à¤°à¥‡à¤¸ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¤²à¤¾à¤ˆ à¤•à¥‹ à¤•à¥‹à¤¹à¤¨à¥€ à¤•à¥‡ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤ªà¥à¤°à¥‡à¤¸ à¤•à¥‡ à¤¦à¥Œà¤°à¤¾à¤¨ à¤Šà¤ªà¤°à¥€ à¤ªà¥€à¤  à¤®à¥‡à¤‚ à¤¤à¤¨à¤¾à¤µ à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤", "à¤¦à¥‹à¤¨à¥‹à¤‚ à¤¤à¤°à¤« à¤¸à¤®à¤¾à¤¨ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤¬à¤¾à¤° à¤ªà¥à¤°à¥‡à¤¸ à¤•à¤°à¥‡à¤‚à¥¤"]},
    "mountain_climber": {"label": "à¤®à¤¾à¤‰à¤‚à¤Ÿà¥‡à¤¨ à¤•à¥à¤²à¤¾à¤‡à¤‚à¤¬à¤°", "title": "à¤®à¤¾à¤‰à¤‚à¤Ÿà¥‡à¤¨ à¤•à¥à¤²à¤¾à¤‡à¤‚à¤¬à¤° à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‹ à¤¹à¤¥à¥‡à¤²à¤¿à¤¯à¥‹à¤‚ à¤•à¥‡ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¥‹à¤° à¤•à¥‹ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤°à¤–à¥‡à¤‚ à¤¤à¤¾à¤•à¤¿ à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤¸à¥à¤¥à¤¿à¤° à¤°à¤¹à¥‡à¤‚à¥¤", "à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤à¤Ÿà¤•à¥‡ à¤¸à¥‡ à¤¨à¤¹à¥€à¤‚, à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤šà¤²à¤¾à¤à¤à¥¤"]},
    "burpee": {"label": "à¤¬à¤°à¥à¤ªà¥€", "title": "à¤¬à¤°à¥à¤ªà¥€ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤ªà¥à¤²à¥ˆà¤‚à¤• à¤®à¥‡à¤‚ à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¤•à¤° à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤¹à¤²à¥à¤•à¥‡ à¤¸à¥‡ à¤²à¥ˆà¤‚à¤¡ à¤•à¤°à¥‡à¤‚à¥¤", "à¤–à¤¡à¤¼à¥‡ à¤¹à¥‹à¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‹ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‡ à¤¨à¥€à¤šà¥‡ à¤²à¤¾à¤à¤à¥¤", "à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤•à¥‹ à¤œà¤²à¥à¤¦à¤¬à¤¾à¤œà¤¼à¥€ à¤®à¥‡à¤‚ à¤¨à¤¹à¥€à¤‚, à¤¸à¥à¤®à¥‚à¤¦ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "jumping_jack": {"label": "à¤œà¤‚à¤ªà¤¿à¤‚à¤— à¤œà¥ˆà¤•", "title": "à¤œà¤‚à¤ªà¤¿à¤‚à¤— à¤œà¥ˆà¤• à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¹à¤²à¥à¤•à¥‡ à¤¸à¥‡ à¤²à¥ˆà¤‚à¤¡ à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤¥à¥‹à¤¡à¤¼à¤¾ à¤®à¥‹à¤¡à¤¼à¥‡à¤‚à¥¤", "à¤¹à¤¾à¤¥ à¤”à¤° à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥€ à¤²à¤¯ à¤à¤• à¤œà¥ˆà¤¸à¥€ à¤°à¤–à¥‡à¤‚à¥¤", "à¤ªà¥‚à¤°à¥‡ à¤¸à¥‡à¤Ÿ à¤®à¥‡à¤‚ à¤§à¤¡à¤¼ à¤•à¥‹ à¤¸à¥€à¤§à¤¾ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "glute_bridge": {"label": "à¤—à¥à¤²à¥‚à¤Ÿ à¤¬à¥à¤°à¤¿à¤œ", "title": "à¤—à¥à¤²à¥‚à¤Ÿ à¤¬à¥à¤°à¤¿à¤œ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤‰à¤ à¤¾à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤à¤¡à¤¼à¤¿à¤¯à¥‹à¤‚ à¤¸à¥‡ à¤¦à¤¬à¤¾à¤µ à¤¦à¥‡à¤‚à¥¤", "à¤Šà¤ªà¤° à¤ªà¤¹à¥à¤à¤šà¤•à¤° à¤•à¤®à¤° à¤®à¥‹à¤¡à¤¼à¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤—à¥à¤²à¥‚à¤Ÿà¥à¤¸ à¤•à¥‹ à¤•à¤¸à¥‡à¤‚à¥¤", "à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤¸à¥€à¤§à¥€ à¤¦à¤¿à¤¶à¤¾ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "calf_raise": {"label": "à¤•à¤¾à¤« à¤°à¥‡à¤œà¤¼", "title": "à¤•à¤¾à¤« à¤°à¥‡à¤œà¤¼ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‡ à¤…à¤—à¤²à¥‡ à¤¹à¤¿à¤¸à¥à¤¸à¥‡ à¤ªà¤° à¤¸à¥€à¤§à¤¾ à¤Šà¤ªà¤° à¤‰à¤ à¥‡à¤‚à¥¤", "à¤Šà¤ªà¤° à¤à¤• à¤›à¥‹à¤Ÿà¤¾ à¤µà¤¿à¤°à¤¾à¤® à¤²à¥‡à¤‚à¥¤", "à¤Ÿà¤–à¤¨à¥‹à¤‚ à¤•à¥‹ à¤¬à¤¾à¤¹à¤° à¤¯à¤¾ à¤…à¤‚à¤¦à¤° à¤°à¥‹à¤² à¤¨ à¤¹à¥‹à¤¨à¥‡ à¤¦à¥‡à¤‚à¥¤"]},
    "russian_twist": {"label": "à¤°à¤¶à¤¿à¤¯à¤¨ à¤Ÿà¥à¤µà¤¿à¤¸à¥à¤Ÿ", "title": "à¤°à¤¶à¤¿à¤¯à¤¨ à¤Ÿà¥à¤µà¤¿à¤¸à¥à¤Ÿ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¸à¤¿à¤°à¥à¤« à¤¹à¤¾à¤¥ à¤¨à¤¹à¥€à¤‚, à¤ªà¤¸à¤²à¤¿à¤¯à¥‹à¤‚ à¤¸à¥‡ à¤°à¥‹à¤Ÿà¥‡à¤¶à¤¨ à¤•à¤°à¥‡à¤‚à¥¤", "à¤ªà¥€à¤›à¥‡ à¤à¥à¤•à¤¤à¥‡ à¤¹à¥à¤ à¤›à¤¾à¤¤à¥€ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤§à¥€à¤°à¥‡ à¤•à¤°à¥‡à¤‚ à¤¤à¤¾à¤•à¤¿ à¤‘à¤¬à¥à¤²à¤¿à¤•à¥à¤¸ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤°à¤¹à¥‡à¤‚à¥¤"]},
    "bicycle_crunch": {"label": "à¤¬à¤¾à¤‡à¤¸à¤¿à¤•à¤² à¤•à¥à¤°à¤‚à¤š", "title": "à¤¬à¤¾à¤‡à¤¸à¤¿à¤•à¤² à¤•à¥à¤°à¤‚à¤š à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤—à¤°à¥à¤¦à¤¨ à¤–à¥€à¤‚à¤šà¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤§à¤¡à¤¼ à¤•à¥‹ à¤®à¥‹à¤¡à¤¼à¥‡à¤‚à¥¤", "à¤µà¤¿à¤ªà¤°à¥€à¤¤ à¤ªà¥ˆà¤° à¤•à¥‹ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤ªà¥‚à¤°à¤¾ à¤¸à¥€à¤§à¤¾ à¤•à¤°à¥‡à¤‚à¥¤", "à¤²à¥‹à¤…à¤° à¤¬à¥ˆà¤• à¤•à¥‹ à¤¹à¤²à¥à¤•à¤¾ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "side_lunge": {"label": "à¤¸à¤¾à¤‡à¤¡ à¤²à¤‚à¤œ", "title": "à¤¸à¤¾à¤‡à¤¡ à¤²à¤‚à¤œ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¤¾à¤® à¤•à¤°à¤¨à¥‡ à¤µà¤¾à¤²à¥‡ à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤®à¥‡à¤‚ à¤ªà¥€à¤›à¥‡ à¤¬à¥ˆà¤ à¥‡à¤‚à¥¤", "à¤¸à¥à¤¥à¤¿à¤° à¤ªà¥ˆà¤° à¤•à¥‹ à¤ªà¥‚à¤°à¤¾ à¤œà¤®à¥€à¤¨ à¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤¸à¤¾à¤‡à¤¡ à¤®à¥‡à¤‚ à¤œà¤¾à¤¤à¥‡ à¤¹à¥à¤ à¤›à¤¾à¤¤à¥€ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤"]},
    "high_knees": {"label": "à¤¹à¤¾à¤ˆ à¤¨à¥€à¤œà¤¼", "title": "à¤¹à¤¾à¤ˆ à¤¨à¥€à¤œà¤¼ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤¤à¥‡à¤œà¤¼ à¤²à¥‡à¤•à¤¿à¤¨ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤Šà¤ªà¤° à¤¡à¥à¤°à¤¾à¤‡à¤µ à¤•à¤°à¥‡à¤‚à¥¤", "à¤§à¤¡à¤¼ à¤¸à¥€à¤§à¤¾ à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤ªà¥ˆà¤°à¥‹à¤‚ à¤ªà¤° à¤¹à¤²à¥à¤•à¥‡ à¤°à¤¹à¥‡à¤‚à¥¤", "à¤°à¤¿à¤¦à¥à¤® à¤•à¥‡ à¤²à¤¿à¤ à¤¹à¤¾à¤¥à¥‹à¤‚ à¤•à¤¾ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤¸à¥à¤µà¤¿à¤‚à¤— à¤°à¤–à¥‡à¤‚à¥¤"]},
    "step_up": {"label": "à¤¸à¥à¤Ÿà¥‡à¤ª-à¤…à¤ª", "title": "à¤¸à¥à¤Ÿà¥‡à¤ª-à¤…à¤ª à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¸à¥à¤Ÿà¥‡à¤ª à¤¯à¤¾ à¤¬à¥‡à¤‚à¤š à¤ªà¤° à¤ªà¥‚à¤°à¥‡ à¤ªà¥ˆà¤° à¤¸à¥‡ à¤¦à¤¬à¤¾à¤µ à¤¦à¥‡à¤‚à¥¤", "à¤Šà¤ªà¤° à¤–à¤¡à¤¼à¥‡ à¤¹à¥‹à¤¤à¥‡ à¤¸à¤®à¤¯ à¤†à¤—à¥‡ à¤¨ à¤à¥à¤•à¥‡à¤‚à¥¤", "à¤¨à¥€à¤šà¥‡ à¤²à¥Œà¤Ÿà¤¤à¥‡ à¤¸à¤®à¤¯ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "tadasana": {"label": "à¤¤à¤¾à¤¡à¤¼à¤¾à¤¸à¤¨", "title": "à¤¤à¤¾à¤¡à¤¼à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¦à¥‹à¤¨à¥‹à¤‚ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤ªà¤° à¤¸à¤®à¤¾à¤¨ à¤­à¤¾à¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤¸à¤¿à¤° à¤•à¥‡ à¤¶à¥€à¤°à¥à¤· à¤¸à¥‡ à¤Šà¤ªà¤° à¤•à¥€ à¤“à¤° à¤²à¤‚à¤¬à¤¾à¤ˆ à¤¬à¤¨à¤¾à¤à¤à¥¤", "à¤›à¤¾à¤¤à¥€ à¤–à¥à¤²à¥€ à¤°à¤–à¤¤à¥‡ à¤¹à¥à¤ à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‹ à¤¢à¥€à¤²à¤¾ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "vrikshasana": {"label": "à¤µà¥ƒà¤•à¥à¤·à¤¾à¤¸à¤¨", "title": "à¤µà¥ƒà¤•à¥à¤·à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤–à¤¡à¤¼à¥‡ à¤ªà¥ˆà¤° à¤•à¥‹ à¤®à¤œà¤¬à¥‚à¤¤à¥€ à¤¸à¥‡ à¤œà¤®à¥€à¤¨ à¤®à¥‡à¤‚ à¤¦à¤¬à¤¾à¤à¤à¥¤", "à¤‰à¤ à¥‡ à¤¹à¥à¤ à¤˜à¥à¤Ÿà¤¨à¥‡ à¤•à¥‹ à¤†à¤—à¥‡ à¤—à¤¿à¤°à¤¨à¥‡ à¤¨ à¤¦à¥‡à¤‚à¥¤", "à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤•à¥‡ à¤²à¤¿à¤ à¤à¤• à¤¬à¤¿à¤‚à¤¦à¥ à¤ªà¤° à¤¦à¥ƒà¤·à¥à¤Ÿà¤¿ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "utkatasana": {"label": "à¤‰à¤¤à¥à¤•à¤Ÿà¤¾à¤¸à¤¨", "title": "à¤‰à¤¤à¥à¤•à¤Ÿà¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤ªà¥€à¤›à¥‡ à¤­à¥‡à¤œà¥‡à¤‚ à¤”à¤° à¤›à¤¾à¤¤à¥€ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤ªà¥‚à¤°à¤¾ à¤­à¤¾à¤° à¤ªà¥‚à¤°à¥‡ à¤ªà¥ˆà¤° à¤®à¥‡à¤‚ à¤¬à¤¾à¤à¤Ÿà¥‡à¤‚à¥¤", "à¤•à¤‚à¤§à¥‡ à¤šà¤¢à¤¼à¤¾à¤ à¤¬à¤¿à¤¨à¤¾ à¤¹à¤¾à¤¥ à¤²à¤‚à¤¬à¥‡ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "virabhadrasana_i": {"label": "à¤µà¥€à¤°à¤­à¤¦à¥à¤°à¤¾à¤¸à¤¨ I", "title": "à¤µà¥€à¤°à¤­à¤¦à¥à¤°à¤¾à¤¸à¤¨ I à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¸à¤¾à¤®à¤¨à¥‡ à¤µà¤¾à¤²à¥‡ à¤˜à¥à¤Ÿà¤¨à¥‡ à¤•à¥‹ à¤®à¥‹à¤¡à¤¼à¥‡à¤‚ à¤”à¤° à¤ªà¥€à¤›à¥‡ à¤•à¥€ à¤à¤¡à¤¼à¥€ à¤Ÿà¤¿à¤•à¤¾à¤à¤à¥¤", "à¤œà¤¹à¤¾à¤ à¤¤à¤• à¤¸à¤‚à¤­à¤µ à¤¹à¥‹ à¤§à¤¡à¤¼ à¤•à¥‹ à¤¸à¥€à¤§à¤¾ à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¤®à¤° à¤¦à¤¬à¤¾à¤ à¤¬à¤¿à¤¨à¤¾ à¤¹à¤¾à¤¥ à¤Šà¤ªà¤° à¤‰à¤ à¤¾à¤à¤à¥¤"]},
    "virabhadrasana_ii": {"label": "à¤µà¥€à¤°à¤­à¤¦à¥à¤°à¤¾à¤¸à¤¨ II", "title": "à¤µà¥€à¤°à¤­à¤¦à¥à¤°à¤¾à¤¸à¤¨ II à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¸à¤¾à¤®à¤¨à¥‡ à¤µà¤¾à¤²à¥‡ à¤˜à¥à¤Ÿà¤¨à¥‡ à¤•à¥‹ à¤Ÿà¤–à¤¨à¥‡ à¤•à¥‡ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤¦à¥‹à¤¨à¥‹à¤‚ à¤¹à¤¾à¤¥à¥‹à¤‚ à¤•à¥‹ à¤¬à¤°à¤¾à¤¬à¤° à¤²à¤‚à¤¬à¤¾à¤ˆ à¤®à¥‡à¤‚ à¤«à¥ˆà¤²à¤¾à¤à¤à¥¤", "à¤§à¤¡à¤¼ à¤•à¥‹ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‡ à¤¬à¥€à¤š à¤¸à¤‚à¤¤à¥à¤²à¤¿à¤¤ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "trikonasana": {"label": "à¤¤à¥à¤°à¤¿à¤•à¥‹à¤£à¤¾à¤¸à¤¨", "title": "à¤¤à¥à¤°à¤¿à¤•à¥‹à¤£à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¨à¥€à¤šà¥‡ à¤œà¤¾à¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤•à¤®à¤° à¤•à¥‡ à¤¦à¥‹à¤¨à¥‹à¤‚ à¤“à¤° à¤²à¤‚à¤¬à¤¾à¤ˆ à¤¬à¤¨à¤¾à¤à¤à¥¤", "à¤œà¤¹à¤¾à¤ à¤¸à¤‚à¤­à¤µ à¤¹à¥‹ à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‹ à¤à¤• à¤²à¤¾à¤‡à¤¨ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤", "à¤¦à¥‹à¤¨à¥‹à¤‚ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‹ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤°à¤–à¥‡à¤‚ à¤²à¥‡à¤•à¤¿à¤¨ à¤²à¥‰à¤• à¤¨ à¤•à¤°à¥‡à¤‚à¥¤"]},
    "adho_mukha_svanasana": {"label": "à¤…à¤§à¥‹ à¤®à¥à¤– à¤¶à¥à¤µà¤¾à¤¨à¤¾à¤¸à¤¨", "title": "à¤…à¤§à¥‹ à¤®à¥à¤– à¤¶à¥à¤µà¤¾à¤¨à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤¹à¤¥à¥‡à¤²à¤¿à¤¯à¥‹à¤‚ à¤¸à¥‡ à¤œà¤¼à¤®à¥€à¤¨ à¤•à¥‹ à¤®à¤œà¤¬à¥‚à¤¤ à¤§à¤•à¥à¤•à¤¾ à¤¦à¥‡à¤‚à¥¤", "à¤°à¥€à¤¢à¤¼ à¤²à¤‚à¤¬à¥€ à¤•à¤°à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤Šà¤ªà¤° à¤‰à¤ à¤¾à¤à¤à¥¤", "à¤¯à¤¦à¤¿ à¤ªà¥€à¤  à¤—à¥‹à¤² à¤¹à¥‹ à¤°à¤¹à¥€ à¤¹à¥‹ à¤¤à¥‹ à¤˜à¥à¤Ÿà¤¨à¥‡ à¤¨à¤°à¤® à¤°à¤–à¥‡à¤‚à¥¤"]},
    "bhujangasana": {"label": "à¤­à¥à¤œà¤‚à¤—à¤¾à¤¸à¤¨", "title": "à¤­à¥à¤œà¤‚à¤—à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤—à¤°à¥à¤¦à¤¨ à¤®à¥‹à¤¡à¤¼à¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤›à¤¾à¤¤à¥€ à¤•à¥‹ à¤Šà¤ªà¤° à¤‰à¤ à¤¾à¤à¤à¥¤", "à¤•à¥‹à¤¹à¤¨à¤¿à¤¯à¥‹à¤‚ à¤•à¥‹ à¤¥à¥‹à¤¡à¤¼à¤¾ à¤®à¥à¤¡à¤¼à¤¾ à¤”à¤° à¤ªà¤¸à¤²à¤¿à¤¯à¥‹à¤‚ à¤•à¥‡ à¤ªà¤¾à¤¸ à¤°à¤–à¥‡à¤‚à¥¤", "à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‡ à¤Šà¤ªà¤°à¥€ à¤¹à¤¿à¤¸à¥à¤¸à¥‡ à¤•à¥‹ à¤®à¥ˆà¤Ÿ à¤ªà¤° à¤¦à¤¬à¤¾à¤à¤à¥¤"]},
    "setu_bandhasana": {"label": "à¤¸à¥‡à¤¤à¥ à¤¬à¤‚à¤§à¤¾à¤¸à¤¨", "title": "à¤¸à¥‡à¤¤à¥ à¤¬à¤‚à¤§à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤ªà¥ˆà¤°à¥‹à¤‚ à¤¸à¥‡ à¤¦à¤¬à¤¾à¤µ à¤¦à¥‡à¤•à¤° à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤‰à¤ à¤¾à¤à¤à¥¤", "à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤¬à¤¾à¤¹à¤° à¤«à¥ˆà¤²à¤¨à¥‡ à¤¨ à¤¦à¥‡à¤‚à¥¤", "à¤—à¤°à¥à¤¦à¤¨ à¤ªà¤° à¤¦à¤¬à¤¾à¤µ à¤¡à¤¾à¤²à¥‡ à¤¬à¤¿à¤¨à¤¾ à¤›à¤¾à¤¤à¥€ à¤–à¥‹à¤²à¥‡à¤‚à¥¤"]},
    "naukasana": {"label": "à¤¨à¥Œà¤•à¤¾à¤¸à¤¨", "title": "à¤¨à¥Œà¤•à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤°à¥€à¤¢à¤¼ à¤—à¥‹à¤² à¤¹à¥‹à¤¨à¥‡ à¤¸à¥‡ à¤¬à¤šà¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤›à¤¾à¤¤à¥€ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¥‡à¤‚ à¤”à¤° à¤¬à¥ˆà¤ à¤¨à¥‡ à¤•à¥€ à¤¹à¤¡à¥à¤¡à¤¿à¤¯à¥‹à¤‚ à¤ªà¤° à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¤‚à¤ªà¤¨ à¤¹à¥‹à¤¨à¥‡ à¤ªà¤° à¤­à¥€ à¤¸à¤¾à¤à¤¸ à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¥‡à¤‚à¥¤"]},
    "balasana": {"label": "à¤¬à¤¾à¤²à¤¾à¤¸à¤¨", "title": "à¤¬à¤¾à¤²à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤à¤¡à¤¼à¤¿à¤¯à¥‹à¤‚ à¤•à¥€ à¤“à¤° à¤†à¤°à¤¾à¤® à¤¸à¥‡ à¤¨à¥€à¤šà¥‡ à¤œà¤¾à¤¨à¥‡ à¤¦à¥‡à¤‚à¥¤", "à¤•à¤‚à¤§à¥‹à¤‚ à¤”à¤° à¤œà¤¬à¤¡à¤¼à¥‡ à¤•à¥‹ à¤¢à¥€à¤²à¤¾ à¤°à¤–à¥‡à¤‚à¥¤", "à¤¸à¤¾à¤à¤¸ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤°à¥€à¤¢à¤¼ à¤•à¥‹ à¤§à¥€à¤°à¥‡ à¤²à¤‚à¤¬à¤¾ à¤•à¤°à¥‡à¤‚à¥¤"]},
    "phalakasana": {"label": "à¤«à¤²à¤•à¤¾à¤¸à¤¨", "title": "à¤«à¤²à¤•à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤•à¤‚à¤§à¥‹à¤‚ à¤¸à¥‡ à¤à¤¡à¤¼à¤¿à¤¯à¥‹à¤‚ à¤¤à¤• à¤¸à¥€à¤§à¥€ à¤²à¤¾à¤‡à¤¨ à¤°à¤–à¥‡à¤‚à¥¤", "à¤Šà¤ªà¤°à¥€ à¤ªà¥€à¤  à¤šà¥Œà¤¡à¤¼à¥€ à¤°à¤–à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤œà¤¼à¤®à¥€à¤¨ à¤•à¥‹ à¤§à¤•à¥à¤•à¤¾ à¤¦à¥‡à¤‚à¥¤", "à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¥‡à¤‚ à¤¤à¤¾à¤•à¤¿ à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤¨à¥€à¤šà¥‡ à¤¨ à¤à¥à¤•à¥‡à¤‚à¥¤"]},
    "virabhadrasana_iii": {"label": "à¤µà¥€à¤°à¤­à¤¦à¥à¤°à¤¾à¤¸à¤¨ III", "title": "à¤µà¥€à¤°à¤­à¤¦à¥à¤°à¤¾à¤¸à¤¨ III à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤†à¤—à¥‡ à¤”à¤° à¤ªà¥€à¤›à¥‡ à¤¸à¤®à¤¾à¤¨ à¤Šà¤°à¥à¤œà¤¾ à¤¸à¥‡ à¤«à¥ˆà¤²à¥‡à¤‚à¥¤", "à¤œà¤¹à¤¾à¤ à¤¤à¤• à¤¸à¤‚à¤­à¤µ à¤¹à¥‹ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤¬à¤°à¤¾à¤¬à¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤¡à¤—à¤®à¤—à¤¾à¤ à¤¤à¥‹ à¤–à¤¡à¤¼à¥‡ à¤ªà¥ˆà¤° à¤®à¥‡à¤‚ à¤¹à¤²à¥à¤•à¤¾ à¤®à¥‹à¤¡à¤¼ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "ardha_chandrasana": {"label": "à¤…à¤°à¥à¤§ à¤šà¤‚à¤¦à¥à¤°à¤¾à¤¸à¤¨", "title": "à¤…à¤°à¥à¤§ à¤šà¤‚à¤¦à¥à¤°à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤Šà¤ªà¤°à¥€ à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤•à¥‹ à¤–à¤¡à¤¼à¥‡ à¤ªà¥ˆà¤° à¤•à¥‡ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤–à¤¡à¤¼à¥‡ à¤ªà¥ˆà¤° à¤¸à¥‡ à¤œà¤¼à¤®à¥€à¤¨ à¤ªà¤° à¤®à¤œà¤¬à¥‚à¤¤à¥€ à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¤®à¤° à¤—à¤¿à¤°à¤¾à¤ à¤¬à¤¿à¤¨à¤¾ à¤›à¤¾à¤¤à¥€ à¤–à¥‹à¤²à¥‡à¤‚à¥¤"]},
    "paschimottanasana": {"label": "à¤ªà¤¶à¥à¤šà¤¿à¤®à¥‹à¤¤à¥à¤¤à¤¾à¤¨à¤¾à¤¸à¤¨", "title": "à¤ªà¤¶à¥à¤šà¤¿à¤®à¥‹à¤¤à¥à¤¤à¤¾à¤¨à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤†à¤—à¥‡ à¤à¥à¤•à¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤°à¥€à¤¢à¤¼ à¤²à¤‚à¤¬à¥€ à¤•à¤°à¥‡à¤‚à¥¤", "à¤Šà¤ªà¤°à¥€ à¤ªà¥€à¤  à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤¸à¥‡ à¤…à¤§à¤¿à¤• à¤à¥à¤•à¥‡à¤‚à¥¤", "à¤¸à¤¾à¤à¤¸ à¤•à¥‹ à¤¨à¤°à¤® à¤”à¤° à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¥‡à¤‚à¥¤"]},
    "ustrasana": {"label": "à¤‰à¤·à¥à¤Ÿà¥à¤°à¤¾à¤¸à¤¨", "title": "à¤‰à¤·à¥à¤Ÿà¥à¤°à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤à¤¡à¤¼à¤¿à¤¯à¥‹à¤‚ à¤¤à¤• à¤ªà¤¹à¥à¤à¤šà¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤›à¤¾à¤¤à¥€ à¤‰à¤ à¤¾à¤à¤à¥¤", "à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‡ à¤Šà¤ªà¤° à¤†à¤—à¥‡ à¤°à¤–à¥‡à¤‚à¥¤", "à¤—à¤°à¥à¤¦à¤¨ à¤•à¥‹ à¤—à¤¿à¤°à¤¾à¤¨à¥‡ à¤•à¥‡ à¤¬à¤œà¤¾à¤¯ à¤²à¤‚à¤¬à¤¾ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "malasana": {"label": "à¤®à¤¾à¤²à¤¾à¤¸à¤¨", "title": "à¤®à¤¾à¤²à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‹ à¤œà¤¡à¤¼ à¤¸à¥‡ à¤Ÿà¤¿à¤•à¤¾à¤à¤ à¤”à¤° à¤›à¤¾à¤¤à¥€ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤•à¥‹à¤¹à¤¨à¤¿à¤¯à¥‹à¤‚ à¤¸à¥‡ à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤¹à¤²à¥à¤•à¤¾ à¤¬à¤¾à¤¹à¤° à¤–à¥‹à¤²à¥‡à¤‚à¥¤", "à¤œà¤¿à¤¤à¤¨à¥€ à¤—à¤¤à¤¿à¤¶à¥€à¤²à¤¤à¤¾ à¤¹à¥‹ à¤‰à¤¤à¤¨à¤¾ à¤¨à¥€à¤šà¥‡ à¤¬à¥ˆà¤ à¥‡à¤‚, à¤²à¥‡à¤•à¤¿à¤¨ à¤¢à¤¹à¥‡à¤‚ à¤¨à¤¹à¥€à¤‚à¥¤"]},
    "dhanurasana": {"label": "à¤§à¤¨à¥à¤°à¤¾à¤¸à¤¨", "title": "à¤§à¤¨à¥à¤°à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤›à¤¾à¤¤à¥€ à¤‰à¤ à¤¾à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‹ à¤¹à¤¾à¤¥à¥‹à¤‚ à¤•à¥€ à¤“à¤° à¤ªà¥€à¤›à¥‡ à¤•à¤¿à¤• à¤•à¤°à¥‡à¤‚à¥¤", "à¤†à¤—à¥‡ à¤¶à¤°à¥€à¤° à¤–à¥à¤²à¤¨à¥‡ à¤ªà¤° à¤­à¥€ à¤¸à¤¾à¤à¤¸ à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¥‡à¤‚à¥¤", "à¤²à¥‹à¤…à¤° à¤¬à¥ˆà¤• à¤ªà¤° à¤…à¤§à¤¿à¤• à¤¦à¤¬à¤¾à¤µ à¤¨ à¤¡à¤¾à¤²à¥‡à¤‚à¥¤"]},
    "salabhasana": {"label": "à¤¶à¤²à¤­à¤¾à¤¸à¤¨", "title": "à¤¶à¤²à¤­à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤ªà¥ˆà¤°à¥‹à¤‚ à¤”à¤° à¤›à¤¾à¤¤à¥€ à¤•à¥‹ à¤¸à¤¾à¤¥ à¤®à¥‡à¤‚ à¤‰à¤ à¤¾à¤à¤à¥¤", "à¤—à¤°à¥à¤¦à¤¨ à¤•à¥‡ à¤ªà¥€à¤›à¥‡ à¤¹à¤¿à¤¸à¥à¤¸à¥‡ à¤•à¥‹ à¤²à¤‚à¤¬à¤¾ à¤°à¤–à¥‡à¤‚à¥¤", "à¤‰à¤‚à¤—à¤²à¤¿à¤¯à¥‹à¤‚ à¤”à¤° à¤ªà¥ˆà¤° à¤•à¥€ à¤‰à¤‚à¤—à¤²à¤¿à¤¯à¥‹à¤‚ à¤¤à¤• à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤ªà¤¹à¥à¤à¤š à¤¬à¤¨à¤¾à¤ à¤°à¤–à¥‡à¤‚à¥¤"]},
    "supta_baddha_konasana": {"label": "à¤¸à¥à¤ªà¥à¤¤ à¤¬à¤¦à¥à¤§ à¤•à¥‹à¤£à¤¾à¤¸à¤¨", "title": "à¤¸à¥à¤ªà¥à¤¤ à¤¬à¤¦à¥à¤§ à¤•à¥‹à¤£à¤¾à¤¸à¤¨ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£", "tips": ["à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤¬à¤¿à¤¨à¤¾ à¤œà¤¼à¥‹à¤° à¤¦à¤¿à¤ à¤¸à¥à¤µà¤¾à¤­à¤¾à¤µà¤¿à¤• à¤°à¥‚à¤ª à¤¸à¥‡ à¤–à¥à¤²à¤¨à¥‡ à¤¦à¥‡à¤‚à¥¤", "à¤°à¤¿à¤¬à¥à¤¸ à¤”à¤° à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‹ à¤œà¤¼à¤®à¥€à¤¨ à¤ªà¤° à¤¨à¤°à¤® à¤°à¤–à¥‡à¤‚à¥¤", "à¤¶à¤¾à¤‚à¤¤ à¤¸à¤¾à¤à¤¸ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤§à¥€à¤°à¥‡-à¤§à¥€à¤°à¥‡ à¤µà¤¿à¤¶à¥à¤°à¤¾à¤® à¤¬à¤¢à¤¼à¤¾à¤à¤à¥¤"]},
}


def ex_label(exercise: str) -> str:
    if st.session_state.get("language", "en") == "hi":
        return EXERCISE_HI.get(exercise, {}).get("label", EXERCISE_COPY[exercise]["label"])
    return EXERCISE_COPY[exercise]["label"]


def ex_title(exercise: str) -> str:
    if st.session_state.get("language", "en") == "hi":
        return EXERCISE_HI.get(exercise, {}).get("title", EXERCISE_COPY[exercise]["title"])
    return EXERCISE_COPY[exercise]["title"]


def ex_tips(exercise: str) -> List[str]:
    if st.session_state.get("language", "en") == "hi":
        return EXERCISE_HI.get(exercise, {}).get("tips", EXERCISE_COPY[exercise]["tips"])
    return EXERCISE_COPY[exercise]["tips"]


EXERCISE_COPY: Dict[str, Dict[str, List[str] | str]] = {
    "squat": {
        "label": "Squat",
        "title": "Squat analysis",
        "tips": [
            "Keep your chest lifted and your heels grounded.",
            "Lower until your thighs are close to parallel if mobility allows.",
            "Track your knees in line with your toes.",
        ],
    },
    "pushup": {
        "label": "Push-up",
        "title": "Push-up analysis",
        "tips": [
            "Place the camera from the side for cleaner elbow tracking.",
            "Keep shoulders, hips, and ankles in one line.",
            "Avoid flaring your elbows too wide.",
        ],
    },
    "plank": {
        "label": "Plank",
        "title": "Plank analysis",
        "tips": [
            "Brace your core and squeeze your glutes.",
            "Keep your neck neutral and look slightly down.",
            "Maintain a straight line from shoulders to ankles.",
        ],
    },
}

GYM_EXERCISES: Dict[str, Dict[str, List[str] | str]] = {
    "squat": EXERCISE_COPY["squat"],
    "pushup": EXERCISE_COPY["pushup"],
    "plank": EXERCISE_COPY["plank"],
    "lunge": {
        "label": "Lunge",
        "title": "Lunge analysis",
        "tips": [
            "Keep your front knee stacked over the ankle.",
            "Lower straight down instead of drifting forward.",
            "Stay tall through the chest and hips.",
        ],
    },
    "deadlift": {
        "label": "Deadlift",
        "title": "Deadlift analysis",
        "tips": [
            "Hinge from the hips with a long neutral spine.",
            "Keep the weight close to your legs.",
            "Brace your core before every rep.",
        ],
    },
    "shoulder_press": {
        "label": "Shoulder Press",
        "title": "Shoulder press analysis",
        "tips": [
            "Keep ribs down and avoid arching your lower back.",
            "Press in a straight path over the shoulders.",
            "Stay balanced through both feet.",
        ],
    },
    "bicep_curl": {
        "label": "Bicep Curl",
        "title": "Bicep curl analysis",
        "tips": [
            "Keep elbows close to your torso.",
            "Avoid swinging the body to lift the weight.",
            "Control the lowering phase slowly.",
        ],
    },
    "tricep_dip": {
        "label": "Tricep Dip",
        "title": "Tricep dip analysis",
        "tips": [
            "Keep shoulders away from the ears.",
            "Bend elbows straight back instead of flaring wide.",
            "Move with control through a pain-free range.",
        ],
    },
    "pullup": {
        "label": "Pull-up",
        "title": "Pull-up analysis",
        "tips": [
            "Start from a stable hang with the core engaged.",
            "Pull elbows down instead of shrugging the shoulders.",
            "Keep the body from swinging between reps.",
        ],
    },
    "bench_press": {
        "label": "Bench Press",
        "title": "Bench press analysis",
        "tips": [
            "Keep wrists stacked over elbows.",
            "Maintain upper-back tension while pressing.",
            "Drive the bar with even control on both sides.",
        ],
    },
    "mountain_climber": {
        "label": "Mountain Climber",
        "title": "Mountain climber analysis",
        "tips": [
            "Keep shoulders directly over the hands.",
            "Brace the core so the hips stay level.",
            "Drive knees with control instead of bouncing.",
        ],
    },
    "burpee": {
        "label": "Burpee",
        "title": "Burpee analysis",
        "tips": [
            "Land softly and keep the core tight in the plank.",
            "Bring feet underneath the hips before standing.",
            "Keep the movement smooth instead of rushed.",
        ],
    },
    "jumping_jack": {
        "label": "Jumping Jack",
        "title": "Jumping jack analysis",
        "tips": [
            "Land softly with knees slightly bent.",
            "Move arms and legs rhythmically together.",
            "Keep your torso tall throughout the set.",
        ],
    },
    "glute_bridge": {
        "label": "Glute Bridge",
        "title": "Glute bridge analysis",
        "tips": [
            "Drive through the heels to lift the hips.",
            "Squeeze the glutes at the top instead of over-arching the back.",
            "Keep knees tracking straight ahead.",
        ],
    },
    "calf_raise": {
        "label": "Calf Raise",
        "title": "Calf raise analysis",
        "tips": [
            "Lift straight up through the balls of the feet.",
            "Pause briefly at the top of each rep.",
            "Avoid rolling the ankles outward or inward.",
        ],
    },
    "russian_twist": {
        "label": "Russian Twist",
        "title": "Russian twist analysis",
        "tips": [
            "Rotate from the ribs instead of just swinging the arms.",
            "Keep the chest lifted while leaning back.",
            "Move slowly enough to feel the obliques work.",
        ],
    },
    "bicycle_crunch": {
        "label": "Bicycle Crunch",
        "title": "Bicycle crunch analysis",
        "tips": [
            "Rotate the torso rather than pulling the neck.",
            "Extend the opposite leg fully with control.",
            "Keep the lower back gently braced.",
        ],
    },
    "side_lunge": {
        "label": "Side Lunge",
        "title": "Side lunge analysis",
        "tips": [
            "Sit back into the working hip.",
            "Keep the planted foot fully grounded.",
            "Stay tall through the chest as you shift sideways.",
        ],
    },
    "high_knees": {
        "label": "High Knees",
        "title": "High knees analysis",
        "tips": [
            "Drive knees up with quick but controlled rhythm.",
            "Stay light on the feet and upright through the torso.",
            "Use active arm swing to support the pace.",
        ],
    },
    "step_up": {
        "label": "Step-up",
        "title": "Step-up analysis",
        "tips": [
            "Push through the full foot on the bench or step.",
            "Stand tall at the top without leaning forward.",
            "Lower back down under control.",
        ],
    },
}

ASANA_EXERCISES: Dict[str, Dict[str, List[str] | str]] = {
    "tadasana": {
        "label": "Tadasana",
        "title": "Tadasana analysis",
        "tips": [
            "Stand evenly through both feet.",
            "Lengthen through the crown of the head.",
            "Relax the shoulders while keeping the chest open.",
        ],
    },
    "vrikshasana": {
        "label": "Vrikshasana",
        "title": "Tree pose analysis",
        "tips": [
            "Press the standing foot firmly into the floor.",
            "Avoid letting the lifted knee collapse forward.",
            "Find one steady gaze point for balance.",
        ],
    },
    "utkatasana": {
        "label": "Utkatasana",
        "title": "Chair pose analysis",
        "tips": [
            "Sit the hips back while keeping the chest lifted.",
            "Keep weight spread across the whole foot.",
            "Reach arms long without shrugging the shoulders.",
        ],
    },
    "virabhadrasana_i": {
        "label": "Virabhadrasana I",
        "title": "Warrior I analysis",
        "tips": [
            "Bend the front knee while grounding the back heel.",
            "Square the torso as much as your hips allow.",
            "Reach upward without compressing the lower back.",
        ],
    },
    "virabhadrasana_ii": {
        "label": "Virabhadrasana II",
        "title": "Warrior II analysis",
        "tips": [
            "Stack the front knee over the ankle.",
            "Extend both arms evenly out to the sides.",
            "Keep the torso centered between the legs.",
        ],
    },
    "trikonasana": {
        "label": "Trikonasana",
        "title": "Triangle pose analysis",
        "tips": [
            "Lengthen both sides of the waist before reaching down.",
            "Stack the shoulders vertically if possible.",
            "Keep both legs active and straight without locking.",
        ],
    },
    "adho_mukha_svanasana": {
        "label": "Adho Mukha Svanasana",
        "title": "Downward dog analysis",
        "tips": [
            "Push the floor away strongly through the hands.",
            "Lift the hips high to lengthen the spine.",
            "Soften the knees if the back rounds.",
        ],
    },
    "bhujangasana": {
        "label": "Bhujangasana",
        "title": "Cobra pose analysis",
        "tips": [
            "Lift through the chest instead of cranking the neck.",
            "Keep elbows slightly bent and close to the ribs.",
            "Press the tops of the feet into the mat.",
        ],
    },
    "setu_bandhasana": {
        "label": "Setu Bandhasana",
        "title": "Bridge pose analysis",
        "tips": [
            "Lift the hips by pressing through the feet.",
            "Keep knees tracking forward instead of splaying out.",
            "Open the chest without overloading the neck.",
        ],
    },
    "naukasana": {
        "label": "Naukasana",
        "title": "Boat pose analysis",
        "tips": [
            "Lift the chest to avoid rounding the spine.",
            "Brace the core and balance on the sit bones.",
            "Keep the breath steady even when shaking.",
        ],
    },
    "balasana": {
        "label": "Balasana",
        "title": "Child's pose analysis",
        "tips": [
            "Let the hips sink back toward the heels.",
            "Relax the shoulders and jaw.",
            "Use the breath to lengthen the spine gently.",
        ],
    },
    "phalakasana": {
        "label": "Phalakasana",
        "title": "Plank pose analysis",
        "tips": [
            "Keep a straight line from shoulders to heels.",
            "Push the floor away to stay broad through the upper back.",
            "Brace the core so the hips do not sag.",
        ],
    },
    "virabhadrasana_iii": {
        "label": "Virabhadrasana III",
        "title": "Warrior III analysis",
        "tips": [
            "Reach forward and back with equal energy.",
            "Keep the hips as level as possible.",
            "Micro-bend the standing knee if balance is shaky.",
        ],
    },
    "ardha_chandrasana": {
        "label": "Ardha Chandrasana",
        "title": "Half moon analysis",
        "tips": [
            "Stack the top hip over the standing leg.",
            "Press firmly through the standing foot.",
            "Open the chest without collapsing the waist.",
        ],
    },
    "paschimottanasana": {
        "label": "Paschimottanasana",
        "title": "Seated forward fold analysis",
        "tips": [
            "Lengthen the spine before folding forward.",
            "Reach from the hips more than the upper back.",
            "Keep the breath soft and even.",
        ],
    },
    "ustrasana": {
        "label": "Ustrasana",
        "title": "Camel pose analysis",
        "tips": [
            "Lift the chest before reaching for the heels.",
            "Keep the hips pressing forward over the knees.",
            "Support the neck by lengthening, not dropping.",
        ],
    },
    "malasana": {
        "label": "Malasana",
        "title": "Garland pose analysis",
        "tips": [
            "Keep the feet rooted and the chest lifted.",
            "Use the elbows to gently open the knees.",
            "Sit as low as mobility allows without collapsing.",
        ],
    },
    "dhanurasana": {
        "label": "Dhanurasana",
        "title": "Bow pose analysis",
        "tips": [
            "Kick the feet back into the hands to lift the chest.",
            "Keep the breath steady even as the front body opens.",
            "Avoid over-compressing the lower back.",
        ],
    },
    "salabhasana": {
        "label": "Salabhasana",
        "title": "Locust pose analysis",
        "tips": [
            "Lift through the legs and chest together.",
            "Keep the back of the neck long.",
            "Reach actively through the fingertips and toes.",
        ],
    },
    "supta_baddha_konasana": {
        "label": "Supta Baddha Konasana",
        "title": "Reclined bound angle analysis",
        "tips": [
            "Let the knees open naturally without force.",
            "Soften the ribs and shoulders into the floor.",
            "Use calm breathing to relax deeper into the pose.",
        ],
    },
}

EXERCISE_COPY = {**GYM_EXERCISES, **ASANA_EXERCISES}
EXERCISE_GROUPS = {
    "Gym Exercises": list(GYM_EXERCISES.keys()),
    "Asana Exercises": list(ASANA_EXERCISES.keys()),
}

PROFILE_OVERVIEWS = {
    "squat": "A lower-body movement focused on hip and knee control, posture, and balance.",
    "pushup": "An upper-body or full-body pattern that trains pressing strength and trunk stability.",
    "plank": "A stability-focused movement or pose that challenges alignment, balance, and core control.",
    "generic": "A controlled movement pattern where posture, joint alignment, and breathing quality matter most.",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #fff9f2;
            --bg-bottom: #efe0d2;
            --surface: rgba(255, 253, 249, 0.94);
            --surface-strong: rgba(255, 255, 255, 0.98);
            --surface-soft: rgba(248, 238, 228, 0.90);
            --border: rgba(85, 58, 31, 0.12);
            --shadow: 0 22px 48px rgba(53, 34, 16, 0.12);
            --text-main: #20150f;
            --text-soft: #5c4a3b;
            --text-muted: #7d6654;
            --accent: #c85e30;
            --accent-deep: #9b451e;
            --teal: #216d63;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(230, 122, 59, 0.18), transparent 24%),
                radial-gradient(circle at bottom right, rgba(33, 109, 99, 0.16), transparent 24%),
                linear-gradient(145deg, var(--bg-top) 0%, #f7ebde 40%, var(--bg-bottom) 100%);
        }
        .block-container {
            max-width: 1380px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.80), rgba(244, 230, 216, 0.92));
            border-right: 1px solid rgba(85, 58, 31, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }
        .hero-card, .panel-card, .metric-tile, .section-card, .media-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 26px;
            box-shadow: var(--shadow);
        }
        .hero-card {
            padding: 36px 38px;
            margin-bottom: 1.2rem;
            background:
                radial-gradient(circle at top right, rgba(200, 94, 48, 0.14), transparent 24%),
                linear-gradient(135deg, rgba(255, 253, 249, 0.98), rgba(255, 245, 234, 0.92));
        }
        .hero-layout {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 28px;
            align-items: center;
        }
        .hero-copy-block {
            max-width: 640px;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(200, 94, 48, 0.10);
            border: 1px solid rgba(200, 94, 48, 0.16);
            color: var(--accent-deep);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }
        .hero-visual {
            border-radius: 28px;
            overflow: hidden;
            border: 1px solid rgba(85, 58, 31, 0.10);
            background: rgba(255,255,255,0.72);
            box-shadow: 0 22px 46px rgba(53, 34, 16, 0.14);
        }
        .hero-visual img {
            display: block;
            width: 100%;
            height: auto;
        }
        .hero-eyebrow {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent-deep);
            font-size: 0.75rem;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: clamp(2.4rem, 3.3vw, 3.6rem);
            line-height: 0.98;
            margin: 0;
            color: var(--text-main);
            max-width: 10ch;
            letter-spacing: -0.03em;
        }
        .hero-text {
            color: var(--text-soft);
            max-width: 34rem;
            line-height: 1.72;
            margin-top: 1.15rem;
            margin-bottom: 0;
            font-size: 1.02rem;
        }
        .hero-meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 1.85rem;
        }
        .hero-meta-item {
            padding: 16px 18px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(85, 58, 31, 0.08);
        }
        .hero-meta-label {
            color: var(--text-muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .hero-meta-value {
            color: var(--text-main);
            font-size: 1.15rem;
            font-weight: 700;
        }
        .section-card {
            padding: 18px 20px;
            margin-bottom: 0.9rem;
            background: linear-gradient(180deg, var(--surface-strong), var(--surface-soft));
        }
        .section-title {
            color: var(--text-main);
            margin: 0;
            font-size: 1.35rem;
        }
        .section-copy {
            color: var(--text-soft);
            line-height: 1.6;
            margin: 0.55rem 0 0;
        }
        .metric-tile {
            padding: 18px 20px;
            min-height: 116px;
            background: linear-gradient(180deg, var(--surface-strong), rgba(248, 239, 229, 0.95));
        }
        .metric-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: var(--text-main);
            font-size: 2.05rem;
            font-weight: 800;
        }
        .panel-card {
            padding: 22px 24px;
            background: linear-gradient(180deg, var(--surface-strong), rgba(247, 238, 229, 0.92));
        }
        .panel-kicker {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent-deep);
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }
        .panel-title {
            color: var(--text-main);
            margin: 0 0 0.7rem 0;
            font-size: 1.4rem;
        }
        .panel-copy, .panel-card li, .panel-card ol {
            color: var(--text-soft);
            line-height: 1.6;
        }
        .panel-card strong {
            color: var(--text-main);
        }
        .panel-card ul, .panel-card ol {
            margin-bottom: 0;
            padding-left: 1.2rem;
        }
        .guide-overview {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 18px;
            align-items: stretch;
            margin-bottom: 1rem;
        }
        .guide-fact-grid {
            display: grid;
            gap: 12px;
        }
        .guide-fact {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(85, 58, 31, 0.08);
        }
        .guide-fact-label {
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .guide-fact-value {
            color: var(--text-main);
            margin-top: 0.3rem;
            line-height: 1.5;
            font-weight: 600;
        }
        .guide-image-shell {
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 18px 38px rgba(53, 34, 16, 0.12);
            background: rgba(255, 255, 255, 0.7);
            margin-bottom: 0.8rem;
        }
        .guide-image-shell img {
            display: block;
            width: 100%;
            height: auto;
        }
        .video-link-caption {
            color: var(--text-muted);
            font-size: 0.92rem;
            margin-top: 0.7rem;
        }
        .coach-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 24px;
        }
        .upload-shell {
            margin-top: 1rem;
        }
        .library-stack, .right-rail-stack {
            display: grid;
            gap: 18px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.62);
            border-radius: 14px;
            padding: 10px 16px;
            color: var(--text-soft);
            border: 1px solid rgba(85, 58, 31, 0.08);
        }
        .stTabs [aria-selected="true"] {
            background: rgba(200, 94, 48, 0.12) !important;
            color: var(--accent-deep) !important;
        }
        .stButton button, .stDownloadButton button, .stLinkButton a {
            border-radius: 14px !important;
            border: 1px solid rgba(85, 58, 31, 0.10) !important;
            min-height: 46px !important;
            font-weight: 700 !important;
        }
        .stFileUploader > div {
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.55);
        }
        video {
            border-radius: 20px;
            overflow: hidden;
        }
        img {
            max-width: 100%;
        }
        @media (max-width: 1200px) {
            .block-container {
                max-width: 1180px;
                padding-left: 1.1rem;
                padding-right: 1.1rem;
            }
            .hero-card {
                padding: 30px 30px;
            }
            .panel-card, .section-card, .metric-tile, .media-card {
                border-radius: 22px;
            }
            .hero-meta-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
            }
        }
        @media (max-width: 1024px) {
            .block-container {
                padding-left: 0.95rem;
                padding-right: 0.95rem;
            }
            .hero-layout {
                grid-template-columns: 1fr;
                gap: 18px;
            }
            .hero-copy-block {
                max-width: none;
            }
            .hero-title {
                max-width: 12ch;
                font-size: clamp(2.2rem, 6vw, 3.1rem);
            }
            .hero-text {
                max-width: none;
                font-size: 0.98rem;
            }
            .guide-overview {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.7rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .hero-card {
                padding: 22px 18px;
                border-radius: 22px;
            }
            .hero-badge {
                font-size: 0.74rem;
                padding: 8px 12px;
                margin-bottom: 0.8rem;
            }
            .hero-title {
                font-size: clamp(1.9rem, 8vw, 2.6rem);
                max-width: none;
            }
            .hero-text {
                font-size: 0.95rem;
                line-height: 1.62;
            }
            .hero-meta-grid {
                grid-template-columns: 1fr;
                gap: 10px;
                margin-top: 1.2rem;
            }
            .hero-meta-item {
                padding: 14px 14px;
            }
            .panel-card, .section-card, .metric-tile, .media-card {
                padding: 16px 16px;
                border-radius: 18px;
            }
            .panel-title {
                font-size: 1.2rem;
            }
            .section-title {
                font-size: 1.1rem;
            }
            .metric-value {
                font-size: 1.7rem;
            }
            .guide-image-shell {
                border-radius: 18px;
            }
            [data-testid="stSidebar"] {
                min-width: auto;
            }
        }
        @media (max-width: 640px) {
            .block-container {
                padding-left: 0.55rem;
                padding-right: 0.55rem;
                padding-bottom: 2rem;
            }
            .hero-card {
                padding: 18px 14px;
            }
            .hero-eyebrow {
                font-size: 0.68rem;
                margin-bottom: 0.55rem;
            }
            .hero-title {
                font-size: clamp(1.7rem, 9vw, 2.2rem);
                line-height: 1.02;
            }
            .hero-text {
                font-size: 0.91rem;
            }
            .panel-kicker {
                font-size: 0.66rem;
                margin-bottom: 0.45rem;
            }
            .panel-title, .section-title {
                font-size: 1.05rem;
            }
            .panel-copy, .panel-card li, .panel-card ol, .section-copy {
                font-size: 0.92rem;
            }
            .metric-tile {
                min-height: 96px;
            }
            .metric-value {
                font-size: 1.45rem;
            }
            .stButton button, .stDownloadButton button, .stLinkButton a {
                width: 100% !important;
                min-height: 42px !important;
            }
        }
        .stLinkButton a {
            background: linear-gradient(135deg, var(--accent), #de7a4a) !important;
            color: white !important;
            box-shadow: 0 14px 28px rgba(200, 94, 48, 0.24);
        }
        .stSelectbox label {
            font-weight: 700;
            color: var(--text-main);
        }
        .stFileUploader label, .stMarkdown, .stCaption {
            color: var(--text-main);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "exercise": "squat",
        "session_state": "Idle",
        "language": "en",
        "camera_facing": "front",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_point(landmarks, index: int) -> Optional[object]:
    point = landmarks[index]
    visibility = getattr(point, "visibility", 1.0)
    if visibility < 0.45:
        return None
    return point


def angle_between(a, b, c) -> Optional[float]:
    if not all([a, b, c]):
        return None
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    degrees = abs(math.degrees(radians))
    return degrees if degrees <= 180 else 360 - degrees


def average_point(*points) -> Optional[Dict[str, float]]:
    valid = [point for point in points if point]
    if not valid:
        return None
    return {
        "x": sum(point.x for point in valid) / len(valid),
        "y": sum(point.y for point in valid) / len(valid),
    }


def side_points(landmarks) -> Dict[str, object]:
    left_shoulder = get_point(landmarks, 11)
    right_shoulder = get_point(landmarks, 12)
    left_hip = get_point(landmarks, 23)
    right_hip = get_point(landmarks, 24)
    left_knee = get_point(landmarks, 25)
    right_knee = get_point(landmarks, 26)
    left_ankle = get_point(landmarks, 27)
    right_ankle = get_point(landmarks, 28)
    left_elbow = get_point(landmarks, 13)
    right_elbow = get_point(landmarks, 14)
    left_wrist = get_point(landmarks, 15)
    right_wrist = get_point(landmarks, 16)

    left_visible = len([p for p in [left_shoulder, left_hip, left_knee, left_ankle] if p])
    right_visible = len([p for p in [right_shoulder, right_hip, right_knee, right_ankle] if p])
    side = "left" if left_visible >= right_visible else "right"

    return {
        "shoulder_mid": average_point(left_shoulder, right_shoulder),
        "hip_mid": average_point(left_hip, right_hip),
        "knee_mid": average_point(left_knee, right_knee),
        "ankle_mid": average_point(left_ankle, right_ankle),
        "shoulder": left_shoulder if side == "left" else right_shoulder,
        "hip": left_hip if side == "left" else right_hip,
        "knee": left_knee if side == "left" else right_knee,
        "ankle": left_ankle if side == "left" else right_ankle,
        "elbow": left_elbow if side == "left" else right_elbow,
        "wrist": left_wrist if side == "left" else right_wrist,
    }


def default_result(exercise: str, message: str) -> Dict[str, object]:
    return {
        "title": ex_title(exercise),
        "message": message,
        "tips": ex_tips(exercise),
        "score": 0,
    }


def get_exercise_profile(exercise: str) -> str:
    squat_profile = {
        "squat",
        "lunge",
        "deadlift",
        "glute_bridge",
        "calf_raise",
        "side_lunge",
        "step_up",
        "utkatasana",
        "virabhadrasana_i",
        "virabhadrasana_ii",
        "malasana",
    }
    push_profile = {
        "pushup",
        "shoulder_press",
        "tricep_dip",
        "bench_press",
        "mountain_climber",
        "burpee",
        "jumping_jack",
        "pullup",
        "phalakasana",
        "adho_mukha_svanasana",
        "bhujangasana",
        "setu_bandhasana",
        "dhanurasana",
        "salabhasana",
    }
    plank_profile = {
        "plank",
        "russian_twist",
        "bicycle_crunch",
        "high_knees",
        "tadasana",
        "vrikshasana",
        "trikonasana",
        "naukasana",
        "balasana",
        "virabhadrasana_iii",
        "ardha_chandrasana",
        "paschimottanasana",
        "ustrasana",
        "supta_baddha_konasana",
    }

    if exercise in squat_profile:
        return "squat"
    if exercise in push_profile:
        return "pushup"
    if exercise in plank_profile:
        return "plank"
    return "generic"


def get_exercise_category(exercise: str) -> str:
    return t("asana") if exercise in ASANA_EXERCISES else t("gym")


def get_exercise_benefits(exercise: str) -> List[str]:
    profile = get_exercise_profile(exercise)
    category = get_exercise_category(exercise)
    hi = st.session_state.get("language", "en") == "hi"

    if exercise in ASANA_EXERCISES:
        if profile == "squat":
            return [
                "à¤¨à¤¿à¤šà¤²à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥€ à¤—à¤¤à¤¿à¤¶à¥€à¤²à¤¤à¤¾ à¤”à¤° à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤¬à¥‡à¤¹à¤¤à¤° à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚, à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤”à¤° à¤Ÿà¤–à¤¨à¥‹à¤‚ à¤•à¥€ à¤¬à¥‡à¤¹à¤¤à¤° à¤œà¤¾à¤—à¤°à¥‚à¤•à¤¤à¤¾ à¤¦à¥‡à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "à¤¤à¤¨à¤¾à¤µ à¤®à¥‡à¤‚ à¤­à¥€ à¤¸à¤¾à¤à¤¸ à¤•à¥‹ à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¤¨à¤¾ à¤¸à¤¿à¤–à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            ] if hi else [
                "Improves lower-body mobility and balance.",
                "Builds awareness of hip, knee, and ankle alignment.",
                "Encourages steady breathing under tension.",
            ]
        if profile == "pushup":
            return [
                "à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥€ à¤¸à¥à¤¥à¤¿à¤°à¤¤à¤¾ à¤”à¤° à¤°à¥€à¤¢à¤¼ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¬à¤¢à¤¼à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "à¤µà¥‡à¤Ÿ-à¤¬à¥‡à¤¯à¤°à¤¿à¤‚à¤— à¤ªà¥‹à¤œà¤¼ à¤®à¥‡à¤‚ à¤Šà¤ªà¤°à¥€ à¤¶à¤°à¥€à¤° à¤•à¥€ à¤¸à¤¹à¤¨à¤¶à¤•à¥à¤¤à¤¿ à¤¬à¤¢à¤¼à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "à¤ªà¥‹à¤¶à¥à¤šà¤° à¤”à¤° à¤¸à¤¾à¤à¤¸ à¤•à¥‡ à¤¤à¤¾à¤²à¤®à¥‡à¤² à¤•à¥‹ à¤¬à¥‡à¤¹à¤¤à¤° à¤¬à¤¨à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            ] if hi else [
                "Builds shoulder stability and spinal control.",
                "Improves upper-body endurance in weight-bearing positions.",
                "Helps connect posture with breathing rhythm.",
            ]
        return [
            "à¤¸à¤‚à¤¤à¥à¤²à¤¨, à¤«à¥‹à¤•à¤¸ à¤”à¤° à¤¬à¥‰à¤¡à¥€ à¤…à¤µà¥‡à¤¯à¤°à¤¨à¥‡à¤¸ à¤¬à¤¢à¤¼à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            "à¤—à¤¹à¤°à¥‡ à¤•à¥‹à¤° à¤¸à¥à¤Ÿà¥‡à¤¬à¤¿à¤²à¤¿à¤Ÿà¥€ à¤”à¤° à¤ªà¥‹à¤¶à¥à¤šà¤° à¤•à¤‚à¤Ÿà¥à¤°à¥‹à¤² à¤•à¥‹ à¤®à¤œà¤¬à¥‚à¤¤ à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            "à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤¬à¤¨à¤¾à¤ à¤°à¤–à¤¤à¥‡ à¤¹à¥à¤ à¤¶à¤¾à¤‚à¤¤ à¤¸à¤¾à¤à¤¸ à¤¸à¤¿à¤–à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
        ] if hi else [
            "Improves balance, focus, and body awareness.",
            "Builds deep core stability and postural control.",
            "Encourages calm, steady breathing while holding alignment.",
        ]

    if profile == "squat":
        return [
            "à¤¨à¤¿à¤šà¤²à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥€ à¤¤à¤¾à¤•à¤¤ à¤”à¤° à¤¸à¤¿à¤‚à¤—à¤²-à¤²à¥‡à¤— à¤•à¤‚à¤Ÿà¥à¤°à¥‹à¤² à¤¬à¤¢à¤¼à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            "à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥€ à¤—à¤¤à¤¿à¤¶à¥€à¤²à¤¤à¤¾ à¤”à¤° à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥€ à¤¸à¥à¤¥à¤¿à¤° à¤¦à¤¿à¤¶à¤¾ à¤®à¥‡à¤‚ à¤®à¤¦à¤¦ à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            "à¤²à¥‹à¤¡à¥‡à¤¡ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤®à¥‡à¤‚ à¤¬à¥à¤°à¥‡à¤¸à¥‡à¤¡ à¤§à¤¡à¤¼ à¤¬à¤¨à¤¾à¤¨à¤¾ à¤¸à¤¿à¤–à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
        ] if hi else [
            "Builds lower-body strength and single-leg control.",
            "Improves hip mobility and stable knee tracking.",
            "Reinforces a strong braced torso during loaded movement.",
        ]
    if profile == "pushup":
        return [
            "à¤Šà¤ªà¤°à¥€ à¤¶à¤°à¥€à¤° à¤•à¥€ à¤ªà¥à¤¶à¤¿à¤‚à¤— à¤¤à¤¾à¤•à¤¤ à¤”à¤° à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥€ à¤¸à¥à¤¥à¤¿à¤°à¤¤à¤¾ à¤¬à¤¢à¤¼à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            "à¤¡à¤¾à¤¯à¤¨à¤¾à¤®à¤¿à¤• à¤ªà¥à¤°à¤¯à¤¾à¤¸ à¤•à¥‡ à¤¦à¥Œà¤°à¤¾à¤¨ à¤§à¤¡à¤¼ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¬à¥‡à¤¹à¤¤à¤° à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
            "à¤¹à¤¾à¤¥, à¤§à¤¡à¤¼ à¤”à¤° à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‡ à¤¸à¤®à¤¨à¥à¤µà¤¯ à¤•à¥‹ à¤¸à¤¿à¤–à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
        ] if hi else [
            "Builds upper-body pushing strength and shoulder stability.",
            "Improves trunk control during dynamic effort.",
            "Teaches better coordination between arms, torso, and hips.",
        ]
    return [
        "à¤•à¥‹à¤° à¤à¤‚à¤¡à¥à¤¯à¥‹à¤°à¥‡à¤‚à¤¸ à¤”à¤° à¤ªà¥‚à¤°à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥‡ à¤¸à¤®à¤¨à¥à¤µà¤¯ à¤•à¥‹ à¤¬à¥‡à¤¹à¤¤à¤° à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
        "à¤ªà¥‹à¤¶à¥à¤šà¤°, à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤•à¤‚à¤Ÿà¥à¤°à¥‹à¤² à¤”à¤° à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤…à¤µà¥‡à¤¯à¤°à¤¨à¥‡à¤¸ à¤¬à¤¢à¤¼à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
        "à¤¬à¥‡à¤¹à¤¤à¤° à¤²à¤¯ à¤”à¤° à¤¸à¥à¤¥à¤¿à¤°à¤¤à¤¾ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤®à¥‚à¤µ à¤•à¤°à¤¨à¥‡ à¤®à¥‡à¤‚ à¤®à¤¦à¤¦ à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
    ] if hi else [
        "Builds core endurance and full-body coordination.",
        "Improves posture, movement control, and alignment awareness.",
        "Helps you move with better rhythm and stability.",
    ]


def get_exercise_steps(exercise: str) -> List[str]:
    label = ex_label(exercise)
    profile = get_exercise_profile(exercise)
    movement_tips = ex_tips(exercise)
    hi = st.session_state.get("language", "en") == "hi"

    if exercise in ASANA_EXERCISES:
        base_steps = [
            f"{label} à¤•à¥‡ à¤²à¤¿à¤ à¤¸à¥à¤¥à¤¿à¤° à¤”à¤° à¤¸à¤‚à¤¤à¥à¤²à¤¿à¤¤ à¤¶à¥à¤°à¥à¤†à¤¤à¥€ à¤¸à¥à¤¥à¤¿à¤¤à¤¿ à¤¬à¤¨à¤¾à¤à¤‚à¥¤",
            "à¤ªà¤¹à¤²à¥‡ à¤°à¥€à¤¢à¤¼ à¤²à¤‚à¤¬à¥€ à¤•à¤°à¥‡à¤‚, à¤«à¤¿à¤° à¤¬à¤¿à¤¨à¤¾ à¤œà¤¼à¤¬à¤°à¤¦à¤¸à¥à¤¤à¥€ à¤•à¤¿à¤ à¤§à¥€à¤°à¥‡-à¤§à¥€à¤°à¥‡ à¤ªà¥‹à¤œà¤¼ à¤®à¥‡à¤‚ à¤œà¤¾à¤à¤à¥¤",
            movement_tips[0],
            "20 à¤¸à¥‡ 45 à¤¸à¥‡à¤•à¤‚à¤¡ à¤¤à¤• à¤¶à¤¾à¤‚à¤¤ à¤¨à¤¾à¤• à¤•à¥€ à¤¸à¤¾à¤à¤¸ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤°à¥à¤•à¥‡à¤‚, à¤«à¤¿à¤° à¤§à¥€à¤°à¥‡ à¤¬à¤¾à¤¹à¤° à¤†à¤à¤à¥¤",
        ] if hi else [
            f"Start in a steady setup for {label}, with your feet, hands, or seat placed evenly.",
            "Lengthen the spine first, then move into the pose gradually without forcing range.",
            movement_tips[0],
            "Hold for 20 to 45 seconds with calm nasal breathing, then exit slowly and reset.",
        ]
        return base_steps

    if profile == "squat":
        return [
            f"{label} à¤•à¥‡ à¤²à¤¿à¤ à¤¸à¥à¤¥à¤¿à¤° à¤¸à¥à¤Ÿà¤¾à¤‚à¤¸ à¤”à¤° à¤¬à¥à¤°à¥‡à¤¸à¥‡à¤¡ à¤•à¥‹à¤° à¤•à¥‡ à¤¸à¤¾à¤¥ à¤¸à¥‡à¤Ÿ à¤¹à¥‹à¤‚à¥¤",
            "à¤°à¥‡à¤ª à¤•à¥€ à¤¶à¥à¤°à¥à¤†à¤¤ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤ªà¥€à¤›à¥‡ à¤­à¥‡à¤œà¤•à¤° à¤¯à¤¾ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤¨à¥€à¤šà¥‡ à¤œà¤¾à¤•à¤° à¤•à¤°à¥‡à¤‚à¥¤",
            movement_tips[0],
            "à¤¸à¥à¤®à¥‚à¤¦ à¤¤à¤°à¥€à¤•à¥‡ à¤¸à¥‡ à¤¶à¥à¤°à¥à¤†à¤¤à¥€ à¤¸à¥à¤¥à¤¿à¤¤à¤¿ à¤®à¥‡à¤‚ à¤²à¥Œà¤Ÿà¥‡à¤‚ à¤”à¤° à¤«à¥‰à¤°à¥à¤® à¤¬à¤¿à¤—à¤¡à¤¼à¤¨à¥‡ à¤ªà¤° à¤¸à¥‡à¤Ÿ à¤°à¥‹à¤•à¥‡à¤‚à¥¤",
        ] if hi else [
            f"Set up for {label} with a stable stance and a braced core.",
            "Begin the rep by sitting the hips back or lowering with control.",
            movement_tips[0],
            "Return to the start position smoothly and stop the set when form fades.",
        ]
    if profile == "pushup":
        return [
            f"{label} à¤•à¥‡ à¤²à¤¿à¤ à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‹ à¤µà¥à¤¯à¤µà¤¸à¥à¤¥à¤¿à¤¤ à¤”à¤° à¤•à¥‹à¤° à¤•à¥‹ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤•à¤°à¤•à¥‡ à¤¸à¥‡à¤Ÿ à¤¹à¥‹à¤‚à¥¤",
            "à¤ªà¥‚à¤°à¥‡ à¤°à¥‡à¤ª à¤®à¥‡à¤‚ à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤à¤• à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤²à¤¾à¤‡à¤¨ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤",
            movement_tips[0],
            "à¤¹à¤° à¤°à¥‡à¤ª à¤•à¥‡ à¤¬à¤¾à¤¦ à¤¸à¤¾à¤«à¤¼ à¤°à¥€à¤¸à¥‡à¤Ÿ à¤•à¤°à¥‡à¤‚, à¤«à¤¿à¤° à¤…à¤—à¤²à¤¾ à¤°à¥‡à¤ª à¤¶à¥à¤°à¥‚ à¤•à¤°à¥‡à¤‚à¥¤",
        ] if hi else [
            f"Set up for {label} with shoulders organized and the core switched on.",
            "Move through the rep in one controlled line without rushing the tempo.",
            movement_tips[0],
            "Finish each rep with a clean reset before starting the next one.",
        ]
    return [
        f"{label} à¤•à¥‡ à¤²à¤¿à¤ à¤à¤¸à¥€ à¤¸à¥à¤¥à¤¿à¤¤à¤¿ à¤²à¥‡à¤‚ à¤œà¤¿à¤¸à¥‡ à¤†à¤ª à¤¬à¤¿à¤¨à¤¾ à¤¡à¤—à¤®à¤—à¤¾à¤ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤•à¤° à¤¸à¤•à¥‡à¤‚à¥¤",
        "à¤®à¥‚à¤µ à¤•à¤°à¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤§à¤¡à¤¼ à¤•à¥‹ à¤¹à¤²à¥à¤•à¤¾ à¤¬à¥à¤°à¥‡à¤¸à¥‡à¤¡ à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤ªà¥‹à¤¶à¥à¤šà¤° à¤¸à¥‡à¤Ÿ à¤•à¤°à¥‡à¤‚à¥¤",
        movement_tips[0],
        "à¤¸à¥à¤®à¥‚à¤¦ à¤¸à¤¾à¤à¤¸ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤¹à¥‹à¤²à¥à¤¡ à¤¯à¤¾ à¤°à¥‡à¤ªà¥à¤¸ à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤¤à¤•à¤¨à¥€à¤• à¤Ÿà¥‚à¤Ÿà¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤°à¥à¤•à¥‡à¤‚à¥¤",
    ] if hi else [
        f"Set up for {label} in a position you can control without wobbling.",
        "Brace gently through the trunk and organize your posture before moving.",
        movement_tips[0],
        "Perform the hold or reps with smooth breathing and stop before technique breaks.",
    ]


def get_camera_tip(exercise: str) -> str:
    profile = get_exercise_profile(exercise)
    hi = st.session_state.get("language", "en") == "hi"
    if profile in {"pushup", "plank"}:
        return "à¤¸à¤°à¥à¤µà¤¶à¥à¤°à¥‡à¤·à¥à¤  à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤²: à¤¸à¤¾à¤‡à¤¡ à¤µà¥à¤¯à¥‚ à¤¤à¤¾à¤•à¤¿ à¤•à¤‚à¤§à¥‡, à¤•à¥‚à¤²à¥à¤¹à¥‡ à¤”à¤° à¤Ÿà¤–à¤¨à¥‡ à¤•à¥€ à¤²à¤¾à¤‡à¤¨ à¤¸à¤¾à¤«à¤¼ à¤¦à¤¿à¤–à¥‡à¥¤" if hi else "Best camera angle: side view so shoulder, hip, and ankle alignment stay visible."
    if profile == "squat":
        return "à¤¸à¤°à¥à¤µà¤¶à¥à¤°à¥‡à¤·à¥à¤  à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤²: 45-à¤¡à¤¿à¤—à¥à¤°à¥€ à¤«à¥à¤°à¤‚à¤Ÿ à¤¯à¤¾ à¤¸à¤¾à¤‡à¤¡ à¤µà¥à¤¯à¥‚ à¤¤à¤¾à¤•à¤¿ à¤•à¥‚à¤²à¥à¤¹à¥‡, à¤˜à¥à¤Ÿà¤¨à¥‡ à¤”à¤° à¤Ÿà¤–à¤¨à¥‡ à¤•à¥€ à¤—à¤¹à¤°à¤¾à¤ˆ à¤¦à¤¿à¤–à¥‡à¥¤" if hi else "Best camera angle: 45-degree front or side view so hip, knee, and ankle depth can be seen."
    return "à¤¸à¤°à¥à¤µà¤¶à¥à¤°à¥‡à¤·à¥à¤  à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤²: à¤ªà¥‚à¤°à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤«à¥à¤°à¥‡à¤® à¤•à¥‡ à¤¬à¥€à¤š à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤ªà¥‚à¤°à¤¾ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤¦à¤¿à¤–à¤¨à¥‡ à¤œà¤¿à¤¤à¤¨à¥€ à¤¦à¥‚à¤°à¥€ à¤°à¤–à¥‡à¤‚à¥¤" if hi else "Best camera angle: keep your full body centered in frame with enough distance to see the whole movement."


def build_exercise_video_url(exercise: str) -> str:
    label = EXERCISE_COPY[exercise]["label"]
    query = f"{label} {get_exercise_category(exercise)} exercise tutorial"
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def build_exercise_image_data_uri(exercise: str) -> str:
    label = EXERCISE_COPY[exercise]["label"]
    category = get_exercise_category(exercise)
    accent = "#2f8f63" if category == "Asana" else "#d86f45"
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='1200' height='720' viewBox='0 0 1200 720'>
      <defs>
        <linearGradient id='bg' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#fff8ee'/>
          <stop offset='100%' stop-color='#f0e1cf'/>
        </linearGradient>
      </defs>
      <rect width='1200' height='720' rx='40' fill='url(#bg)'/>
      <circle cx='980' cy='140' r='120' fill='{accent}' opacity='0.15'/>
      <circle cx='180' cy='570' r='150' fill='{accent}' opacity='0.10'/>
      <rect x='90' y='90' width='1020' height='540' rx='28' fill='white' opacity='0.72' stroke='{accent}' stroke-opacity='0.28' />
      <text x='130' y='190' font-family='Arial, sans-serif' font-size='30' fill='{accent}' letter-spacing='4'>{category.upper()} GUIDE</text>
      <text x='130' y='290' font-family='Arial, sans-serif' font-size='72' font-weight='700' fill='#24180f'>{label}</text>
      <text x='130' y='360' font-family='Arial, sans-serif' font-size='28' fill='#6e5a47'>Use the tutorial panel for step-by-step instructions, benefits, and posture tips.</text>
      <text x='130' y='450' font-family='Arial, sans-serif' font-size='24' fill='#6e5a47'>Focused on alignment, breathing, and controlled movement quality.</text>
      <rect x='130' y='500' width='270' height='66' rx='18' fill='{accent}' opacity='0.92'/>
      <text x='165' y='542' font-family='Arial, sans-serif' font-size='28' font-weight='700' fill='white'>Trainer Demo Card</text>
    </svg>
    """.strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def get_exercise_details(exercise: str) -> Dict[str, object]:
    profile = get_exercise_profile(exercise)
    category = get_exercise_category(exercise)
    label = ex_label(exercise)
    return {
        "label": label,
        "category": category,
        "overview": (
            {
                "squat": "à¤à¤• à¤¨à¤¿à¤šà¤²à¥‡ à¤¶à¤°à¥€à¤° à¤•à¤¾ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤œà¥‹ à¤•à¥‚à¤²à¥à¤¹à¥‡, à¤˜à¥à¤Ÿà¤¨à¥‡, à¤ªà¥‹à¤¶à¥à¤šà¤° à¤”à¤° à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤ªà¤° à¤•à¤¾à¤® à¤•à¤°à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "pushup": "à¤à¤• à¤Šà¤ªà¤°à¥€ à¤¯à¤¾ à¤ªà¥‚à¤°à¥‡ à¤¶à¤°à¥€à¤° à¤•à¤¾ à¤ªà¥ˆà¤Ÿà¤°à¥à¤¨ à¤œà¥‹ à¤ªà¥à¤¶à¤¿à¤‚à¤— à¤¸à¥à¤Ÿà¥à¤°à¥‡à¤‚à¤¥ à¤”à¤° à¤Ÿà¥à¤°à¤‚à¤• à¤¸à¥à¤Ÿà¥‡à¤¬à¤¿à¤²à¤¿à¤Ÿà¥€ à¤¬à¤¨à¤¾à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "plank": "à¤à¤• à¤¸à¥à¤¥à¤¿à¤°à¤¤à¤¾-à¤†à¤§à¤¾à¤°à¤¿à¤¤ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤¯à¤¾ à¤ªà¥‹à¤œà¤¼ à¤œà¥‹ à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ, à¤¸à¤‚à¤¤à¥à¤²à¤¨ à¤”à¤° à¤•à¥‹à¤° à¤•à¤‚à¤Ÿà¥à¤°à¥‹à¤² à¤•à¥‹ à¤šà¥à¤¨à¥Œà¤¤à¥€ à¤¦à¥‡à¤¤à¤¾ à¤¹à¥ˆà¥¤",
                "generic": "à¤à¤• à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤ªà¥ˆà¤Ÿà¤°à¥à¤¨ à¤œà¤¹à¤¾à¤ à¤ªà¥‹à¤¶à¥à¤šà¤°, à¤œà¥‰à¤‡à¤‚à¤Ÿ à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤”à¤° à¤¸à¤¾à¤à¤¸ à¤•à¥€ à¤—à¥à¤£à¤µà¤¤à¥à¤¤à¤¾ à¤®à¤¹à¤¤à¥à¤µà¤ªà¥‚à¤°à¥à¤£ à¤¹à¥‹à¤¤à¥€ à¤¹à¥ˆà¥¤",
            } if st.session_state.get("language", "en") == "hi" else PROFILE_OVERVIEWS
        )[profile],
        "steps": get_exercise_steps(exercise),
        "benefits": get_exercise_benefits(exercise),
        "camera_tip": get_camera_tip(exercise),
        "image_uri": build_exercise_image_data_uri(exercise),
        "video_url": build_exercise_video_url(exercise),
    }


def build_hero_image_data_uri() -> str:
    hi = st.session_state.get("language", "en") == "hi"
    brand = "à¤«à¤¿à¤Ÿà¤¨à¥‡à¤¸ à¤•à¥‹à¤š" if hi else "FITNESS COACH"
    headline = "à¤¬à¥‡à¤¹à¤¤à¤° à¤®à¥‚à¤µ à¤•à¤°à¥‡à¤‚à¥¤ à¤¸à¤®à¤à¤¦à¤¾à¤°à¥€ à¤¸à¥‡ à¤Ÿà¥à¤°à¥‡à¤¨ à¤•à¤°à¥‡à¤‚à¥¤" if hi else "Move better. Train smarter."
    sub1 = "à¤²à¤¾à¤‡à¤µ à¤ªà¥‹à¥› à¤Ÿà¥à¤°à¥ˆà¤•à¤¿à¤‚à¤—, à¤—à¤¾à¤‡à¤¡à¥‡à¤¡ à¤µà¤°à¥à¤•à¤†à¤‰à¤Ÿ à¤”à¤° à¤…à¤ªà¤²à¥‹à¤¡ à¤°à¤¿à¤µà¥à¤¯à¥‚," if hi else "Live pose tracking, guided workouts, upload review,"
    sub2 = "à¤”à¤° à¤°à¥‹à¤œà¤¼à¤¾à¤¨à¤¾ à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤•à¥‡ à¤²à¤¿à¤ à¤‰à¤ªà¤¯à¥‹à¤—à¥€ à¤¸à¤‚à¤•à¥‡à¤¤à¥¤" if hi else "and practical cues for stronger daily training."
    support = "à¤ªà¤°à¥à¤¸à¤¨à¤² à¤¸à¤ªà¥‹à¤°à¥à¤Ÿ" if hi else "PERSONALIZED SUPPORT"
    support_title = "à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤«à¥€à¤¡à¤¬à¥ˆà¤•" if hi else "Movement feedback"
    support_1 = "à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ à¤¸à¤‚à¤•à¥‡à¤¤, à¤…à¤ªà¤²à¥‹à¤¡ à¤°à¤¿à¤µà¥à¤¯à¥‚, à¤”à¤°" if hi else "Exercise cues, upload review, and"
    support_2 = "à¤à¤• à¤¹à¥€ à¤œà¤—à¤¹ à¤¸à¥à¤ªà¤·à¥à¤Ÿ à¤ªà¥‹à¤¶à¥à¤šà¤° à¤—à¤¾à¤‡à¤¡à¥‡à¤‚à¤¸à¥¤" if hi else "clear posture guidance in one place."
    visual_title = "à¤¬à¥‡à¤¹à¤¤à¤° à¤«à¥‰à¤°à¥à¤®, à¤¸à¤¾à¤«à¤¼ à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤—" if hi else "Better form, clearer training"
    visual_sub = "à¤«à¥‹à¤•à¤¸à¥à¤¡ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤ªà¥à¤°à¥ˆà¤•à¥à¤Ÿà¤¿à¤¸ à¤•à¥‡ à¤²à¤¿à¤ à¤à¤• à¤†à¤¸à¤¾à¤¨ à¤µà¤¿à¤œà¤¼à¥à¤…à¤² à¤—à¤¾à¤‡à¤¡" if hi else "A simple visual guide for focused movement practice"
    focus = "à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤«à¥‹à¤•à¤¸" if hi else "TRAINING FOCUS"
    focus_title = "à¤«à¥‰à¤°à¥à¤® à¤¸à¤‚à¤•à¥‡à¤¤ à¤”à¤° à¤—à¤¾à¤‡à¤¡à¥‡à¤¡ à¤°à¤¿à¤µà¥à¤¯à¥‚" if hi else "Form cues and guided review"
    focus_sub = "à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ à¤¸à¥‡à¤Ÿà¤…à¤ª, à¤²à¤¾à¤‡à¤µ à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤«à¥€à¤¡à¤¬à¥ˆà¤•, à¤”à¤° à¤…à¤ªà¤²à¥‹à¤¡ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£à¥¤" if hi else "Exercise setup help, live alignment feedback, and upload analysis."
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='1200' height='900' viewBox='0 0 1200 900'>
      <defs>
        <linearGradient id='bg' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#fff4e8'/>
          <stop offset='100%' stop-color='#f0d8c2'/>
        </linearGradient>
        <linearGradient id='card' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#ffffff' stop-opacity='0.96'/>
          <stop offset='100%' stop-color='#f8ede1' stop-opacity='0.95'/>
        </linearGradient>
        <linearGradient id='screen' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#fffdf9'/>
          <stop offset='100%' stop-color='#f6eadf'/>
        </linearGradient>
      </defs>
      <rect width='1200' height='900' rx='42' fill='url(#bg)'/>
      <circle cx='960' cy='140' r='150' fill='#d86f45' opacity='0.16'/>
      <circle cx='215' cy='735' r='190' fill='#216d63' opacity='0.14'/>
      <rect x='90' y='90' width='1020' height='720' rx='32' fill='url(#card)' stroke='#d7b89d' stroke-opacity='0.6'/>
      <text x='130' y='180' font-family='Arial, sans-serif' font-size='28' fill='#a44c23' letter-spacing='4'>{brand}</text>
      <text x='130' y='280' font-family='Arial, sans-serif' font-size='68' font-weight='700' fill='#20150f'>{headline}</text>
      <text x='130' y='345' font-family='Arial, sans-serif' font-size='28' fill='#5c4a3b'>{sub1}</text>
      <text x='130' y='385' font-family='Arial, sans-serif' font-size='28' fill='#5c4a3b'>{sub2}</text>
      <rect x='130' y='430' width='360' height='122' rx='26' fill='#fffaf4' stroke='#d8bba0'/>
      <text x='156' y='486' font-family='Arial, sans-serif' font-size='18' font-weight='700' fill='#a44c23' letter-spacing='2'>{support}</text>
      <text x='156' y='528' font-family='Arial, sans-serif' font-size='31' font-weight='700' fill='#20150f'>{support_title}</text>
      <text x='156' y='558' font-family='Arial, sans-serif' font-size='20' fill='#5c4a3b'>{support_1}</text>
      <text x='156' y='585' font-family='Arial, sans-serif' font-size='20' fill='#5c4a3b'>{support_2}</text>
      <rect x='650' y='170' width='360' height='470' rx='34' fill='#fff7ef' stroke='#d8bba0'/>
      <rect x='676' y='198' width='308' height='414' rx='24' fill='url(#screen)' stroke='#ead6c5'/>
      <text x='698' y='252' font-family='Arial, sans-serif' font-size='28' font-weight='700' fill='#20150f'>{visual_title}</text>
      <text x='698' y='288' font-family='Arial, sans-serif' font-size='19' fill='#6f5b4b'>{visual_sub}</text>
      <circle cx='830' cy='392' r='58' fill='#f0ddd0'/>
      <rect x='782' y='450' width='96' height='146' rx='42' fill='#d86f45'/>
      <rect x='724' y='467' width='52' height='138' rx='24' fill='#216d63'/>
      <rect x='886' y='467' width='52' height='138' rx='24' fill='#216d63'/>
      <rect x='776' y='600' width='42' height='108' rx='20' fill='#20150f'/>
      <rect x='842' y='600' width='42' height='108' rx='20' fill='#20150f'/>
      <rect x='130' y='590' width='480' height='164' rx='28' fill='#fffaf4' stroke='#d8bba0'/>
      <text x='165' y='640' font-family='Arial, sans-serif' font-size='22' font-weight='700' fill='#a44c23' letter-spacing='2'>{focus}</text>
      <text x='165' y='691' font-family='Arial, sans-serif' font-size='30' font-weight='700' fill='#20150f'>{focus_title}</text>
      <text x='165' y='728' font-family='Arial, sans-serif' font-size='23' fill='#5c4a3b'>{focus_sub}</text>
    </svg>
    """.strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def create_pose_tracker(static_image_mode: bool) -> object:
    return Pose(
        static_image_mode=static_image_mode,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


class PoseCoach:
    def __init__(self, exercise: str, enable_rep_count: bool = False):
        self.exercise = exercise
        self.enable_rep_count = enable_rep_count
        self.rep_count = 0
        self.form_score = 0
        self.hold_time = 0.0
        self.stage = "up"
        self.plank_start_time: Optional[float] = None

    def set_exercise(self, exercise: str) -> None:
        if self.exercise != exercise:
            self.exercise = exercise
            self.rep_count = 0
            self.form_score = 0
            self.hold_time = 0.0
            self.stage = "up"
            self.plank_start_time = None

    def analyze(self, landmarks, elapsed: Optional[float] = None) -> Dict[str, object]:
        points = side_points(landmarks)
        profile = get_exercise_profile(self.exercise)
        if profile == "squat":
            result = self._analyze_squat(points)
        elif profile == "pushup":
            result = self._analyze_pushup(points)
        elif profile == "plank":
            result = self._analyze_plank(points, elapsed=elapsed)
        else:
            result = self._analyze_generic(points)

        self.form_score = float(result["score"])
        return result

    def reset_tracking(self) -> None:
        self.form_score = 0
        self.hold_time = 0.0
        self.plank_start_time = None

    def stats(self, result: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        feedback = result or default_result(
            self.exercise,
            "à¤¬à¥‰à¤¡à¥€ à¤²à¥ˆà¤‚à¤¡à¤®à¤¾à¤°à¥à¤•à¥à¤¸ à¤•à¤¾ à¤‡à¤‚à¤¤à¤œà¤¼à¤¾à¤° à¤¹à¥ˆ..." if st.session_state.get("language", "en") == "hi" else "Waiting for body landmarks...",
        )
        return {
            "exercise": ex_label(self.exercise),
            "reps": self.rep_count if self.enable_rep_count else 0,
            "score": round(max(0, self.form_score)),
            "hold_time": self.hold_time,
            "title": feedback["title"],
            "message": feedback["message"],
            "tips": feedback["tips"],
        }

    def _analyze_squat(self, points: Dict[str, object]) -> Dict[str, object]:
        knee_angle = angle_between(points["hip"], points["knee"], points["ankle"])
        torso_angle = angle_between(points["shoulder"], points["hip"], points["knee"])
        hip_depth = None
        if points["hip"] and points["knee"]:
            hip_depth = points["hip"].y - points["knee"].y

        if knee_angle is None or torso_angle is None or hip_depth is None:
            return default_result("squat", "à¤¥à¥‹à¤¡à¤¼à¤¾ à¤ªà¥€à¤›à¥‡ à¤œà¤¾à¤à¤ à¤¤à¤¾à¤•à¤¿ à¤•à¥‚à¤²à¥à¤¹à¥‡, à¤˜à¥à¤Ÿà¤¨à¥‡ à¤”à¤° à¤Ÿà¤–à¤¨à¥‡ à¤¸à¤¾à¤«à¤¼ à¤¦à¤¿à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Step back so your hips, knees, and ankles are all visible.")

        if self.stage == "up" and knee_angle < 105 and hip_depth > -0.04:
            self.stage = "down"
        elif self.stage == "down" and knee_angle > 155:
            self.stage = "up"
            if self.enable_rep_count:
                self.rep_count += 1

        score = 100
        cues: List[str] = []

        if knee_angle > 115:
            score -= 25
            cues.append("à¤¬à¥‡à¤¹à¤¤à¤° à¤¸à¥à¤•à¥à¤µà¤¾à¤Ÿ à¤—à¤¹à¤°à¤¾à¤ˆ à¤•à¥‡ à¤²à¤¿à¤ à¤¥à¥‹à¤¡à¤¼à¤¾ à¤”à¤° à¤¨à¥€à¤šà¥‡ à¤¬à¥ˆà¤ à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Try sitting deeper to reach better squat depth.")

        if torso_angle < 45:
            score -= 20
            cues.append("à¤†à¤—à¥‡ à¤à¥à¤•à¤¨à¥‡ à¤¸à¥‡ à¤¬à¤šà¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤›à¤¾à¤¤à¥€ à¤¥à¥‹à¤¡à¤¼à¤¾ à¤Šà¤ªà¤° à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Lift your chest a little to avoid folding forward.")

        if points["knee"] and points["ankle"] and abs(points["knee"].x - points["ankle"].x) > 0.09:
            score -= 15
            cues.append("à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‡ à¤Šà¤ªà¤° à¤…à¤§à¤¿à¤• à¤¸à¥€à¤§à¤¾ à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Keep your knees stacked more directly over your feet.")

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("à¤…à¤šà¥à¤›à¤¾ à¤¸à¥à¤•à¥à¤µà¤¾à¤Ÿ à¤«à¥‰à¤°à¥à¤®à¥¤ à¤°à¤¿à¤¦à¥à¤® à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Nice squat mechanics. Keep the rhythm steady."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }

    def _analyze_pushup(self, points: Dict[str, object]) -> Dict[str, object]:
        elbow_angle = angle_between(points["shoulder"], points["elbow"], points["wrist"])
        body_angle = angle_between(points["shoulder"], points["hip"], points["ankle"])

        if elbow_angle is None or body_angle is None:
            return default_result(
                "pushup",
                "à¤•à¥ˆà¤®à¤°à¥‡ à¤•à¥€ à¤“à¤° à¤¸à¤¾à¤‡à¤¡ à¤®à¥‡à¤‚ à¤°à¤¹à¥‡à¤‚ à¤¤à¤¾à¤•à¤¿ à¤•à¤‚à¤§à¤¾, à¤•à¥‹à¤¹à¤¨à¥€ à¤”à¤° à¤•à¤²à¤¾à¤ˆ à¤¦à¤¿à¤–à¥‡à¥¤" if st.session_state.get("language", "en") == "hi" else "Turn sideways to the camera so your shoulder, elbow, and wrist are visible.",
            )

        if self.stage == "up" and elbow_angle < 95:
            self.stage = "down"
        elif self.stage == "down" and elbow_angle > 155:
            self.stage = "up"
            if self.enable_rep_count:
                self.rep_count += 1

        score = 100
        cues: List[str] = []

        if body_angle < 150:
            score -= 30
            cues.append("à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤•à¤‚à¤§à¥‹à¤‚ à¤”à¤° à¤Ÿà¤–à¤¨à¥‹à¤‚ à¤•à¥€ à¤²à¤¾à¤‡à¤¨ à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Keep your hips in line with your shoulders and ankles.")

        if elbow_angle > 110 and self.stage == "down":
            score -= 20
            cues.append("à¤¬à¥‡à¤¹à¤¤à¤° à¤°à¥‡à¤ª à¤•à¥‡ à¤²à¤¿à¤ à¤¥à¥‹à¤¡à¤¼à¤¾ à¤”à¤° à¤¨à¥€à¤šà¥‡ à¤œà¤¾à¤à¤à¥¤" if st.session_state.get("language", "en") == "hi" else "Lower a little deeper for a stronger rep.")

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("à¤…à¤šà¥à¤›à¤¾ à¤ªà¥à¤¶-à¤…à¤ª à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿà¥¤ à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¤•à¤° à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Solid push-up alignment. Keep your core braced."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }

    def _analyze_plank(self, points: Dict[str, object], elapsed: Optional[float] = None) -> Dict[str, object]:
        body_angle = angle_between(points["shoulder"], points["hip"], points["ankle"])
        hip_height = None
        if points["hip"] and points["shoulder"]:
            hip_height = points["hip"].y - points["shoulder"].y

        if body_angle is None or hip_height is None:
            self.plank_start_time = None
            self.hold_time = 0.0
            return default_result("plank", "à¤¸à¤¾à¤‡à¤¡ à¤®à¥‡à¤‚ à¤°à¤¹à¥‡à¤‚ à¤”à¤° à¤ªà¥‚à¤°à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤«à¥à¤°à¥‡à¤® à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Turn sideways and keep your whole body in frame.")

        score = 100
        cues: List[str] = []

        if body_angle < 158:
            score -= 30
            cues.append("à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤”à¤° à¤•à¥‹à¤° à¤•à¥‹ à¤¸à¤•à¥à¤°à¤¿à¤¯ à¤•à¤°à¤•à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥€ à¤²à¤¾à¤‡à¤¨ à¤¸à¥€à¤§à¥€ à¤•à¤°à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Straighten your body line by lifting through the hips and core.")

        if hip_height > 0.02:
            score -= 18
            cues.append("à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤•à¤‚à¤§à¥‹à¤‚ à¤•à¥‡ à¤¸à¥à¤¤à¤° à¤¸à¥‡ à¤¨à¥€à¤šà¥‡ à¤¨ à¤—à¤¿à¤°à¤¨à¥‡ à¤¦à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Avoid letting your hips sag below shoulder level.")

        if hip_height < -0.10:
            score -= 18
            cues.append("à¤«à¥à¤²à¥ˆà¤Ÿ à¤ªà¥à¤²à¥ˆà¤‚à¤• à¤¬à¤¨à¤¾à¤ à¤°à¤–à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤•à¥‹ à¤¥à¥‹à¤¡à¤¼à¤¾ à¤¨à¥€à¤šà¥‡ à¤²à¤¾à¤à¤à¥¤" if st.session_state.get("language", "en") == "hi" else "Lower your hips slightly to maintain a flat plank.")

        if score >= 80:
            if elapsed is not None and elapsed > 0:
                self.hold_time += elapsed
            else:
                if self.plank_start_time is None:
                    self.plank_start_time = time.time()
                self.hold_time = time.time() - self.plank_start_time
        else:
            self.plank_start_time = None
            self.hold_time = 0.0

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("à¤®à¤œà¤¬à¥‚à¤¤ à¤ªà¥à¤²à¥ˆà¤‚à¤• à¤²à¤¾à¤‡à¤¨à¥¤ à¤¸à¤¾à¤à¤¸ à¤²à¥‡à¤¤à¥‡ à¤°à¤¹à¥‡à¤‚ à¤”à¤° à¤•à¥‹à¤° à¤•à¥‹ à¤•à¤¸à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Strong plank line. Keep breathing and stay braced."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }

    def _analyze_generic(self, points: Dict[str, object]) -> Dict[str, object]:
        shoulder = points["shoulder"]
        hip = points["hip"]
        knee = points["knee"]
        ankle = points["ankle"]
        body_angle = angle_between(shoulder, hip, ankle)
        knee_angle = angle_between(hip, knee, ankle)

        if not shoulder or not hip:
            return default_result(
                self.exercise,
                "à¤ªà¥‚à¤°à¤¾ à¤¶à¤°à¥€à¤° à¤¦à¤¿à¤–à¤¾à¤à¤ à¤¤à¤¾à¤•à¤¿ à¤ªà¥‹à¤¶à¥à¤šà¤° à¤•à¤¾ à¤¬à¥‡à¤¹à¤¤à¤° à¤†à¤•à¤²à¤¨ à¤¹à¥‹ à¤¸à¤•à¥‡à¥¤" if st.session_state.get("language", "en") == "hi" else "Keep your full body visible so the trainer can assess your posture more clearly.",
            )

        score = 100
        cues: List[str] = []

        if body_angle is not None and body_angle < 145:
            score -= 25
            cues.append("à¤§à¤¡à¤¼ à¤•à¥‹ à¤²à¤‚à¤¬à¤¾ à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤•à¥‚à¤²à¥à¤¹à¥‹à¤‚ à¤ªà¤° à¤¢à¤¹à¤¨à¥‡ à¤¸à¥‡ à¤¬à¤šà¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Lengthen through the torso and avoid collapsing through the hips.")

        if knee_angle is not None and knee_angle < 120:
            score -= 15
            cues.append("à¤ªà¥ˆà¤°à¥‹à¤‚ à¤•à¥‹ à¤¸à¥à¤¥à¤¿à¤° à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤˜à¥à¤Ÿà¤¨à¥‹à¤‚ à¤•à¥‹ à¤…à¤¨à¤¾à¤µà¤¶à¥à¤¯à¤• à¤°à¥‚à¤ª à¤¸à¥‡ à¤…à¤‚à¤¦à¤° à¤—à¤¿à¤°à¤¨à¥‡ à¤¨ à¤¦à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Stabilize through the legs and avoid unnecessary knee collapse.")

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("à¤ªà¥‹à¤¶à¥à¤šà¤° à¤¸à¥à¤¥à¤¿à¤° à¤¦à¤¿à¤– à¤°à¤¹à¤¾ à¤¹à¥ˆà¥¤ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤¸à¤¾à¤à¤¸ à¤”à¤° à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤•à¥‡ à¤¸à¤¾à¤¥ à¤œà¤¾à¤°à¥€ à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Posture looks stable. Keep moving with controlled breathing and alignment."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }


class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.pose = create_pose_tracker(static_image_mode=False)
        self.exercise = "squat"
        self.coach = PoseCoach(self.exercise, enable_rep_count=True)
        self.lock = threading.Lock()
        self.latest_stats = self.coach.stats()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(114, 224, 174), thickness=3, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 157, 111), thickness=2, circle_radius=3),
            )
            result = self.coach.analyze(results.pose_landmarks.landmark)
        else:
            self.coach.reset_tracking()
            result = default_result(self.exercise, "Step into frame so the pose model can see your body.")

        with self.lock:
            self.latest_stats = self.coach.stats(result)

        cv2.putText(
            image,
            f"{ex_label(self.exercise)} | {t('form_score')} {round(max(0, self.coach.form_score))}%",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 250, 245),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def set_exercise(self, exercise: str) -> None:
        with self.lock:
            self.exercise = exercise
            self.coach.set_exercise(exercise)
            self.latest_stats = self.coach.stats()

    def get_stats(self) -> Dict[str, object]:
        with self.lock:
            return dict(self.latest_stats)


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="panel-kicker">{kicker}</div>
            <h3 class="section-title">{title}</h3>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feedback(stats: Dict[str, object]) -> None:
    tips_html = "".join(f"<li>{tip}</li>" for tip in stats["tips"])
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">{t("feedback")}</div>
            <h3 class="panel-title">{stats["title"]}</h3>
            <p class="panel-copy">{stats["message"]}</p>
            <ul>{tips_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_exercise_library_panel(exercise: str) -> None:
    details = get_exercise_details(exercise)
    step_items = "".join(f"<li>{step}</li>" for step in details["steps"])
    benefit_items = "".join(f"<li>{item}</li>" for item in details["benefits"])

    st.markdown(
        f"""
        <div class="guide-overview">
            <div class="panel-card">
                <div class="panel-kicker">{t("exercise_guide")}</div>
                <h3 class="panel-title">{details["label"]}</h3>
                <p class="panel-copy">{details["overview"]}</p>
            </div>
            <div class="guide-fact-grid">
                <div class="guide-fact">
                    <div class="guide-fact-label">Category</div>
                    <div class="guide-fact-value">{details["category"]}</div>
                </div>
                <div class="guide-fact">
                    <div class="guide-fact-label">Camera Setup</div>
                    <div class="guide-fact-value">{details["camera_tip"]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    media_col, info_col = st.columns([1.05, 1.2], gap="large")
    with media_col:
        st.markdown(
            f"""
            <div class="media-card panel-card">
                <div class="panel-kicker">Visual Guide</div>
                <div class="guide-image-shell">
                    <img src="{details["image_uri"]}" alt="{details["label"]} demo guide" />
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("à¤Ÿà¥à¤¯à¥‚à¤Ÿà¥‹à¤°à¤¿à¤¯à¤² à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤–à¥‹à¤²à¥‡à¤‚" if st.session_state.get("language", "en") == "hi" else "Open tutorial video", details["video_url"], use_container_width=True)
        st.markdown(
            '<p class="video-link-caption">à¤¯à¤¹ à¤à¤• à¤Ÿà¥à¤¯à¥‚à¤Ÿà¥‹à¤°à¤¿à¤¯à¤² à¤¸à¤°à¥à¤š à¤–à¥‹à¤²à¤¤à¤¾ à¤¹à¥ˆ à¤¤à¤¾à¤•à¤¿ à¤¯à¥‚à¤œà¤¼à¤° à¤œà¤²à¥à¤¦à¥€ à¤¸à¥‡ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤¡à¥‡à¤®à¥‹ à¤¦à¥‡à¤– à¤¸à¤•à¥‡à¥¤</p>' if st.session_state.get("language", "en") == "hi" else '<p class="video-link-caption">Opens a tutorial search so the user can quickly review movement demos.</p>',
            unsafe_allow_html=True,
        )

    with info_col:
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="panel-kicker">{t("how_to_perform")}</div>
                <h3 class="panel-title">{t("step_by_step")}</h3>
                <ol>{step_items}</ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="panel-kicker">{t("why_it_helps")}</div>
                <h3 class="panel-title">{t("benefits")}</h3>
                <ul>{benefit_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_training_plan(exercise: str, stats: Dict[str, object], source_label: str) -> None:
    score = int(stats["score"])
    if score >= 85:
        intensity_tip = "à¤†à¤ªà¤•à¤¾ à¤«à¥‰à¤°à¥à¤® à¤…à¤šà¥à¤›à¤¾ à¤¦à¤¿à¤– à¤°à¤¹à¤¾ à¤¹à¥ˆà¥¤ à¤§à¥€à¤®à¥‡, à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤°à¥‡à¤ªà¥à¤¸ à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤§à¥€à¤°à¥‡-à¤§à¥€à¤°à¥‡ à¤µà¥‰à¤²à¥à¤¯à¥‚à¤® à¤¬à¤¢à¤¼à¤¾à¤à¤à¥¤" if st.session_state.get("language", "en") == "hi" else "Your form looks strong here. Focus on slow, controlled reps and add volume gradually."
    elif score >= 60:
        intensity_tip = "à¤†à¤ª à¤¸à¤¹à¥€ à¤¦à¤¿à¤¶à¤¾ à¤®à¥‡à¤‚ à¤¹à¥ˆà¤‚à¥¤ à¤ªà¤¹à¤²à¥‡ à¤¸à¥à¤ªà¥€à¤¡ à¤•à¤® à¤•à¤°à¥‡à¤‚ à¤”à¤° à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤¸à¤¾à¤«à¤¼ à¤•à¤°à¥‡à¤‚, à¤«à¤¿à¤° à¤°à¥‡à¤ªà¥à¤¸ à¤¬à¤¢à¤¼à¤¾à¤à¤à¥¤" if st.session_state.get("language", "en") == "hi" else "You are close. Reduce speed, clean up alignment first, and then build reps."
    else:
        intensity_tip = "à¤‡à¤¸à¥‡ à¤¤à¤•à¤¨à¥€à¤• à¤…à¤­à¥à¤¯à¤¾à¤¸ à¤¸à¥‡à¤Ÿ à¤•à¥€ à¤¤à¤°à¤¹ à¤²à¥‡à¤‚à¥¤ à¤›à¥‹à¤Ÿà¥‡ à¤¸à¥‡à¤Ÿ, à¤§à¥€à¤®à¤¾ à¤Ÿà¥‡à¤®à¥à¤ªà¥‹ à¤”à¤° à¤¬à¤¾à¤°-à¤¬à¤¾à¤° à¤°à¥€à¤¸à¥‡à¤Ÿ à¤¸à¤¬à¤¸à¥‡ à¤…à¤§à¤¿à¤• à¤®à¤¦à¤¦ à¤•à¤°à¥‡à¤‚à¤—à¥‡à¥¤" if st.session_state.get("language", "en") == "hi" else "Use this as a technique practice set. Short sets, slow tempo, and frequent resets will help most."

    is_asana = exercise in ASANA_EXERCISES
    if is_asana:
        workout_guides = [
            "20 à¤¸à¥‡ 45 à¤¸à¥‡à¤•à¤‚à¤¡ à¤¤à¤• à¤®à¥à¤¦à¥à¤°à¤¾ à¤ªà¤•à¤¡à¤¼à¥‡à¤‚ à¤”à¤° à¤¨à¤¾à¤• à¤¸à¥‡ à¤§à¥€à¤°à¥‡ à¤¸à¤¾à¤à¤¸ à¤²à¥‡à¤‚à¥¤",
            "à¤®à¥à¤¦à¥à¤°à¤¾ à¤®à¥‡à¤‚ à¤ªà¥à¤°à¤µà¥‡à¤¶ à¤”à¤° à¤¬à¤¾à¤¹à¤° à¤¨à¤¿à¤•à¤²à¤¨à¤¾ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤£ à¤¸à¥‡ à¤•à¤°à¥‡à¤‚, à¤œà¤¼à¤¬à¤°à¤¦à¤¸à¥à¤¤à¥€ à¤¨à¤¹à¥€à¤‚à¥¤",
            "à¤¯à¤¦à¤¿ à¤…à¤²à¤¾à¤‡à¤¨à¤®à¥‡à¤‚à¤Ÿ à¤Ÿà¥‚à¤Ÿà¥‡ à¤¤à¥‹ à¤ªà¥à¤°à¥‰à¤ªà¥à¤¸ à¤¯à¤¾ à¤•à¤® à¤°à¥‡à¤‚à¤œ à¤•à¤¾ à¤‰à¤ªà¤¯à¥‹à¤— à¤•à¤°à¥‡à¤‚à¥¤",
        ] if st.session_state.get("language", "en") == "hi" else [
            "Hold the posture for 20 to 45 seconds while breathing slowly through the nose.",
            "Enter and exit the pose with control rather than forcing range of motion.",
            "Use props or a shorter range if alignment breaks before the breath stays steady.",
        ]
    else:
        workout_guides = [
            "à¤µà¤°à¥à¤•à¤¿à¤‚à¤— à¤¸à¥‡à¤Ÿà¥à¤¸ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ 1 à¤¸à¥‡ 2 à¤†à¤¸à¤¾à¤¨ à¤¤à¤•à¤¨à¥€à¤•à¥€ à¤¸à¥‡à¤Ÿ à¤•à¤°à¥‡à¤‚à¥¤",
            "3 à¤¸à¥‡à¤Ÿ, 6 à¤¸à¥‡ 12 à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤°à¥‡à¤ªà¥à¤¸ à¤•à¤°à¥‡à¤‚, à¤¯à¤¾ à¤¸à¥à¤Ÿà¥ˆà¤Ÿà¤¿à¤• à¤à¤•à¥à¤¸à¤°à¤¸à¤¾à¤‡à¤œ à¤•à¥‡ à¤²à¤¿à¤ à¤›à¥‹à¤Ÿà¥‡ à¤¹à¥‹à¤²à¥à¤¡ à¤°à¤–à¥‡à¤‚à¥¤",
            "à¤ªà¥‹à¤¶à¥à¤šà¤° à¤¬à¤¿à¤—à¤¡à¤¼à¤¨à¤¾ à¤¶à¥à¤°à¥‚ à¤¹à¥‹à¤¤à¥‡ à¤¹à¥€ à¤¸à¥‡à¤Ÿ à¤°à¥‹à¤• à¤¦à¥‡à¤‚, à¤…à¤¤à¤¿à¤°à¤¿à¤•à¥à¤¤ à¤°à¥‡à¤ªà¥à¤¸ à¤•à¥‡ à¤ªà¥€à¤›à¥‡ à¤¨ à¤­à¤¾à¤—à¥‡à¤‚à¥¤",
        ] if st.session_state.get("language", "en") == "hi" else [
            "Warm up with 1 to 2 easier technique sets before your working sets.",
            "Train 3 sets of 6 to 12 controlled reps, or shorter timed holds for static exercises.",
            "Stop the set once posture starts slipping instead of chasing extra reps.",
        ]

    exercise_guides = ex_tips(exercise)
    tips_html = "".join(f"<li>{tip}</li>" for tip in [*workout_guides, *exercise_guides])
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">{'à¤Ÿà¥à¤°à¥‡à¤¨à¤¿à¤‚à¤— à¤Ÿà¤¿à¤ªà¥à¤¸' if st.session_state.get('language','en') == 'hi' else 'Training Tips'}</div>
            <h3 class="panel-title">{f'à¤‡à¤¸ {source_label.lower()} à¤¸à¥‡ à¤¸à¥à¤§à¤¾à¤° à¤•à¥ˆà¤¸à¥‡ à¤•à¤°à¥‡à¤‚' if st.session_state.get('language','en') == 'hi' else f'How to improve from this {source_label.lower()}'}</h3>
            <p class="panel-copy">{intensity_tip}</p>
            <ul>{tips_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trainer_care() -> None:
    care_items = [
        "à¤°à¥‡à¤ªà¥à¤¸ à¤¯à¤¾ à¤²à¤‚à¤¬à¥‡ à¤¹à¥‹à¤²à¥à¤¡ à¤¶à¥à¤°à¥‚ à¤•à¤°à¤¨à¥‡ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ 5 à¤¸à¥‡ 10 à¤®à¤¿à¤¨à¤Ÿ à¤µà¤¾à¤°à¥à¤®-à¤…à¤ª à¤•à¤°à¥‡à¤‚ã€‚",
        "à¤¯à¤¦à¤¿ à¤¤à¥‡à¤œ à¤¦à¤°à¥à¤¦, à¤šà¤•à¥à¤•à¤° à¤¯à¤¾ à¤œà¥‰à¤‡à¤‚à¤Ÿ à¤…à¤¸à¥à¤¥à¤¿à¤°à¤¤à¤¾ à¤¹à¥‹ à¤¤à¥‹ à¤¤à¥à¤°à¤‚à¤¤ à¤°à¥à¤•à¥‡à¤‚à¥¤",
        "à¤ªà¤¾à¤¸ à¤®à¥‡à¤‚ à¤ªà¤¾à¤¨à¥€ à¤°à¤–à¥‡à¤‚ à¤”à¤° à¤¸à¥‡à¤Ÿà¥à¤¸ à¤•à¥‡ à¤¬à¥€à¤š à¤›à¥‹à¤Ÿà¥‡ à¤¬à¥à¤°à¥‡à¤• à¤²à¥‡à¤‚à¥¤",
        "à¤¬à¥‡à¤¹à¤¤à¤° à¤«à¥€à¤¡à¤¬à¥ˆà¤• à¤•à¥‡ à¤²à¤¿à¤ à¤à¤¸à¤¾ à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤² à¤°à¤–à¥‡à¤‚ à¤œà¤¿à¤¸à¤®à¥‡à¤‚ à¤ªà¥‚à¤°à¤¾ à¤¶à¤°à¥€à¤° à¤¦à¤¿à¤–à¥‡à¥¤",
        "à¤¸à¥à¤ªà¥€à¤¡ à¤¯à¤¾ à¤°à¥‡à¤ªà¥à¤¸ à¤¸à¥‡ à¤ªà¤¹à¤²à¥‡ à¤¨à¤¿à¤¯à¤‚à¤¤à¥à¤°à¤¿à¤¤ à¤«à¥‰à¤°à¥à¤® à¤•à¥‹ à¤ªà¥à¤°à¤¾à¤¥à¤®à¤¿à¤•à¤¤à¤¾ à¤¦à¥‡à¤‚à¥¤",
    ] if st.session_state.get("language", "en") == "hi" else [
        "Warm up for 5 to 10 minutes before starting reps or longer holds.",
        "Stop immediately if you feel sharp pain, dizziness, or joint instability.",
        "Keep water nearby and take short recovery breaks between sets.",
        "Use a camera angle that shows your full body for more accurate feedback.",
        "Prioritize controlled form over speed, depth, or rep count.",
    ]
    care_html = "".join(f"<li>{item}</li>" for item in care_items)
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">{t("trainer_care")}</div>
            <h3 class="panel-title">{t("trainer_care_title")}</h3>
            <p class="panel-copy">{t("trainer_care_copy")}</p>
            <ul>{care_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_camera_permission_help() -> None:
    hi = st.session_state.get("language", "en") == "hi"
    message = (
        "à¤¯à¤¦à¤¿ à¤¬à¥à¤°à¤¾à¤‰à¤œà¤¼à¤° à¤®à¥‡à¤‚ `NotAllowedError: Permission denied` à¤¦à¤¿à¤–à¥‡, à¤¤à¥‹ à¤•à¥ˆà¤®à¤°à¤¾ à¤…à¤¨à¥à¤®à¤¤à¤¿ à¤¦à¥‡à¤‚, à¤¸à¤¾à¤‡à¤Ÿ à¤•à¥‹ `https://` à¤¯à¤¾ `localhost` à¤ªà¤° à¤–à¥‹à¤²à¥‡à¤‚, à¤”à¤° Zoom/Meet à¤œà¥ˆà¤¸à¥‡ à¤à¤ªà¥à¤¸ à¤®à¥‡à¤‚ à¤•à¥ˆà¤®à¤°à¤¾ à¤‰à¤ªà¤¯à¥‹à¤— à¤¬à¤‚à¤¦ à¤•à¤°à¥‡à¤‚à¥¤"
        if hi
        else "If you see `NotAllowedError: Permission denied`, allow camera access for this site, open the app on `https://` or `localhost`, and close other apps (Zoom/Meet/Teams) that may be using the camera."
    )
    st.info(message)


def analyze_pose_frame(
    image_bgr,
    exercise: str,
    pose_tracker,
    coach: PoseCoach,
    frame_interval: Optional[float] = None,
) -> tuple[object, Dict[str, object]]:
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_tracker.process(rgb_image)
    annotated = image_bgr.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(114, 224, 174), thickness=3, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 157, 111), thickness=2, circle_radius=3),
        )
        result = coach.analyze(results.pose_landmarks.landmark, elapsed=frame_interval)
    else:
        coach.reset_tracking()
        result = default_result(exercise, "à¤•à¥‹à¤ˆ à¤¸à¥à¤ªà¤·à¥à¤Ÿ à¤ªà¥‹à¥› à¤¨à¤¹à¥€à¤‚ à¤®à¤¿à¤²à¤¾à¥¤ à¤ªà¥‚à¤°à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤¦à¤¿à¤–à¤¾à¤à¤ à¤”à¤° à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤² à¤¸à¤¾à¤«à¤¼ à¤°à¤–à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "No clear pose found. Keep the whole body visible and try a cleaner angle.")

    cv2.putText(
        annotated,
        f"{ex_label(exercise)} | {t('form_score')} {round(max(0, coach.form_score))}%",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 250, 245),
        2,
        cv2.LINE_AA,
    )
    return annotated, coach.stats(result)


def analyze_uploaded_image(uploaded_file, exercise: str) -> None:
    file_bytes = uploaded_file.read()
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("à¤¯à¤¹ à¤‡à¤®à¥‡à¤œ à¤ªà¤¢à¤¼à¥€ à¤¨à¤¹à¥€à¤‚ à¤œà¤¾ à¤¸à¤•à¥€à¥¤ à¤•à¥ƒà¤ªà¤¯à¤¾ JPG, JPEG, à¤¯à¤¾ PNG à¤«à¤¼à¤¾à¤‡à¤² à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "We could not read that image. Please upload a JPG, JPEG, or PNG file.")
        return

    pose_tracker = create_pose_tracker(static_image_mode=True)
    coach = PoseCoach(exercise)
    annotated, stats = analyze_pose_frame(image_bgr, exercise, pose_tracker, coach)
    pose_tracker.close()

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="à¤®à¥‚à¤² à¤«à¥‹à¤Ÿà¥‹" if st.session_state.get("language", "en") == "hi" else "Original photo", use_container_width=True)
    with right_col:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="à¤ªà¥‹à¥› à¤¡à¤¿à¤Ÿà¥‡à¤•à¥à¤¶à¤¨ à¤ªà¤°à¤¿à¤£à¤¾à¤®" if st.session_state.get("language", "en") == "hi" else "Pose detection result", use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        render_metric(t("exercise"), stats["exercise"])
    with metric_col2:
        render_metric(t("form_score"), f"{stats['score']}%")
    with metric_col3:
        render_metric(t("hold_time"), f"{stats['hold_time']:.1f}s")

    render_feedback(stats)
    render_training_plan(exercise, stats, "à¤«à¥‹à¤Ÿà¥‹" if st.session_state.get("language", "en") == "hi" else "Photo")


def analyze_uploaded_video(uploaded_file, exercise: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(uploaded_file.read())
        input_path = input_tmp.name

    output_path = f"{input_path}.processed.mp4"
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        st.error("à¤¯à¤¹ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤–à¥à¤² à¤¨à¤¹à¥€à¤‚ à¤¸à¤•à¤¾à¥¤ à¤•à¥ƒà¤ªà¤¯à¤¾ MP4, MOV, AVI, à¤¯à¤¾ MPEG à¤«à¤¼à¤¾à¤‡à¤² à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "We could not open that video. Please upload an MP4, MOV, AVI, or MPEG file.")
        os.unlink(input_path)
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 24.0

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pose_tracker = create_pose_tracker(static_image_mode=False)
    coach = PoseCoach(exercise, enable_rep_count=True)
    progress = st.progress(0, text="à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤ªà¥‹à¤¶à¥à¤šà¤° à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£ à¤¹à¥‹ à¤°à¤¹à¤¾ à¤¹à¥ˆ..." if st.session_state.get("language", "en") == "hi" else "Analyzing video posture...")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    processed_frames = 0
    final_stats = coach.stats(default_result(exercise, "à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤¿à¤ à¤—à¤ à¤µà¤°à¥à¤•à¤†à¤‰à¤Ÿ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤•à¤¾ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£ à¤¹à¥‹ à¤°à¤¹à¤¾ à¤¹à¥ˆ..." if st.session_state.get("language", "en") == "hi" else "Analyzing uploaded workout video..."))

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated, final_stats = analyze_pose_frame(
            frame,
            exercise,
            pose_tracker,
            coach,
            frame_interval=(1.0 / fps),
        )
        writer.write(annotated)
        processed_frames += 1

        if frame_count > 0:
            progress.progress(min(processed_frames / frame_count, 1.0), text="à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤ªà¥‹à¤¶à¥à¤šà¤° à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£ à¤¹à¥‹ à¤°à¤¹à¤¾ à¤¹à¥ˆ..." if st.session_state.get("language", "en") == "hi" else "Analyzing video posture...")

    capture.release()
    writer.release()
    pose_tracker.close()
    progress.empty()

    if processed_frames == 0:
        st.error("à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤¿à¤ à¤—à¤ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤®à¥‡à¤‚ à¤ªà¤¢à¤¼à¥‡ à¤œà¤¾ à¤¸à¤•à¤¨à¥‡ à¤µà¤¾à¤²à¥‡ à¤«à¥à¤°à¥‡à¤® à¤¨à¤¹à¥€à¤‚ à¤®à¤¿à¤²à¥‡à¥¤" if st.session_state.get("language", "en") == "hi" else "The uploaded video did not contain readable frames.")
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return

    with open(output_path, "rb") as processed_file:
        video_bytes = processed_file.read()

    st.video(video_bytes)
    st.download_button(
        "à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤¿à¤¤ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤¡à¤¾à¤‰à¤¨à¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚" if st.session_state.get("language", "en") == "hi" else "Download analyzed video",
        data=video_bytes,
        file_name=f"analyzed_{uploaded_file.name.rsplit('.', 1)[0]}.mp4",
        mime="video/mp4",
    )
    os.unlink(input_path)
    os.unlink(output_path)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        render_metric(t("exercise"), final_stats["exercise"])
    with metric_col2:
        render_metric(t("form_score"), f"{final_stats['score']}%")
    with metric_col3:
        render_metric(t("reps"), str(final_stats["reps"]))
    with metric_col4:
        render_metric(t("hold_time"), f"{final_stats['hold_time']:.1f}s")

    render_feedback(final_stats)
    render_training_plan(exercise, final_stats, "à¤µà¥€à¤¡à¤¿à¤¯à¥‹" if st.session_state.get("language", "en") == "hi" else "Video")


def render_sidebar_help(exercise: str) -> None:
    exercise_copy = ex_tips(exercise)
    with st.sidebar:
        st.markdown(f"## {t('coach_settings')}")
        selected_language = st.selectbox(
            t("language"),
            options=list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda key: LANGUAGE_OPTIONS[key],
            index=list(LANGUAGE_OPTIONS.keys()).index(st.session_state["language"]),
        )
        st.session_state["language"] = selected_language
        camera_options = ["front", "back"]
        camera_labels = {
            "front": "Front Camera",
            "back": "Back Camera",
        }
        selected_camera = st.selectbox(
            "Camera",
            options=camera_options,
            format_func=lambda key: camera_labels[key],
            index=camera_options.index(st.session_state["camera_facing"]),
        )
        st.session_state["camera_facing"] = selected_camera
        st.caption("à¤•à¥‹à¤ˆ à¤®à¥‚à¤µà¤®à¥‡à¤‚à¤Ÿ à¤šà¥à¤¨à¥‡à¤‚ à¤”à¤° à¤ªà¥‚à¤°à¥‡ à¤¶à¤°à¥€à¤° à¤•à¥‹ à¤«à¥à¤°à¥‡à¤® à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚ à¤¤à¤¾à¤•à¤¿ à¤Ÿà¥à¤°à¥ˆà¤•à¤¿à¤‚à¤— à¤…à¤§à¤¿à¤• à¤¸à¥à¤¥à¤¿à¤° à¤°à¤¹à¥‡à¥¤" if st.session_state["language"] == "hi" else "Pick a movement and keep your body fully visible for more stable landmarks.")
        current_group = t("gym exercises") if exercise in GYM_EXERCISES else t("asana exercises")
        selected_group = st.selectbox(
            t("category"),
            options=[t("gym exercises"), t("asana exercises")],
            index=[t("gym exercises"), t("asana exercises")].index(current_group),
        )
        selected_options = EXERCISE_GROUPS["Gym Exercises"] if selected_group == t("gym exercises") else EXERCISE_GROUPS["Asana Exercises"]
        selected = st.selectbox(
            t("exercise"),
            options=selected_options,
            format_func=lambda key: ex_label(key),
            index=selected_options.index(exercise)
            if exercise in selected_options
            else 0,
        )
        st.session_state["exercise"] = selected

        st.markdown(f"## {t('setup_tips')}")
        for tip in exercise_copy:
            st.write(f"- {tip}")

        st.markdown(f"## {t('notes')}")
        st.caption("à¤ªà¥à¤¶-à¤…à¤ª à¤”à¤° à¤ªà¥à¤²à¥ˆà¤‚à¤• à¤•à¥‡ à¤²à¤¿à¤ à¤¸à¤¾à¤‡à¤¡ à¤•à¥ˆà¤®à¤°à¤¾ à¤à¤‚à¤—à¤² à¤¸à¤¬à¤¸à¥‡ à¤¬à¥‡à¤¹à¤¤à¤° à¤°à¤¹à¤¤à¤¾ à¤¹à¥ˆà¥¤" if st.session_state["language"] == "hi" else "Push-ups and planks work best with a side camera angle.")
        st.caption("à¤¯à¤¹ à¤à¤• à¤•à¥‹à¤šà¤¿à¤‚à¤— à¤ªà¥à¤°à¥‹à¤Ÿà¥‹à¤Ÿà¤¾à¤‡à¤ª à¤¹à¥ˆ, à¤®à¥‡à¤¡à¤¿à¤•à¤² à¤¯à¤¾ à¤¬à¤¾à¤¯à¥‹à¤®à¥ˆà¤•à¥‡à¤¨à¤¿à¤•à¤² à¤¡à¤¾à¤¯à¤—à¥à¤¨à¥‹à¤¸à¥à¤Ÿà¤¿à¤• à¤Ÿà¥‚à¤² à¤¨à¤¹à¥€à¤‚ à¤¹à¥ˆà¥¤" if st.session_state["language"] == "hi" else "This is a coaching prototype, not a medical assessment tool.")


def main() -> None:
    init_session_state()
    inject_styles()
    render_sidebar_help(st.session_state["exercise"])
    active_label = ex_label(st.session_state["exercise"])
    active_group = t("asana") if st.session_state["exercise"] in ASANA_EXERCISES else t("gym")
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-layout">
                <div class="hero-copy-block">
                    <div class="hero-eyebrow">{t("app_name")}</div>
                    <div class="hero-badge">{t("hero_badge")}</div>
                    <h1 class="hero-title">{t("hero_title")}</h1>
                    <p class="hero-text">{t("hero_text")}</p>
                    <div class="hero-meta-grid">
                        <div class="hero-meta-item">
                            <div class="hero-meta-label">{t("selected_exercise")}</div>
                            <div class="hero-meta-value">{active_label}</div>
                        </div>
                        <div class="hero-meta-item">
                            <div class="hero-meta-label">{t("training_library")}</div>
                            <div class="hero-meta-value">{t("gym_count")}</div>
                        </div>
                        <div class="hero-meta-item">
                            <div class="hero-meta-label">{t("active_category")}</div>
                            <div class="hero-meta-value">{active_group}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_intro(
        t("movement_library"),
        t("exercise_reference_title"),
        t("exercise_reference_copy"),
    )
    render_exercise_library_panel(st.session_state["exercise"])

    stats_placeholder = {
        "exercise": ex_label(st.session_state["exercise"]),
        "reps": 0,
        "score": 0,
        "hold_time": 0.0,
        "title": ex_title(st.session_state["exercise"]),
        "message": "à¤ªà¥‹à¤¶à¥à¤šà¤° à¤Ÿà¥à¤°à¥ˆà¤•à¤¿à¤‚à¤— à¤¶à¥à¤°à¥‚ à¤•à¤°à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤µà¥‡à¤¬à¤•à¥ˆà¤® à¤šà¤¾à¤²à¥‚ à¤•à¤°à¥‡à¤‚à¥¤" if st.session_state.get("language", "en") == "hi" else "Start the webcam to begin posture tracking.",
        "tips": ex_tips(st.session_state["exercise"]),
    }

    col1, col2 = st.columns([1.5, 1.0], gap="large")

    with col1:
        render_section_intro(
            t("live_coach"),
            t("webcam_posture"),
            t("webcam_posture_copy"),
        )

        facing_mode = "user" if st.session_state["camera_facing"] == "front" else "environment"
        ctx = webrtc_streamer(
            key=f"ai-fitness-trainer-{st.session_state['camera_facing']}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PoseVideoProcessor,
            media_stream_constraints={
                "video": {"facingMode": {"ideal": facing_mode}},
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.set_exercise(st.session_state["exercise"])
            stats_placeholder = ctx.video_processor.get_stats()

        st.session_state["session_state"] = "Live" if ctx.state.playing else "Idle"
        if not ctx.state.playing:
            render_camera_permission_help()

        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
        render_section_intro(
            t("upload_analysis"),
            t("photo_video_review"),
            t("photo_video_review_copy"),
        )

        photo_tab, video_tab = st.tabs([t("photo_upload"), t("video_upload")])
        with photo_tab:
            uploaded_image = st.file_uploader(
                t("upload_workout_photo"),
                type=["jpg", "jpeg", "png"],
                key="photo-uploader",
            )
            if uploaded_image:
                analyze_uploaded_image(uploaded_image, st.session_state["exercise"])

        with video_tab:
            uploaded_video = st.file_uploader(
                t("upload_workout_video"),
                type=["mp4", "mov", "avi", "mpeg"],
                key="video-uploader",
            )
            if uploaded_video:
                analyze_uploaded_video(uploaded_video, st.session_state["exercise"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        render_section_intro(
            t("session_insights"),
            t("live_metrics_title"),
            t("live_metrics_copy"),
        )
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            render_metric(t("session_state"), st.session_state["session_state"])
        with metric_col2:
            render_metric(t("exercise"), stats_placeholder["exercise"])

        metric_col3, metric_col4, metric_col5 = st.columns(3)
        with metric_col3:
            render_metric(t("reps"), str(stats_placeholder["reps"]))
        with metric_col4:
            render_metric(t("form_score"), f"{stats_placeholder['score']}%")
        with metric_col5:
            render_metric(t("hold_time"), f"{stats_placeholder['hold_time']:.1f}s")

        render_feedback(stats_placeholder)
        render_training_plan(st.session_state["exercise"], stats_placeholder, "Live session")
        render_trainer_care()

    if ctx.state.playing and ctx.video_processor:
        live_metrics = st.empty()
        live_feedback = st.empty()

        while ctx.state.playing:
            stats = ctx.video_processor.get_stats()
            live_metrics.markdown(
                f"""
                <div class="panel-card">
                    <div class="panel-kicker">Live Snapshot</div>
                    <h3 class="panel-title">{stats["exercise"]}</h3>
                    <p class="panel-copy">
                        Reps: {stats["reps"]} | Form Score: {stats["score"]}% | Hold: {stats["hold_time"]:.1f}s
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            live_feedback.markdown(
                f"""
                <div class="panel-card">
                    <div class="panel-kicker">Coach Cue</div>
                    <h3 class="panel-title">{stats["title"]}</h3>
                    <p class="panel-copy">{stats["message"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(0.25)


if __name__ == "__main__":
    main()
