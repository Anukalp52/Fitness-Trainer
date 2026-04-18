import math
import importlib
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
    "hi": "हिंदी",
}

TRANSLATIONS = {
    "app_name": {"en": "AI Fitness Trainer", "hi": "एआई फिटनेस ट्रेनर"},
    "hero_badge": {"en": "Smart Movement Guidance", "hi": "स्मार्ट मूवमेंट गाइडेंस"},
    "hero_title": {"en": "Fitness Coach", "hi": "फिटनेस कोच"},
    "hero_text": {
        "en": "Build better form with live posture feedback, guided exercise references, and upload-based review for gym training and yoga practice.",
        "hi": "लाइव पोश्चर फीडबैक, गाइडेड एक्सरसाइज रेफरेंस, और अपलोड रिव्यू के साथ जिम ट्रेनिंग और योग अभ्यास को बेहतर बनाएं।",
    },
    "selected_exercise": {"en": "Selected Exercise", "hi": "चयनित एक्सरसाइज"},
    "training_library": {"en": "Training Library", "hi": "ट्रेनिंग लाइब्रेरी"},
    "active_category": {"en": "Active Category", "hi": "सक्रिय श्रेणी"},
    "gym_count": {"en": "20 Gym + 20 Asana", "hi": "20 जिम + 20 आसन"},
    "gym": {"en": "Gym", "hi": "जिम"},
    "asana": {"en": "Asana", "hi": "आसन"},
    "movement_library": {"en": "Movement Library", "hi": "मूवमेंट लाइब्रेरी"},
    "exercise_reference_title": {"en": "Exercise reference and setup guide", "hi": "एक्सरसाइज रेफरेंस और सेटअप गाइड"},
    "exercise_reference_copy": {
        "en": "Review the selected movement before training so the camera angle, setup, and key technique cues are clear.",
        "hi": "ट्रेनिंग से पहले चयनित मूवमेंट को देखें ताकि कैमरा एंगल, सेटअप और मुख्य तकनीक संकेत स्पष्ट रहें।",
    },
    "live_coach": {"en": "Live Coach", "hi": "लाइव कोच"},
    "webcam_posture": {"en": "Webcam posture stream", "hi": "वेबकैम पोश्चर स्ट्रीम"},
    "webcam_posture_copy": {
        "en": "Use the live camera feed for real-time pose feedback, then switch to uploaded media for slower review if needed.",
        "hi": "रियल-टाइम पोज़ फीडबैक के लिए लाइव कैमरा फीड का उपयोग करें, और जरूरत होने पर अपलोड मीडिया से धीमा रिव्यू करें।",
    },
    "upload_analysis": {"en": "Upload Analysis", "hi": "अपलोड विश्लेषण"},
    "photo_video_review": {"en": "Photo or video review", "hi": "फोटो या वीडियो रिव्यू"},
    "photo_video_review_copy": {
        "en": "Upload a workout photo for a quick snapshot or a video for frame-by-frame posture review with downloadable output.",
        "hi": "तेज़ स्नैपशॉट के लिए वर्कआउट फोटो या फ्रेम-दर-फ्रेम पोश्चर रिव्यू के लिए वीडियो अपलोड करें।",
    },
    "photo_upload": {"en": "Photo Upload", "hi": "फोटो अपलोड"},
    "video_upload": {"en": "Video Upload", "hi": "वीडियो अपलोड"},
    "upload_workout_photo": {"en": "Upload a workout photo", "hi": "वर्कआउट फोटो अपलोड करें"},
    "upload_workout_video": {"en": "Upload a workout video", "hi": "वर्कआउट वीडियो अपलोड करें"},
    "session_insights": {"en": "Session Insights", "hi": "सेशन इनसाइट्स"},
    "live_metrics_title": {"en": "Live metrics and coaching", "hi": "लाइव मेट्रिक्स और कोचिंग"},
    "live_metrics_copy": {
        "en": "Monitor your current form quality, hold time, and coaching prompts while you train.",
        "hi": "ट्रेनिंग के दौरान अपने फॉर्म, होल्ड टाइम और कोचिंग संकेतों पर नज़र रखें।",
    },
    "session_state": {"en": "Session State", "hi": "सेशन स्थिति"},
    "exercise": {"en": "Exercise", "hi": "एक्सरसाइज"},
    "reps": {"en": "Reps", "hi": "रेप्स"},
    "form_score": {"en": "Form Score", "hi": "फॉर्म स्कोर"},
    "hold_time": {"en": "Hold Time", "hi": "होल्ड टाइम"},
    "feedback": {"en": "Feedback", "hi": "फीडबैक"},
    "exercise_guide": {"en": "Exercise Guide", "hi": "एक्सरसाइज गाइड"},
    "how_to_perform": {"en": "How To Perform", "hi": "कैसे करें"},
    "step_by_step": {"en": "Step by step", "hi": "स्टेप बाय स्टेप"},
    "why_it_helps": {"en": "Why It Helps", "hi": "यह क्यों उपयोगी है"},
    "benefits": {"en": "Benefits", "hi": "फायदे"},
    "trainer_care": {"en": "Trainer Care", "hi": "ट्रेनर केयर"},
    "trainer_care_title": {"en": "Train safely and consistently", "hi": "सुरक्षित और नियमित रूप से ट्रेन करें"},
    "trainer_care_copy": {
        "en": "Use the coach as a support tool, but pace your workout like a real training session: prepare first, move with control, and recover between efforts.",
        "hi": "कोच को सपोर्ट टूल की तरह उपयोग करें, लेकिन वर्कआउट को असली ट्रेनिंग सेशन की तरह करें: पहले तैयारी करें, नियंत्रण से मूव करें, और बीच में रिकवरी लें।",
    },
    "coach_settings": {"en": "Coach Settings", "hi": "कोच सेटिंग्स"},
    "language": {"en": "Language", "hi": "भाषा"},
    "category": {"en": "Category", "hi": "श्रेणी"},
    "setup_tips": {"en": "Setup Tips", "hi": "सेटअप टिप्स"},
    "notes": {"en": "Notes", "hi": "नोट्स"},
    "gym exercises": {"en": "Gym Exercises", "hi": "जिम एक्सरसाइज"},
    "asana exercises": {"en": "Asana Exercises", "hi": "आसन एक्सरसाइज"},
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get("en", key))


EXERCISE_HI = {
    "squat": {"label": "स्क्वाट", "title": "स्क्वाट विश्लेषण", "tips": ["छाती को ऊपर रखें और एड़ियों को स्थिर रखें।", "यदि संभव हो तो जांघें समानांतर के करीब आने तक नीचे जाएँ।", "घुटनों को पंजों की दिशा में रखें।"]},
    "pushup": {"label": "पुश-अप", "title": "पुश-अप विश्लेषण", "tips": ["बेहतर एल्बो ट्रैकिंग के लिए कैमरा साइड में रखें।", "कंधे, कूल्हे और टखने एक सीध में रखें।", "कोहनियों को बहुत ज़्यादा बाहर न फैलाएँ।"]},
    "plank": {"label": "प्लैंक", "title": "प्लैंक विश्लेषण", "tips": ["कोर को कसकर रखें और ग्लूट्स को सक्रिय करें।", "गर्दन को न्यूट्रल रखें और हल्का नीचे देखें।", "कंधों से टखनों तक सीधी लाइन बनाए रखें।"]},
    "lunge": {"label": "लंज", "title": "लंज विश्लेषण", "tips": ["सामने वाले घुटने को टखने के ऊपर रखें।", "आगे झुकने के बजाय सीधा नीचे जाएँ।", "छाती और कूल्हों को ऊपर रखें।"]},
    "deadlift": {"label": "डेडलिफ्ट", "title": "डेडलिफ्ट विश्लेषण", "tips": ["रीढ़ सीधी रखते हुए कूल्हों से हिंग करें।", "वज़न को पैरों के पास रखें।", "हर रेप से पहले कोर को कसें।"]},
    "shoulder_press": {"label": "शोल्डर प्रेस", "title": "शोल्डर प्रेस विश्लेषण", "tips": ["रिब्स को नीचे रखें और कमर को ज़्यादा आर्च न करें।", "प्रेस को कंधों के ऊपर सीधी लाइन में करें।", "दोनों पैरों पर संतुलन बनाए रखें।"]},
    "bicep_curl": {"label": "बाइसेप कर्ल", "title": "बाइसेप कर्ल विश्लेषण", "tips": ["कोहनियों को शरीर के पास रखें।", "वज़न उठाने के लिए शरीर को झुलाएँ नहीं।", "नीचे लाने की गति को धीरे नियंत्रित करें।"]},
    "tricep_dip": {"label": "ट्राइसेप डिप", "title": "ट्राइसेप डिप विश्लेषण", "tips": ["कंधों को कानों से दूर रखें।", "कोहनियों को बाहर फैलाने के बजाय पीछे मोड़ें।", "दर्द-रहित रेंज में नियंत्रण से मूव करें।"]},
    "pullup": {"label": "पुल-अप", "title": "पुल-अप विश्लेषण", "tips": ["कोर को सक्रिय रखते हुए स्थिर हैंग से शुरू करें।", "कंधे चढ़ाने के बजाय कोहनियों को नीचे खींचें।", "रेप्स के बीच शरीर को झूलने न दें।"]},
    "bench_press": {"label": "बेंच प्रेस", "title": "बेंच प्रेस विश्लेषण", "tips": ["कलाई को कोहनी के ऊपर रखें।", "प्रेस के दौरान ऊपरी पीठ में तनाव बनाए रखें।", "दोनों तरफ समान नियंत्रण से बार प्रेस करें।"]},
    "mountain_climber": {"label": "माउंटेन क्लाइंबर", "title": "माउंटेन क्लाइंबर विश्लेषण", "tips": ["कंधों को हथेलियों के ऊपर रखें।", "कोर को सक्रिय रखें ताकि कूल्हे स्थिर रहें।", "घुटनों को झटके से नहीं, नियंत्रण से चलाएँ।"]},
    "burpee": {"label": "बर्पी", "title": "बर्पी विश्लेषण", "tips": ["प्लैंक में कोर को कसकर रखें और हल्के से लैंड करें।", "खड़े होने से पहले पैरों को कूल्हों के नीचे लाएँ।", "मूवमेंट को जल्दबाज़ी में नहीं, स्मूद रखें।"]},
    "jumping_jack": {"label": "जंपिंग जैक", "title": "जंपिंग जैक विश्लेषण", "tips": ["हल्के से लैंड करें और घुटनों को थोड़ा मोड़ें।", "हाथ और पैरों की लय एक जैसी रखें।", "पूरे सेट में धड़ को सीधा रखें।"]},
    "glute_bridge": {"label": "ग्लूट ब्रिज", "title": "ग्लूट ब्रिज विश्लेषण", "tips": ["कूल्हे उठाने के लिए एड़ियों से दबाव दें।", "ऊपर पहुँचकर कमर मोड़ने के बजाय ग्लूट्स को कसें।", "घुटनों को सीधी दिशा में रखें।"]},
    "calf_raise": {"label": "काफ रेज़", "title": "काफ रेज़ विश्लेषण", "tips": ["पैरों के अगले हिस्से पर सीधा ऊपर उठें।", "ऊपर एक छोटा विराम लें।", "टखनों को बाहर या अंदर रोल न होने दें।"]},
    "russian_twist": {"label": "रशियन ट्विस्ट", "title": "रशियन ट्विस्ट विश्लेषण", "tips": ["सिर्फ हाथ नहीं, पसलियों से रोटेशन करें।", "पीछे झुकते हुए छाती ऊपर रखें।", "धीरे करें ताकि ऑब्लिक्स सक्रिय रहें।"]},
    "bicycle_crunch": {"label": "बाइसिकल क्रंच", "title": "बाइसिकल क्रंच विश्लेषण", "tips": ["गर्दन खींचने के बजाय धड़ को मोड़ें।", "विपरीत पैर को नियंत्रण से पूरा सीधा करें।", "लोअर बैक को हल्का सक्रिय रखें।"]},
    "side_lunge": {"label": "साइड लंज", "title": "साइड लंज विश्लेषण", "tips": ["काम करने वाले कूल्हे में पीछे बैठें।", "स्थिर पैर को पूरा जमीन पर रखें।", "साइड में जाते हुए छाती ऊपर रखें।"]},
    "high_knees": {"label": "हाई नीज़", "title": "हाई नीज़ विश्लेषण", "tips": ["घुटनों को तेज़ लेकिन नियंत्रण से ऊपर ड्राइव करें।", "धड़ सीधा रखें और पैरों पर हल्के रहें।", "रिद्म के लिए हाथों का सक्रिय स्विंग रखें।"]},
    "step_up": {"label": "स्टेप-अप", "title": "स्टेप-अप विश्लेषण", "tips": ["स्टेप या बेंच पर पूरे पैर से दबाव दें।", "ऊपर खड़े होते समय आगे न झुकें।", "नीचे लौटते समय नियंत्रण बनाए रखें।"]},
    "tadasana": {"label": "ताड़ासन", "title": "ताड़ासन विश्लेषण", "tips": ["दोनों पैरों पर समान भार रखें।", "सिर के शीर्ष से ऊपर की ओर लंबाई बनाएँ।", "छाती खुली रखते हुए कंधों को ढीला रखें।"]},
    "vrikshasana": {"label": "वृक्षासन", "title": "वृक्षासन विश्लेषण", "tips": ["खड़े पैर को मजबूती से जमीन में दबाएँ।", "उठे हुए घुटने को आगे गिरने न दें।", "संतुलन के लिए एक बिंदु पर दृष्टि रखें।"]},
    "utkatasana": {"label": "उत्कटासन", "title": "उत्कटासन विश्लेषण", "tips": ["कूल्हों को पीछे भेजें और छाती ऊपर रखें।", "पूरा भार पूरे पैर में बाँटें।", "कंधे चढ़ाए बिना हाथ लंबे रखें।"]},
    "virabhadrasana_i": {"label": "वीरभद्रासन I", "title": "वीरभद्रासन I विश्लेषण", "tips": ["सामने वाले घुटने को मोड़ें और पीछे की एड़ी टिकाएँ।", "जहाँ तक संभव हो धड़ को सीधा रखें।", "कमर दबाए बिना हाथ ऊपर उठाएँ।"]},
    "virabhadrasana_ii": {"label": "वीरभद्रासन II", "title": "वीरभद्रासन II विश्लेषण", "tips": ["सामने वाले घुटने को टखने के ऊपर रखें।", "दोनों हाथों को बराबर लंबाई में फैलाएँ।", "धड़ को पैरों के बीच संतुलित रखें।"]},
    "trikonasana": {"label": "त्रिकोणासन", "title": "त्रिकोणासन विश्लेषण", "tips": ["नीचे जाने से पहले कमर के दोनों ओर लंबाई बनाएँ।", "जहाँ संभव हो कंधों को एक लाइन में रखें।", "दोनों पैरों को सक्रिय रखें लेकिन लॉक न करें।"]},
    "adho_mukha_svanasana": {"label": "अधो मुख श्वानासन", "title": "अधो मुख श्वानासन विश्लेषण", "tips": ["हथेलियों से ज़मीन को मजबूत धक्का दें।", "रीढ़ लंबी करने के लिए कूल्हों को ऊपर उठाएँ।", "यदि पीठ गोल हो रही हो तो घुटने नरम रखें।"]},
    "bhujangasana": {"label": "भुजंगासन", "title": "भुजंगासन विश्लेषण", "tips": ["गर्दन मोड़ने के बजाय छाती को ऊपर उठाएँ।", "कोहनियों को थोड़ा मुड़ा और पसलियों के पास रखें।", "पैरों के ऊपरी हिस्से को मैट पर दबाएँ।"]},
    "setu_bandhasana": {"label": "सेतु बंधासन", "title": "सेतु बंधासन विश्लेषण", "tips": ["पैरों से दबाव देकर कूल्हों को उठाएँ।", "घुटनों को बाहर फैलने न दें।", "गर्दन पर दबाव डाले बिना छाती खोलें।"]},
    "naukasana": {"label": "नौकासन", "title": "नौकासन विश्लेषण", "tips": ["रीढ़ गोल होने से बचने के लिए छाती ऊपर रखें।", "कोर को कसें और बैठने की हड्डियों पर संतुलन रखें।", "कंपन होने पर भी साँस स्थिर रखें।"]},
    "balasana": {"label": "बालासन", "title": "बालासन विश्लेषण", "tips": ["कूल्हों को एड़ियों की ओर आराम से नीचे जाने दें।", "कंधों और जबड़े को ढीला रखें।", "साँस के साथ रीढ़ को धीरे लंबा करें।"]},
    "phalakasana": {"label": "फलकासन", "title": "फलकासन विश्लेषण", "tips": ["कंधों से एड़ियों तक सीधी लाइन रखें।", "ऊपरी पीठ चौड़ी रखने के लिए ज़मीन को धक्का दें।", "कोर को कसें ताकि कूल्हे नीचे न झुकें।"]},
    "virabhadrasana_iii": {"label": "वीरभद्रासन III", "title": "वीरभद्रासन III विश्लेषण", "tips": ["आगे और पीछे समान ऊर्जा से फैलें।", "जहाँ तक संभव हो कूल्हों को बराबर रखें।", "संतुलन डगमगाए तो खड़े पैर में हल्का मोड़ रखें।"]},
    "ardha_chandrasana": {"label": "अर्ध चंद्रासन", "title": "अर्ध चंद्रासन विश्लेषण", "tips": ["ऊपरी कूल्हे को खड़े पैर के ऊपर रखें।", "खड़े पैर से ज़मीन पर मजबूती बनाए रखें।", "कमर गिराए बिना छाती खोलें।"]},
    "paschimottanasana": {"label": "पश्चिमोत्तानासन", "title": "पश्चिमोत्तानासन विश्लेषण", "tips": ["आगे झुकने से पहले रीढ़ लंबी करें।", "ऊपरी पीठ के बजाय कूल्हों से अधिक झुकें।", "साँस को नरम और स्थिर रखें।"]},
    "ustrasana": {"label": "उष्ट्रासन", "title": "उष्ट्रासन विश्लेषण", "tips": ["एड़ियों तक पहुँचने से पहले छाती उठाएँ।", "कूल्हों को घुटनों के ऊपर आगे रखें।", "गर्दन को गिराने के बजाय लंबा रखें।"]},
    "malasana": {"label": "मालासन", "title": "मालासन विश्लेषण", "tips": ["पैरों को जड़ से टिकाएँ और छाती ऊपर रखें।", "कोहनियों से घुटनों को हल्का बाहर खोलें।", "जितनी गतिशीलता हो उतना नीचे बैठें, लेकिन ढहें नहीं।"]},
    "dhanurasana": {"label": "धनुरासन", "title": "धनुरासन विश्लेषण", "tips": ["छाती उठाने के लिए पैरों को हाथों की ओर पीछे किक करें।", "आगे शरीर खुलने पर भी साँस स्थिर रखें।", "लोअर बैक पर अधिक दबाव न डालें।"]},
    "salabhasana": {"label": "शलभासन", "title": "शलभासन विश्लेषण", "tips": ["पैरों और छाती को साथ में उठाएँ।", "गर्दन के पीछे हिस्से को लंबा रखें।", "उंगलियों और पैर की उंगलियों तक सक्रिय पहुँच बनाए रखें।"]},
    "supta_baddha_konasana": {"label": "सुप्त बद्ध कोणासन", "title": "सुप्त बद्ध कोणासन विश्लेषण", "tips": ["घुटनों को बिना ज़ोर दिए स्वाभाविक रूप से खुलने दें।", "रिब्स और कंधों को ज़मीन पर नरम रखें।", "शांत साँस के साथ धीरे-धीरे विश्राम बढ़ाएँ।"]},
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
                "निचले शरीर की गतिशीलता और संतुलन बेहतर करता है।",
                "कूल्हों, घुटनों और टखनों की बेहतर जागरूकता देता है।",
                "तनाव में भी साँस को स्थिर रखना सिखाता है।",
            ] if hi else [
                "Improves lower-body mobility and balance.",
                "Builds awareness of hip, knee, and ankle alignment.",
                "Encourages steady breathing under tension.",
            ]
        if profile == "pushup":
            return [
                "कंधों की स्थिरता और रीढ़ नियंत्रण बढ़ाता है।",
                "वेट-बेयरिंग पोज़ में ऊपरी शरीर की सहनशक्ति बढ़ाता है।",
                "पोश्चर और साँस के तालमेल को बेहतर बनाता है।",
            ] if hi else [
                "Builds shoulder stability and spinal control.",
                "Improves upper-body endurance in weight-bearing positions.",
                "Helps connect posture with breathing rhythm.",
            ]
        return [
            "संतुलन, फोकस और बॉडी अवेयरनेस बढ़ाता है।",
            "गहरे कोर स्टेबिलिटी और पोश्चर कंट्रोल को मजबूत करता है।",
            "अलाइनमेंट बनाए रखते हुए शांत साँस सिखाता है।",
        ] if hi else [
            "Improves balance, focus, and body awareness.",
            "Builds deep core stability and postural control.",
            "Encourages calm, steady breathing while holding alignment.",
        ]

    if profile == "squat":
        return [
            "निचले शरीर की ताकत और सिंगल-लेग कंट्रोल बढ़ाता है।",
            "कूल्हों की गतिशीलता और घुटनों की स्थिर दिशा में मदद करता है।",
            "लोडेड मूवमेंट में ब्रेसेड धड़ बनाना सिखाता है।",
        ] if hi else [
            "Builds lower-body strength and single-leg control.",
            "Improves hip mobility and stable knee tracking.",
            "Reinforces a strong braced torso during loaded movement.",
        ]
    if profile == "pushup":
        return [
            "ऊपरी शरीर की पुशिंग ताकत और कंधों की स्थिरता बढ़ाता है।",
            "डायनामिक प्रयास के दौरान धड़ नियंत्रण बेहतर करता है।",
            "हाथ, धड़ और कूल्हों के समन्वय को सिखाता है।",
        ] if hi else [
            "Builds upper-body pushing strength and shoulder stability.",
            "Improves trunk control during dynamic effort.",
            "Teaches better coordination between arms, torso, and hips.",
        ]
    return [
        "कोर एंड्योरेंस और पूरे शरीर के समन्वय को बेहतर करता है।",
        "पोश्चर, मूवमेंट कंट्रोल और अलाइनमेंट अवेयरनेस बढ़ाता है।",
        "बेहतर लय और स्थिरता के साथ मूव करने में मदद करता है।",
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
            f"{label} के लिए स्थिर और संतुलित शुरुआती स्थिति बनाएं।",
            "पहले रीढ़ लंबी करें, फिर बिना ज़बरदस्ती किए धीरे-धीरे पोज़ में जाएँ।",
            movement_tips[0],
            "20 से 45 सेकंड तक शांत नाक की साँस के साथ रुकें, फिर धीरे बाहर आएँ।",
        ] if hi else [
            f"Start in a steady setup for {label}, with your feet, hands, or seat placed evenly.",
            "Lengthen the spine first, then move into the pose gradually without forcing range.",
            movement_tips[0],
            "Hold for 20 to 45 seconds with calm nasal breathing, then exit slowly and reset.",
        ]
        return base_steps

    if profile == "squat":
        return [
            f"{label} के लिए स्थिर स्टांस और ब्रेसेड कोर के साथ सेट हों।",
            "रेप की शुरुआत कूल्हों को पीछे भेजकर या नियंत्रण से नीचे जाकर करें।",
            movement_tips[0],
            "स्मूद तरीके से शुरुआती स्थिति में लौटें और फॉर्म बिगड़ने पर सेट रोकें।",
        ] if hi else [
            f"Set up for {label} with a stable stance and a braced core.",
            "Begin the rep by sitting the hips back or lowering with control.",
            movement_tips[0],
            "Return to the start position smoothly and stop the set when form fades.",
        ]
    if profile == "pushup":
        return [
            f"{label} के लिए कंधों को व्यवस्थित और कोर को सक्रिय करके सेट हों।",
            "पूरे रेप में शरीर को एक नियंत्रित लाइन में रखें।",
            movement_tips[0],
            "हर रेप के बाद साफ़ रीसेट करें, फिर अगला रेप शुरू करें।",
        ] if hi else [
            f"Set up for {label} with shoulders organized and the core switched on.",
            "Move through the rep in one controlled line without rushing the tempo.",
            movement_tips[0],
            "Finish each rep with a clean reset before starting the next one.",
        ]
    return [
        f"{label} के लिए ऐसी स्थिति लें जिसे आप बिना डगमगाए नियंत्रित कर सकें।",
        "मूव करने से पहले धड़ को हल्का ब्रेसेड रखें और पोश्चर सेट करें।",
        movement_tips[0],
        "स्मूद साँस के साथ होल्ड या रेप्स करें और तकनीक टूटने से पहले रुकें।",
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
        return "सर्वश्रेष्ठ कैमरा एंगल: साइड व्यू ताकि कंधे, कूल्हे और टखने की लाइन साफ़ दिखे।" if hi else "Best camera angle: side view so shoulder, hip, and ankle alignment stay visible."
    if profile == "squat":
        return "सर्वश्रेष्ठ कैमरा एंगल: 45-डिग्री फ्रंट या साइड व्यू ताकि कूल्हे, घुटने और टखने की गहराई दिखे।" if hi else "Best camera angle: 45-degree front or side view so hip, knee, and ankle depth can be seen."
    return "सर्वश्रेष्ठ कैमरा एंगल: पूरे शरीर को फ्रेम के बीच रखें और पूरा मूवमेंट दिखने जितनी दूरी रखें।" if hi else "Best camera angle: keep your full body centered in frame with enough distance to see the whole movement."


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
                "squat": "एक निचले शरीर का मूवमेंट जो कूल्हे, घुटने, पोश्चर और संतुलन पर काम करता है।",
                "pushup": "एक ऊपरी या पूरे शरीर का पैटर्न जो पुशिंग स्ट्रेंथ और ट्रंक स्टेबिलिटी बनाता है।",
                "plank": "एक स्थिरता-आधारित मूवमेंट या पोज़ जो अलाइनमेंट, संतुलन और कोर कंट्रोल को चुनौती देता है।",
                "generic": "एक नियंत्रित मूवमेंट पैटर्न जहाँ पोश्चर, जॉइंट अलाइनमेंट और साँस की गुणवत्ता महत्वपूर्ण होती है।",
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
    brand = "फिटनेस कोच" if hi else "FITNESS COACH"
    headline = "बेहतर मूव करें। समझदारी से ट्रेन करें।" if hi else "Move better. Train smarter."
    sub1 = "लाइव पोज़ ट्रैकिंग, गाइडेड वर्कआउट और अपलोड रिव्यू," if hi else "Live pose tracking, guided workouts, upload review,"
    sub2 = "और रोज़ाना ट्रेनिंग के लिए उपयोगी संकेत।" if hi else "and practical cues for stronger daily training."
    support = "पर्सनल सपोर्ट" if hi else "PERSONALIZED SUPPORT"
    support_title = "मूवमेंट फीडबैक" if hi else "Movement feedback"
    support_1 = "एक्सरसाइज संकेत, अपलोड रिव्यू, और" if hi else "Exercise cues, upload review, and"
    support_2 = "एक ही जगह स्पष्ट पोश्चर गाइडेंस।" if hi else "clear posture guidance in one place."
    visual_title = "बेहतर फॉर्म, साफ़ ट्रेनिंग" if hi else "Better form, clearer training"
    visual_sub = "फोकस्ड मूवमेंट प्रैक्टिस के लिए एक आसान विज़ुअल गाइड" if hi else "A simple visual guide for focused movement practice"
    focus = "ट्रेनिंग फोकस" if hi else "TRAINING FOCUS"
    focus_title = "फॉर्म संकेत और गाइडेड रिव्यू" if hi else "Form cues and guided review"
    focus_sub = "एक्सरसाइज सेटअप, लाइव अलाइनमेंट फीडबैक, और अपलोड विश्लेषण।" if hi else "Exercise setup help, live alignment feedback, and upload analysis."
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
    def __init__(self, exercise: str):
        self.exercise = exercise
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

    def analyze(self, landmarks) -> Dict[str, object]:
        points = side_points(landmarks)
        profile = get_exercise_profile(self.exercise)
        if profile == "squat":
            result = self._analyze_squat(points)
        elif profile == "pushup":
            result = self._analyze_pushup(points)
        elif profile == "plank":
            result = self._analyze_plank(points)
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
            "बॉडी लैंडमार्क्स का इंतज़ार है..." if st.session_state.get("language", "en") == "hi" else "Waiting for body landmarks...",
        )
        return {
            "exercise": ex_label(self.exercise),
            "reps": self.rep_count,
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
            return default_result("squat", "थोड़ा पीछे जाएँ ताकि कूल्हे, घुटने और टखने साफ़ दिखें।" if st.session_state.get("language", "en") == "hi" else "Step back so your hips, knees, and ankles are all visible.")

        if self.stage == "up" and knee_angle < 105 and hip_depth > -0.04:
            self.stage = "down"
        elif self.stage == "down" and knee_angle > 155:
            self.stage = "up"
            self.rep_count += 1

        score = 100
        cues: List[str] = []

        if knee_angle > 115:
            score -= 25
            cues.append("बेहतर स्क्वाट गहराई के लिए थोड़ा और नीचे बैठें।" if st.session_state.get("language", "en") == "hi" else "Try sitting deeper to reach better squat depth.")

        if torso_angle < 45:
            score -= 20
            cues.append("आगे झुकने से बचने के लिए छाती थोड़ा ऊपर रखें।" if st.session_state.get("language", "en") == "hi" else "Lift your chest a little to avoid folding forward.")

        if points["knee"] and points["ankle"] and abs(points["knee"].x - points["ankle"].x) > 0.09:
            score -= 15
            cues.append("घुटनों को पैरों के ऊपर अधिक सीधा रखें।" if st.session_state.get("language", "en") == "hi" else "Keep your knees stacked more directly over your feet.")

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("अच्छा स्क्वाट फॉर्म। रिद्म स्थिर रखें।" if st.session_state.get("language", "en") == "hi" else "Nice squat mechanics. Keep the rhythm steady."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }

    def _analyze_pushup(self, points: Dict[str, object]) -> Dict[str, object]:
        elbow_angle = angle_between(points["shoulder"], points["elbow"], points["wrist"])
        body_angle = angle_between(points["shoulder"], points["hip"], points["ankle"])

        if elbow_angle is None or body_angle is None:
            return default_result(
                "pushup",
                "कैमरे की ओर साइड में रहें ताकि कंधा, कोहनी और कलाई दिखे।" if st.session_state.get("language", "en") == "hi" else "Turn sideways to the camera so your shoulder, elbow, and wrist are visible.",
            )

        if self.stage == "up" and elbow_angle < 95:
            self.stage = "down"
        elif self.stage == "down" and elbow_angle > 155:
            self.stage = "up"
            self.rep_count += 1

        score = 100
        cues: List[str] = []

        if body_angle < 150:
            score -= 30
            cues.append("कूल्हों को कंधों और टखनों की लाइन में रखें।" if st.session_state.get("language", "en") == "hi" else "Keep your hips in line with your shoulders and ankles.")

        if elbow_angle > 110 and self.stage == "down":
            score -= 20
            cues.append("बेहतर रेप के लिए थोड़ा और नीचे जाएँ।" if st.session_state.get("language", "en") == "hi" else "Lower a little deeper for a stronger rep.")

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("अच्छा पुश-अप अलाइनमेंट। कोर को कसकर रखें।" if st.session_state.get("language", "en") == "hi" else "Solid push-up alignment. Keep your core braced."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }

    def _analyze_plank(self, points: Dict[str, object]) -> Dict[str, object]:
        body_angle = angle_between(points["shoulder"], points["hip"], points["ankle"])
        hip_height = None
        if points["hip"] and points["shoulder"]:
            hip_height = points["hip"].y - points["shoulder"].y

        if body_angle is None or hip_height is None:
            self.plank_start_time = None
            self.hold_time = 0.0
            return default_result("plank", "साइड में रहें और पूरे शरीर को फ्रेम में रखें।" if st.session_state.get("language", "en") == "hi" else "Turn sideways and keep your whole body in frame.")

        score = 100
        cues: List[str] = []

        if body_angle < 158:
            score -= 30
            cues.append("कूल्हों और कोर को सक्रिय करके शरीर की लाइन सीधी करें।" if st.session_state.get("language", "en") == "hi" else "Straighten your body line by lifting through the hips and core.")

        if hip_height > 0.02:
            score -= 18
            cues.append("कूल्हों को कंधों के स्तर से नीचे न गिरने दें।" if st.session_state.get("language", "en") == "hi" else "Avoid letting your hips sag below shoulder level.")

        if hip_height < -0.10:
            score -= 18
            cues.append("फ्लैट प्लैंक बनाए रखने के लिए कूल्हों को थोड़ा नीचे लाएँ।" if st.session_state.get("language", "en") == "hi" else "Lower your hips slightly to maintain a flat plank.")

        if score >= 80:
            if self.plank_start_time is None:
                self.plank_start_time = time.time()
            self.hold_time = time.time() - self.plank_start_time
        else:
            self.plank_start_time = None
            self.hold_time = 0.0

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("मजबूत प्लैंक लाइन। साँस लेते रहें और कोर को कसें।" if st.session_state.get("language", "en") == "hi" else "Strong plank line. Keep breathing and stay braced."),
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
                "पूरा शरीर दिखाएँ ताकि पोश्चर का बेहतर आकलन हो सके।" if st.session_state.get("language", "en") == "hi" else "Keep your full body visible so the trainer can assess your posture more clearly.",
            )

        score = 100
        cues: List[str] = []

        if body_angle is not None and body_angle < 145:
            score -= 25
            cues.append("धड़ को लंबा रखें और कूल्हों पर ढहने से बचें।" if st.session_state.get("language", "en") == "hi" else "Lengthen through the torso and avoid collapsing through the hips.")

        if knee_angle is not None and knee_angle < 120:
            score -= 15
            cues.append("पैरों को स्थिर रखें और घुटनों को अनावश्यक रूप से अंदर गिरने न दें।" if st.session_state.get("language", "en") == "hi" else "Stabilize through the legs and avoid unnecessary knee collapse.")

        return {
            "title": ex_title(self.exercise),
            "message": cues[0] if cues else ("पोश्चर स्थिर दिख रहा है। नियंत्रित साँस और अलाइनमेंट के साथ जारी रखें।" if st.session_state.get("language", "en") == "hi" else "Posture looks stable. Keep moving with controlled breathing and alignment."),
            "tips": cues if cues else ex_tips(self.exercise),
            "score": score,
        }


class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.pose = create_pose_tracker(static_image_mode=False)
        self.exercise = "squat"
        self.coach = PoseCoach(self.exercise)
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
        st.link_button("ट्यूटोरियल वीडियो खोलें" if st.session_state.get("language", "en") == "hi" else "Open tutorial video", details["video_url"], use_container_width=True)
        st.markdown(
            '<p class="video-link-caption">यह एक ट्यूटोरियल सर्च खोलता है ताकि यूज़र जल्दी से मूवमेंट डेमो देख सके।</p>' if st.session_state.get("language", "en") == "hi" else '<p class="video-link-caption">Opens a tutorial search so the user can quickly review movement demos.</p>',
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
        intensity_tip = "आपका फॉर्म अच्छा दिख रहा है। धीमे, नियंत्रित रेप्स करें और धीरे-धीरे वॉल्यूम बढ़ाएँ।" if st.session_state.get("language", "en") == "hi" else "Your form looks strong here. Focus on slow, controlled reps and add volume gradually."
    elif score >= 60:
        intensity_tip = "आप सही दिशा में हैं। पहले स्पीड कम करें और अलाइनमेंट साफ़ करें, फिर रेप्स बढ़ाएँ।" if st.session_state.get("language", "en") == "hi" else "You are close. Reduce speed, clean up alignment first, and then build reps."
    else:
        intensity_tip = "इसे तकनीक अभ्यास सेट की तरह लें। छोटे सेट, धीमा टेम्पो और बार-बार रीसेट सबसे अधिक मदद करेंगे।" if st.session_state.get("language", "en") == "hi" else "Use this as a technique practice set. Short sets, slow tempo, and frequent resets will help most."

    is_asana = exercise in ASANA_EXERCISES
    if is_asana:
        workout_guides = [
            "20 से 45 सेकंड तक मुद्रा पकड़ें और नाक से धीरे साँस लें।",
            "मुद्रा में प्रवेश और बाहर निकलना नियंत्रण से करें, ज़बरदस्ती नहीं।",
            "यदि अलाइनमेंट टूटे तो प्रॉप्स या कम रेंज का उपयोग करें।",
        ] if st.session_state.get("language", "en") == "hi" else [
            "Hold the posture for 20 to 45 seconds while breathing slowly through the nose.",
            "Enter and exit the pose with control rather than forcing range of motion.",
            "Use props or a shorter range if alignment breaks before the breath stays steady.",
        ]
    else:
        workout_guides = [
            "वर्किंग सेट्स से पहले 1 से 2 आसान तकनीकी सेट करें।",
            "3 सेट, 6 से 12 नियंत्रित रेप्स करें, या स्टैटिक एक्सरसाइज के लिए छोटे होल्ड रखें।",
            "पोश्चर बिगड़ना शुरू होते ही सेट रोक दें, अतिरिक्त रेप्स के पीछे न भागें।",
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
            <div class="panel-kicker">{'ट्रेनिंग टिप्स' if st.session_state.get('language','en') == 'hi' else 'Training Tips'}</div>
            <h3 class="panel-title">{f'इस {source_label.lower()} से सुधार कैसे करें' if st.session_state.get('language','en') == 'hi' else f'How to improve from this {source_label.lower()}'}</h3>
            <p class="panel-copy">{intensity_tip}</p>
            <ul>{tips_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trainer_care() -> None:
    care_items = [
        "रेप्स या लंबे होल्ड शुरू करने से पहले 5 से 10 मिनट वार्म-अप करें。",
        "यदि तेज दर्द, चक्कर या जॉइंट अस्थिरता हो तो तुरंत रुकें।",
        "पास में पानी रखें और सेट्स के बीच छोटे ब्रेक लें।",
        "बेहतर फीडबैक के लिए ऐसा कैमरा एंगल रखें जिसमें पूरा शरीर दिखे।",
        "स्पीड या रेप्स से पहले नियंत्रित फॉर्म को प्राथमिकता दें।",
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


def analyze_pose_frame(
    image_bgr,
    exercise: str,
    pose_tracker,
    coach: PoseCoach,
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
        result = coach.analyze(results.pose_landmarks.landmark)
    else:
        coach.reset_tracking()
        result = default_result(exercise, "कोई स्पष्ट पोज़ नहीं मिला। पूरे शरीर को दिखाएँ और कैमरा एंगल साफ़ रखें।" if st.session_state.get("language", "en") == "hi" else "No clear pose found. Keep the whole body visible and try a cleaner angle.")

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
        st.error("यह इमेज पढ़ी नहीं जा सकी। कृपया JPG, JPEG, या PNG फ़ाइल अपलोड करें।" if st.session_state.get("language", "en") == "hi" else "We could not read that image. Please upload a JPG, JPEG, or PNG file.")
        return

    pose_tracker = create_pose_tracker(static_image_mode=True)
    coach = PoseCoach(exercise)
    annotated, stats = analyze_pose_frame(image_bgr, exercise, pose_tracker, coach)
    pose_tracker.close()

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="मूल फोटो" if st.session_state.get("language", "en") == "hi" else "Original photo", use_container_width=True)
    with right_col:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="पोज़ डिटेक्शन परिणाम" if st.session_state.get("language", "en") == "hi" else "Pose detection result", use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        render_metric(t("exercise"), stats["exercise"])
    with metric_col2:
        render_metric(t("form_score"), f"{stats['score']}%")
    with metric_col3:
        render_metric(t("hold_time"), f"{stats['hold_time']:.1f}s")

    render_feedback(stats)
    render_training_plan(exercise, stats, "फोटो" if st.session_state.get("language", "en") == "hi" else "Photo")


def analyze_uploaded_video(uploaded_file, exercise: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(uploaded_file.read())
        input_path = input_tmp.name

    output_path = f"{input_path}.processed.mp4"
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        st.error("यह वीडियो खुल नहीं सका। कृपया MP4, MOV, AVI, या MPEG फ़ाइल अपलोड करें।" if st.session_state.get("language", "en") == "hi" else "We could not open that video. Please upload an MP4, MOV, AVI, or MPEG file.")
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
    coach = PoseCoach(exercise)
    progress = st.progress(0, text="वीडियो पोश्चर विश्लेषण हो रहा है..." if st.session_state.get("language", "en") == "hi" else "Analyzing video posture...")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    processed_frames = 0
    final_stats = coach.stats(default_result(exercise, "अपलोड किए गए वर्कआउट वीडियो का विश्लेषण हो रहा है..." if st.session_state.get("language", "en") == "hi" else "Analyzing uploaded workout video..."))

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated, final_stats = analyze_pose_frame(frame, exercise, pose_tracker, coach)
        writer.write(annotated)
        processed_frames += 1

        if frame_count > 0:
            progress.progress(min(processed_frames / frame_count, 1.0), text="वीडियो पोश्चर विश्लेषण हो रहा है..." if st.session_state.get("language", "en") == "hi" else "Analyzing video posture...")

    capture.release()
    writer.release()
    pose_tracker.close()
    progress.empty()

    if processed_frames == 0:
        st.error("अपलोड किए गए वीडियो में पढ़े जा सकने वाले फ्रेम नहीं मिले।" if st.session_state.get("language", "en") == "hi" else "The uploaded video did not contain readable frames.")
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return

    with open(output_path, "rb") as processed_file:
        video_bytes = processed_file.read()

    st.video(video_bytes)
    st.download_button(
        "विश्लेषित वीडियो डाउनलोड करें" if st.session_state.get("language", "en") == "hi" else "Download analyzed video",
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
    render_training_plan(exercise, final_stats, "वीडियो" if st.session_state.get("language", "en") == "hi" else "Video")


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
        st.caption("कोई मूवमेंट चुनें और पूरे शरीर को फ्रेम में रखें ताकि ट्रैकिंग अधिक स्थिर रहे।" if st.session_state["language"] == "hi" else "Pick a movement and keep your body fully visible for more stable landmarks.")
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
        st.caption("पुश-अप और प्लैंक के लिए साइड कैमरा एंगल सबसे बेहतर रहता है।" if st.session_state["language"] == "hi" else "Push-ups and planks work best with a side camera angle.")
        st.caption("यह एक कोचिंग प्रोटोटाइप है, मेडिकल या बायोमैकेनिकल डायग्नोस्टिक टूल नहीं है।" if st.session_state["language"] == "hi" else "This is a coaching prototype, not a medical assessment tool.")


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
        "message": "पोश्चर ट्रैकिंग शुरू करने के लिए वेबकैम चालू करें।" if st.session_state.get("language", "en") == "hi" else "Start the webcam to begin posture tracking.",
        "tips": ex_tips(st.session_state["exercise"]),
    }

    col1, col2 = st.columns([1.5, 1.0], gap="large")

    with col1:
        render_section_intro(
            t("live_coach"),
            t("webcam_posture"),
            t("webcam_posture_copy"),
        )

        ctx = webrtc_streamer(
            key="ai-fitness-trainer",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PoseVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.set_exercise(st.session_state["exercise"])
            stats_placeholder = ctx.video_processor.get_stats()

        st.session_state["session_state"] = "Live" if ctx.state.playing else "Idle"

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
