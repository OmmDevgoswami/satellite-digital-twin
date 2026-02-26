"""
translations.py
---------------
Multilingual string dictionary for the Streamlit web app.

Languages supported:
  - English (en)
  - Hindi (hi)
  - Telugu (te)

WHY hardcoded strings over dynamic translation API?
  - No internet required at runtime (works offline/on-campus servers)
  - Exact control over terminology (e.g., correct environmental/technical terms)
  - Faster rendering — no API call delay per page load
  - More reliable for a demo/submission environment

To add a new language:
  1. Add its ISO code to LANGUAGES dict
  2. Add translations for every key in STRINGS
"""

LANGUAGES = {
    "English":  "en",
    "हिन्दी (Hindi)":  "hi",
    "తెలుగు (Telugu)": "te",
}

STRINGS = {
    # ── App titles & nav ────────────────────────────────────────────────
    "app_title": {
        "en": "Satellite Dump Detection System",
        "hi": "उपग्रह कचरा डंप पहचान प्रणाली",
        "te": "ఉపగ్రహ వ్యర్థ డంప్ గుర్తింపు వ్యవస్థ",
    },
    "app_subtitle": {
        "en": "AI-Powered Illegal Garbage Dump Detection using Satellite Imagery",
        "hi": "उपग्रह चित्रों का उपयोग करके अवैध कचरा डंप की AI पहचान",
        "te": "ఉపగ్రహ చిత్రాలను ఉపయోగించి చట్టవిరుద్ధ వ్యర్థ డంప్‌ల AI గుర్తింపు",
    },
    "upload_label": {
        "en": "Upload a Satellite / Aerial Image",
        "hi": "एक उपग्रह / हवाई छवि अपलोड करें",
        "te": "ఒక ఉపగ్రహ / వైమానిక చిత్రాన్ని అప్‌లోడ్ చేయండి",
    },
    "analyze_btn": {
        "en": "Analyze Image",
        "hi": "छवि का विश्लेषण करें",
        "te": "చిత్రాన్ని విశ్లేషించండి",
    },
    # ── Results ─────────────────────────────────────────────────────────
    "result_dump": {
        "en": "ILLEGAL DUMP DETECTED",
        "hi": "अवैध डंप पहचाना गया",
        "te": "చట్టవిరుద్ధ డంప్ గుర్తించబడింది",
    },
    "result_no_dump": {
        "en": "No Illegal Dump Detected",
        "hi": "कोई अवैध डंप नहीं पाया गया",
        "te": "చట్టవిరుద్ధ డంప్ గుర్తించబడలేదు",
    },
    "confidence": {
        "en": "Confidence Score",
        "hi": "विश्वास स्कोर",
        "te": "విశ్వాస స్కోరు",
    },
    "seg_map": {
        "en": "Dump Region Map",
        "hi": "डंप क्षेत्र मानचित्र",
        "te": "డంప్ ప్రాంత మ్యాప్",
    },
    "coverage": {
        "en": "Estimated Dump Coverage",
        "hi": "अनुमानित डंप क्षेत्र",
        "te": "అంచనా డంప్ కవరేజ్",
    },
    # ── Sidebar ──────────────────────────────────────────────────────────
    "about_title": {
        "en": "About This System",
        "hi": "इस प्रणाली के बारे में",
        "te": "ఈ వ్యవస్థ గురించి",
    },
    "about_text": {
        "en": ("This system uses deep learning to detect illegal garbage "
               "dumps from satellite imagery. It combines a ResNet34 "
               "classifier (94.7% recall) with a U-Net segmentation model "
               "to locate dump regions at pixel level."),
        "hi": ("यह प्रणाली उपग्रह चित्रों से अवैध कचरा डंप का पता लगाने के "
               "लिए डीप लर्निंग का उपयोग करती है।"),
        "te": ("ఈ వ్యవస్థ ఉపగ్రహ చిత్రాల నుండి చట్టవిరుద్ధ వ్యర్థ డంప్‌లను "
               "గుర్తించడానికి డీప్ లెర్నింగ్‌ని ఉపయోగిస్తుంది."),
    },
    "model_info": {
        "en": "Model Information",
        "hi": "मॉडल जानकारी",
        "te": "మోడల్ సమాచారం",
    },
    "dataset": {
        "en": "Dataset",
        "hi": "डेटासेट",
        "te": "డేటాసెట్",
    },
    "threshold_label": {
        "en": "Detection Threshold",
        "hi": "पहचान सीमा",
        "te": "గుర్తింపు థ్రెషోల్డ్",
    },
    "tab_classify": {
        "en": "Classify",
        "hi": "वर्गीकृत करें",
        "te": "వర్గీకరించు",
    },
    "tab_segment": {
        "en": "Segment",
        "hi": "खंड",
        "te": "విభజించు",
    },
    "tab_metrics": {
        "en": "Model Metrics",
        "hi": "मॉडल मेट्रिक्स",
        "te": "మోడల్ మెట్రిక్స్",
    },
    "tab_history": {
        "en": "History",
        "hi": "इतिहास",
        "te": "చరిత్ర",
    },
    "original_img": {
        "en": "Original Image",
        "hi": "मूल छवि",
        "te": "అసలు చిత్రం",
    },
    "overlay_img": {
        "en": "Detection Overlay",
        "hi": "पहचान ओवरले",
        "te": "గుర్తింపు ఓవర్‌లే",
    },
}


def t(key: str, lang: str) -> str:
    """
    Translate a key to the given language code.
    Falls back to English if key or lang not found.
    """
    return STRINGS.get(key, {}).get(lang, STRINGS.get(key, {}).get("en", key))
