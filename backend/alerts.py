"""
Alert generation + multilingual delivery.

ACCESSIBILITY IS THE POINT. An alert a driver cannot read is not an alert. NER
has eight states and no single shared language, so the alert layer is
data-driven: templates live in LOCALES, each state maps to a default language,
and every alert is emitted in English plus the local language.

HONESTY NOTE (please keep this in the repo and say it if asked): the English,
Hindi, Assamese, Bengali and Nepali strings are reliable. The Mizo, Khasi and
Meitei strings are marked `review_pending` — they are structurally correct
placeholders written without a native speaker, and they must be reviewed before
any real deployment. Shipping unreviewed text in a safety-critical alert is a
real harm, so the system flags it rather than pretending otherwise. The
architecture (swap one JSON block) is the deliverable; verified translation is
a field-partner task, not something to fake in a hackathon.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from . import db

# --------------------------------------------------------------------------
LANGUAGES = {
    "en": {"name": "English", "script": "Latin", "review_pending": False},
    "hi": {"name": "हिन्दी (Hindi)", "script": "Devanagari", "review_pending": False},
    "as": {"name": "অসমীয়া (Assamese)", "script": "Bengali-Assamese",
           "review_pending": False},
    "bn": {"name": "বাংলা (Bengali)", "script": "Bengali", "review_pending": False},
    "ne": {"name": "नेपाली (Nepali)", "script": "Devanagari", "review_pending": False},
    "lus": {"name": "Mizo ṭawng", "script": "Latin", "review_pending": True},
    "kha": {"name": "Ka Ktien Khasi", "script": "Latin", "review_pending": True},
    "mni": {"name": "মৈতৈলোন্ (Meitei)", "script": "Bengali", "review_pending": True},
}

STATE_LANGUAGE = {
    "AS": "as", "ML": "kha", "AR": "hi", "NL": "en", "MN": "mni",
    "MZ": "lus", "TR": "bn", "SK": "ne", "WB": "bn",
}

# --------------------------------------------------------------------------
LOCALES: Dict[str, Dict[str, str]] = {
    "en": {
        "blocked_title": "ROAD BLOCKED — {corridor}",
        "blocked_body": ("{road} is impassable. Cause: {cause}. Estimated "
                         "clearance {hours} h. {advice}"),
        "risky_title": "CAUTION — {corridor}",
        "risky_body": ("{road} is passable but high-risk. Cause: {cause}. Expect "
                       "{hours} h delay. Drive with care and report changes."),
        "cutoff_title": "ISOLATION WARNING — {place}",
        "cutoff_body": ("{place} is at risk of losing its only road link via "
                        "{corridor}. Pre-position essential supplies now."),
        "cargo_title": "CRITICAL CARGO AT RISK",
        "cargo_body": ("{cargo} consignment to {dest}: predicted transit {eta} "
                       "exceeds the {limit} h viability window. {advice}"),
        "advice_alt": "Use alternate route via {alt}.",
        "advice_noalt": "No road alternate exists. Escalate to air or rail.",
    },
    "hi": {
        "blocked_title": "मार्ग अवरुद्ध — {corridor}",
        "blocked_body": ("{road} पर आवागमन बंद है। कारण: {cause}। अनुमानित "
                         "निकासी {hours} घंटे। {advice}"),
        "risky_title": "सावधान — {corridor}",
        "risky_body": ("{road} खुला है परन्तु अत्यधिक जोखिमपूर्ण। कारण: {cause}। "
                       "{hours} घंटे की देरी संभव। सावधानी से चलें।"),
        "cutoff_title": "संपर्क-भंग चेतावनी — {place}",
        "cutoff_body": ("{place} का {corridor} से एकमात्र सड़क संपर्क टूटने का "
                        "खतरा है। आवश्यक सामग्री अभी पहुँचाएँ।"),
        "cargo_title": "अति-आवश्यक माल जोखिम में",
        "cargo_body": ("{dest} हेतु {cargo} खेप: अनुमानित यात्रा {eta}, जो "
                       "{limit} घंटे की सीमा से अधिक है। {advice}"),
        "advice_alt": "वैकल्पिक मार्ग {alt} से जाएँ।",
        "advice_noalt": "कोई सड़क विकल्प नहीं। वायु अथवा रेल मार्ग अपनाएँ।",
    },
    "as": {
        "blocked_title": "পথ বন্ধ — {corridor}",
        "blocked_body": ("{road}ত যাত্ৰা সম্ভৱ নহয়। কাৰণ: {cause}। আনুমানিক "
                         "পৰিষ্কাৰৰ সময় {hours} ঘণ্টা। {advice}"),
        "risky_title": "সাৱধান — {corridor}",
        "risky_body": ("{road} মুকলি যদিও অতি বিপদজনক। কাৰণ: {cause}। {hours} "
                       "ঘণ্টা পলম হ'ব পাৰে। সাৱধানে যাত্ৰা কৰক।"),
        "cutoff_title": "বিচ্ছিন্নতাৰ সকীয়নি — {place}",
        "cutoff_body": ("{place}ৰ {corridor}ৰ একমাত্ৰ পথ সংযোগ বিচ্ছিন্ন হোৱাৰ "
                        "আশঙ্কা আছে। প্ৰয়োজনীয় সামগ্ৰী এতিয়াই পঠিয়াওক।"),
        "cargo_title": "অতি প্ৰয়োজনীয় সামগ্ৰী বিপদত",
        "cargo_body": ("{dest}লৈ {cargo} সামগ্ৰী: আনুমানিক যাত্ৰা {eta}, যি "
                       "{limit} ঘণ্টাৰ সীমা অতিক্ৰম কৰে। {advice}"),
        "advice_alt": "বিকল্প পথ {alt} ব্যৱহাৰ কৰক।",
        "advice_noalt": "কোনো বিকল্প পথ নাই। বিমান বা ৰেলৰ ব্যৱস্থা কৰক।",
    },
    "bn": {
        "blocked_title": "রাস্তা বন্ধ — {corridor}",
        "blocked_body": ("{road} চলাচলের অযোগ্য। কারণ: {cause}। আনুমানিক "
                         "পরিষ্কারের সময় {hours} ঘণ্টা। {advice}"),
        "risky_title": "সতর্কতা — {corridor}",
        "risky_body": ("{road} খোলা থাকলেও অত্যন্ত ঝুঁকিপূর্ণ। কারণ: {cause}। "
                       "{hours} ঘণ্টা বিলম্ব হতে পারে। সতর্কভাবে চলুন।"),
        "cutoff_title": "বিচ্ছিন্নতার সতর্কতা — {place}",
        "cutoff_body": ("{place}-এর {corridor} দিয়ে একমাত্র সড়ক সংযোগ বিচ্ছিন্ন "
                        "হওয়ার আশঙ্কা। প্রয়োজনীয় সামগ্রী এখনই পাঠান।"),
        "cargo_title": "অতি-প্রয়োজনীয় পণ্য ঝুঁকিতে",
        "cargo_body": ("{dest}-এ {cargo} চালান: আনুমানিক যাত্রা {eta}, যা "
                       "{limit} ঘণ্টার সীমা ছাড়িয়ে যায়। {advice}"),
        "advice_alt": "বিকল্প পথ {alt} ব্যবহার করুন।",
        "advice_noalt": "কোনো সড়ক বিকল্প নেই। বিমান বা রেল ব্যবস্থা করুন।",
    },
    "ne": {
        "blocked_title": "सडक अवरुद्ध — {corridor}",
        "blocked_body": ("{road} मा आवतजावत बन्द छ। कारण: {cause}। अनुमानित "
                         "सफाई {hours} घण्टा। {advice}"),
        "risky_title": "सावधान — {corridor}",
        "risky_body": ("{road} खुला छ तर जोखिमपूर्ण। कारण: {cause}। {hours} "
                       "घण्टा ढिलाइ हुन सक्छ। सावधानीपूर्वक यात्रा गर्नुहोस्।"),
        "cutoff_title": "सम्पर्कविच्छेद चेतावनी — {place}",
        "cutoff_body": ("{place} को {corridor} मार्गको एकमात्र सडक सम्पर्क "
                        "टुट्ने खतरा छ। आवश्यक सामग्री अहिले नै पठाउनुहोस्।"),
        "cargo_title": "अत्यावश्यक सामान जोखिममा",
        "cargo_body": ("{dest} लागि {cargo} खेप: अनुमानित यात्रा {eta}, जो "
                       "{limit} घण्टाको सीमा नाघ्छ। {advice}"),
        "advice_alt": "वैकल्पिक मार्ग {alt} प्रयोग गर्नुहोस्।",
        "advice_noalt": "सडक विकल्प छैन। हवाई वा रेल मार्ग अपनाउनुहोस्।",
    },
    # ---- review_pending locales ------------------------------------------
    "lus": {
        "blocked_title": "KAWNG A KHAR — {corridor}",
        "blocked_body": ("{road} hi kal theih loh a ni. Chhan: {cause}. Tihfelh "
                         "hun ~{hours} dar. {advice}"),
        "risky_title": "FIMKHUR ROH — {corridor}",
        "risky_body": ("{road} hi kal theih a ni nain a hlauhawm hle. Chhan: "
                       "{cause}. Hun ~{hours} dar a tlem zawk ang. Fimkhur takin "
                       "kal rawh."),
        "cutoff_title": "INTHENHRANNA VAUNA — {place}",
        "cutoff_body": ("{place} hi {corridor} kawng hmang chauh a nih avangin "
                        "inthenhran a nih theih. Thil pawimawh tûn hian dah "
                        "ṭhat rawh."),
        "cargo_title": "THIL PAWIMAWH HLAUHAWMNA",
        "cargo_body": ("{dest} atâna {cargo}: hun ~{eta} a hun bituk {limit} dar "
                       "a liam. {advice}"),
        "advice_alt": "Kawng dang {alt} hmang rawh.",
        "advice_noalt": "Kawng dang a awm lo. Thlawhna emaw trên hmang rawh.",
    },
    "kha": {
        "blocked_title": "KA SURÒK LA KHAÑ — {corridor}",
        "blocked_body": ("Ym lah ban leit ha {road}. Daw: {cause}. Ban pyndep "
                         "~{hours} por. {advice}"),
        "risky_title": "PEIT BHA — {corridor}",
        "risky_body": ("Ka {road} la kylluid hynrei bad ka jingeh. Daw: {cause}. "
                       "Ngeit ba yn thait ~{hours} por. Leit bad ka jingpeit."),
        "cutoff_title": "JINGSNGEWBHA BA YN BAKHAÑ — {place}",
        "cutoff_body": ("Ka {place} ka don tang ka lynti {corridor}. Lah ban "
                        "bakhañ. Ai ia ki jingkhlaiñ mynta."),
        "cargo_title": "KI JINGTHIE BA DONKAM SHAPHANG KA JINGEH",
        "cargo_body": ("{cargo} sha {dest}: ka por leit ~{eta}, ba la poi shaduh "
                       "{limit} por. {advice}"),
        "advice_alt": "Bteng ia ka lynti bapar {alt}.",
        "advice_noalt": "Ym don lynti bapar. Bteng ia ka plen ne ka relgari.",
    },
    "mni": {
        "blocked_title": "লম্বী থিংলে — {corridor}",
        "blocked_body": ("{road} দা চৎপা য়ারোই। মরম: {cause}। শেমদোকপা "
                         "{hours} পুং। {advice}"),
        "risky_title": "চেকশিন্নবা — {corridor}",
        "risky_body": ("{road} হাংদোক্লি অদুবু অশোনবা য়াম্না লৈ। মরম: {cause}। "
                       "{hours} পুং তাংবা য়াই। চেকশিন্না চৎলু।"),
        "cutoff_title": "তোখায়বগী চেকশিন্নবা — {place}",
        "cutoff_body": ("{place} কী {corridor} অমত্তা লম্বী অসি তোখায়বা য়াই। "
                        "তঙাইফদবা পোৎলম হৌজিক থারকউ।"),
        "cargo_title": "য়াম্না তঙাইফদবা পোৎলম অশোনবদা",
        "cargo_body": ("{dest} দা {cargo}: চৎকদবা মতম {eta}, মসি {limit} পুংগী "
                       "মতমদগী হেন্না লৈ। {advice}"),
        "advice_alt": "অতোপ্পা লম্বী {alt} শিজিন্নবীয়ু।",
        "advice_noalt": "অতোপ্পা লম্বী লৈতে। নোংমাইজিং নত্রগা রেল শিজিন্নবীয়ু।",
    },
}

CAUSE_LABELS = {
    "en": {"landslide": "landslide / slope failure", "flood": "flooding",
           "snow": "snow and ice", "storm": "storm damage",
           "blockade": "blockade / bandh", "saturation": "saturated slopes",
           "other": "road damage"},
    "hi": {"landslide": "भूस्खलन", "flood": "बाढ़", "snow": "बर्फ़ और पाला",
           "storm": "तूफ़ान से क्षति", "blockade": "बंद / नाकाबंदी",
           "saturation": "जलसंतृप्त ढलान", "other": "सड़क क्षति"},
    "as": {"landslide": "ভূমিস্খলন", "flood": "বন্যা", "snow": "বৰফ আৰু শিল",
           "storm": "ধুমুহাৰ ক্ষতি", "blockade": "বন্ধ / অৱৰোধ",
           "saturation": "পানী সংপৃক্ত ঢাল", "other": "পথৰ ক্ষতি"},
    "bn": {"landslide": "ভূমিধস", "flood": "বন্যা", "snow": "বরফ ও তুষার",
           "storm": "ঝড়ের ক্ষতি", "blockade": "বন্ধ / অবরোধ",
           "saturation": "জলসিক্ত ঢাল", "other": "রাস্তার ক্ষতি"},
    "ne": {"landslide": "पहिरो", "flood": "बाढी", "snow": "हिउँ र बरफ",
           "storm": "आँधीको क्षति", "blockade": "बन्द / नाकाबन्दी",
           "saturation": "जलमग्न भिरालो", "other": "सडक क्षति"},
    "lus": {"landslide": "lei tla", "flood": "tuilêt", "snow": "vûr leh ruahtui khal",
            "storm": "thlipui tihchhiat", "blockade": "kawng kharna",
            "saturation": "lei tui khat", "other": "kawng tihchhiatna"},
    "kha": {"landslide": "ka jingduh jaka", "flood": "ka jingtuid um",
            "snow": "ka snow bad ka eitlang", "storm": "ka jingsniew eriong",
            "blockade": "ka jingkhañ lynti", "saturation": "ki jaka ba la dup um",
            "other": "ka jingduh surok"},
    "mni": {"landslide": "লৈ তাবা", "flood": "ঈশিং ইচাও", "snow": "ঊন অমসুং ঈং",
            "storm": "নোংলৈ অমাংবা", "blockade": "লম্বী থিংবা",
            "saturation": "ঈশিং য়াওবা লৈপাক", "other": "লম্বী অমাংবা"},
}


# --------------------------------------------------------------------------
def default_language(state_code: str) -> str:
    return STATE_LANGUAGE.get(state_code, "en")


def _cause_from_segment(seg: dict) -> str:
    if seg.get("disruption"):
        return "blockade"
    f = seg.get("features", {})
    w = seg.get("weather", {})
    if f.get("snow_exposure", 0) > 0.3 and w.get("temperature_c", 30) < 5:
        return "snow"
    if w.get("condition") == "storm":
        return "storm"
    if f.get("api_7d", 0) > 180 and f.get("landslide_base", 0) > 0.3:
        return "saturation"
    if f.get("landslide_base", 0) >= f.get("flood_exposure", 0) and \
            f.get("landslide_base", 0) > 0.15:
        return "landslide"
    if f.get("flood_exposure", 0) > 0.2:
        return "flood"
    return "other"


def render(kind: str, lang: str, **kw) -> Dict[str, str]:
    loc = LOCALES.get(lang, LOCALES["en"])
    base = LOCALES["en"]
    title = loc.get(f"{kind}_title", base[f"{kind}_title"])
    body = loc.get(f"{kind}_body", base[f"{kind}_body"])
    return {"title": title.format(**kw), "body": body.format(**kw)}


def segment_alert(seg: dict, alt_text: Optional[str] = None,
                  langs: Optional[List[str]] = None) -> Optional[dict]:
    """Build a bilingual (English + local) alert for one segment."""
    if seg["risk_label"] == 0:
        return None
    cause_key = _cause_from_segment(seg)
    local = default_language(seg["state"])
    use = langs or (["en"] if local == "en" else ["en", local])

    kind = "blocked" if seg["risk_label"] == 2 else "risky"
    out: Dict[str, dict] = {}
    for lg in use:
        adv_tpl = LOCALES.get(lg, LOCALES["en"])
        advice = (adv_tpl.get("advice_alt", LOCALES["en"]["advice_alt"]).format(
                  alt=alt_text) if alt_text
                  else adv_tpl.get("advice_noalt", LOCALES["en"]["advice_noalt"]))
        out[lg] = render(
            kind, lg,
            corridor=seg["corridor"], road=seg["name"],
            cause=CAUSE_LABELS.get(lg, CAUSE_LABELS["en"]).get(cause_key, cause_key),
            hours=f"{seg['delay_hours']:.0f}", advice=advice)

    return {
        "alert_id": f"ALT-{uuid.uuid4().hex[:10]}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "severity": "critical" if seg["risk_label"] == 2 else "warning",
        "scope": "segment",
        "target": seg["road_id"],
        "corridor": seg["corridor"],
        "state": seg["state"],
        "cause": cause_key,
        "risk_score": seg["risk_score"],
        "severity_score": seg["severity"],
        "delay_hours": seg["delay_hours"],
        "coords": seg["coords"],
        "languages": list(out.keys()),
        "review_pending_languages": [l for l in out
                                     if LANGUAGES.get(l, {}).get("review_pending")],
        "text": out,
    }


def isolation_alert(place: str, corridor: str, state_code: str) -> dict:
    local = default_language(state_code)
    use = ["en"] if local == "en" else ["en", local]
    out = {lg: render("cutoff", lg, place=place, corridor=corridor) for lg in use}
    return {
        "alert_id": f"ALT-{uuid.uuid4().hex[:10]}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "severity": "critical", "scope": "isolation", "target": place,
        "corridor": corridor, "state": state_code,
        "languages": list(out.keys()),
        "review_pending_languages": [l for l in out
                                     if LANGUAGES.get(l, {}).get("review_pending")],
        "text": out,
    }


def cargo_alert(cargo_label: str, dest: str, eta_text: str, limit: int,
                state_code: str, advice: str) -> dict:
    local = default_language(state_code)
    use = ["en"] if local == "en" else ["en", local]
    out = {}
    for lg in use:
        adv = LOCALES.get(lg, LOCALES["en"]).get("advice_noalt",
                                                 LOCALES["en"]["advice_noalt"])
        out[lg] = render("cargo", lg, cargo=cargo_label, dest=dest,
                         eta=eta_text, limit=limit, advice=adv)
    out["en"]["body"] = LOCALES["en"]["cargo_body"].format(
        cargo=cargo_label, dest=dest, eta=eta_text, limit=limit, advice=advice)
    return {
        "alert_id": f"ALT-{uuid.uuid4().hex[:10]}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "severity": "critical", "scope": "cargo", "target": dest,
        "state": state_code,
        "languages": list(out.keys()),
        "review_pending_languages": [l for l in out
                                     if LANGUAGES.get(l, {}).get("review_pending")],
        "text": out,
    }


# --------------------------------------------------------------------------
def build_alert_feed(state: dict, limit: int = 40,
                     include_isolation: bool = True) -> List[dict]:
    """
    The control-room feed. Ordered so the thing that matters most is on top:
    blocked before risky, and within each, the segments whose failure isolates
    the most people first.
    """
    from .routing import criticality

    segs = [s for s in state["segments"].values() if s["risk_label"] > 0]
    segs.sort(key=lambda s: (-s["risk_label"], -s["severity"]))

    feed: List[dict] = []
    for s in segs[:limit]:
        feed.append(segment_alert(s))

    if include_isolation:
        try:
            crit = criticality(state["ts"])
            for r in crit["ranked_by_live_exposure"][:6]:
                if r["isolates_nodes"] > 0 and r["current_risk_status"] != "safe":
                    names = ", ".join(r["isolated_names"][:4])
                    feed.insert(0, isolation_alert(
                        names or r["name"], r["corridor"], r["state"]))
        except Exception:
            pass

    return [f for f in feed if f]


def persist(alerts: List[dict]) -> None:
    rows = [(a["alert_id"], a["created_at"], a["severity"], a["scope"],
             a["target"], a["text"]["en"]["title"], a["text"]["en"]["body"],
             json.dumps(a), 0) for a in alerts]
    db.executemany(
        "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?)", rows)
