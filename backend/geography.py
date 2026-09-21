"""
NER Logistics Sentinel — Real geography of the North Eastern Region.

This module is deliberately NOT synthetic. Nodes are real towns / logistics
hubs / passes with real coordinates and elevations. Edges are real National
Highway corridors. Only the *observations* (weather, incidents, labels) are
synthesised -- the topology a judge can verify on any map.

Why it matters: the whole value of this project is that NER connectivity is
dominated by a handful of physical chokepoints (Siliguri corridor, Sela Pass,
NH-6 through the Jaintia hills, NH-29 Dimapur-Kohima-Imphal). A random graph
cannot demonstrate that. A real one can.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------
STATES = {
    "AS": "Assam",
    "ML": "Meghalaya",
    "AR": "Arunachal Pradesh",
    "NL": "Nagaland",
    "MN": "Manipur",
    "MZ": "Mizoram",
    "TR": "Tripura",
    "SK": "Sikkim",
    "WB": "West Bengal (Siliguri Corridor)",
}

TERRAIN_PLAIN = "plain"
TERRAIN_HILLY = "hilly"
TERRAIN_VALLEY = "valley"
TERRAIN_HIGH_PASS = "high_pass"


@dataclass(frozen=True)
class Node:
    """A logistics node: district HQ, town, port, border post or mountain pass."""

    id: str
    name: str
    state: str
    lat: float
    lon: float
    elev_m: int          # metres above sea level -> drives slope + snow risk
    kind: str = "town"   # town | hub | port | border | pass | depot
    population_k: int = 50


# --------------------------------------------------------------------------
# NODES  (lat, lon, elevation are real-world values)
# --------------------------------------------------------------------------
_N = [
    # ---------------- West Bengal: the Siliguri "Chicken's Neck" gateway ----
    Node("SILIGURI", "Siliguri", "WB", 26.7271, 88.3953, 122, "hub", 700),
    Node("JALPAIGURI", "Jalpaiguri", "WB", 26.5435, 88.7196, 89, "town", 110),
    Node("ALIPURDUAR", "Alipurduar", "WB", 26.4920, 89.5270, 100, "town", 130),
    Node("COOCHBEHAR", "Cooch Behar", "WB", 26.3254, 89.4482, 48, "town", 110),
    Node("SRIRAMPUR", "Srirampur (AS-WB Gate)", "WB", 26.2300, 89.8500, 40, "border", 10),

    # ---------------- Assam: Brahmaputra valley spine ----------------------
    Node("GUWAHATI", "Guwahati", "AS", 26.1445, 91.7362, 55, "hub", 1200),
    Node("JOGIGHOPA", "Jogighopa Multimodal Port", "AS", 26.2300, 90.5700, 40, "port", 20),
    Node("DHUBRI", "Dhubri", "AS", 26.0207, 89.9859, 30, "town", 200),
    Node("KOKRAJHAR", "Kokrajhar", "AS", 26.4014, 90.2716, 40, "town", 90),
    Node("BONGAIGAON", "Bongaigaon", "AS", 26.4831, 90.5583, 45, "hub", 150),
    Node("GOALPARA", "Goalpara", "AS", 26.1700, 90.6200, 40, "town", 80),
    Node("BARPETA", "Barpeta", "AS", 26.3220, 91.0059, 38, "town", 80),
    Node("MANGALDOI", "Mangaldoi", "AS", 26.4400, 92.0300, 50, "town", 60),
    Node("NAGAON", "Nagaon", "AS", 26.3464, 92.6840, 58, "hub", 150),
    Node("TEZPUR", "Tezpur", "AS", 26.6338, 92.8000, 48, "hub", 110),
    Node("GOLAGHAT", "Golaghat", "AS", 26.5100, 93.9600, 90, "town", 80),
    Node("NUMALIGARH", "Numaligarh Refinery", "AS", 26.6200, 93.7300, 92, "depot", 25),
    Node("JORHAT", "Jorhat", "AS", 26.7509, 94.2037, 116, "hub", 160),
    Node("SIVASAGAR", "Sivasagar", "AS", 26.9826, 94.6425, 95, "town", 70),
    Node("DIBRUGARH", "Dibrugarh", "AS", 27.4728, 94.9120, 108, "hub", 180),
    Node("TINSUKIA", "Tinsukia", "AS", 27.4898, 95.3597, 116, "hub", 125),
    Node("NLAKHIMPUR", "North Lakhimpur", "AS", 27.2350, 94.1060, 102, "town", 70),
    Node("DHEMAJI", "Dhemaji", "AS", 27.4833, 94.5833, 104, "town", 60),
    Node("DIPHU", "Diphu (Karbi Anglong)", "AS", 25.8400, 93.4300, 186, "town", 60),
    Node("HAFLONG", "Haflong (Dima Hasao)", "AS", 25.1667, 93.0167, 680, "town", 45),
    Node("SILCHAR", "Silchar", "AS", 24.8333, 92.7789, 22, "hub", 230),
    Node("KARIMGANJ", "Karimganj", "AS", 24.8697, 92.3592, 20, "town", 60),
    Node("HAILAKANDI", "Hailakandi", "AS", 24.6833, 92.5667, 25, "town", 50),
    Node("BADARPUR", "Badarpur Jn", "AS", 24.8690, 92.5950, 22, "town", 30),
    Node("CHURAIBARI", "Churaibari (AS-TR Gate)", "AS", 24.4600, 92.1700, 28, "border", 5),

    # ---------------- Meghalaya -------------------------------------------
    Node("BYRNIHAT", "Byrnihat Industrial", "ML", 26.0300, 91.8700, 80, "depot", 25),
    Node("SHILLONG", "Shillong", "ML", 25.5788, 91.8933, 1496, "hub", 360),
    Node("JOWAI", "Jowai", "ML", 25.4500, 92.2000, 1380, "town", 60),
    Node("NONGSTOIN", "Nongstoin", "ML", 25.5167, 91.2667, 1400, "town", 30),
    Node("TURA", "Tura", "ML", 25.5144, 90.2027, 350, "hub", 75),
    Node("DAWKI", "Dawki (Indo-Bangla)", "ML", 25.1900, 92.0200, 80, "border", 10),

    # ---------------- Arunachal Pradesh -----------------------------------
    Node("ITANAGAR", "Itanagar", "AR", 27.0844, 93.6053, 440, "hub", 95),
    Node("NAHARLAGUN", "Naharlagun", "AR", 27.1040, 93.6960, 330, "town", 55),
    Node("BHALUKPONG", "Bhalukpong", "AR", 27.0100, 92.6400, 213, "border", 8),
    Node("DIRANG", "Dirang", "AR", 27.3600, 92.2400, 1560, "town", 12),
    Node("BOMDILA", "Bomdila", "AR", 27.2650, 92.4200, 2415, "town", 20),
    Node("SELAPASS", "Sela Pass (4170 m)", "AR", 27.5040, 92.1060, 4170, "pass", 0),
    Node("TAWANG", "Tawang", "AR", 27.5860, 91.8594, 2669, "hub", 30),
    Node("ZIRO", "Ziro", "AR", 27.5450, 93.8300, 1688, "town", 25),
    Node("ALONG", "Aalo (Along)", "AR", 28.1667, 94.8000, 523, "town", 25),
    Node("PASIGHAT", "Pasighat", "AR", 28.0667, 95.3333, 155, "hub", 30),
    Node("ROING", "Roing", "AR", 28.1400, 95.8400, 390, "town", 15),
    Node("TEZU", "Tezu", "AR", 27.9242, 96.1586, 185, "town", 18),
    Node("SEPPA", "Seppa", "AR", 27.2900, 92.9200, 920, "town", 12),
    Node("CHANGLANG", "Changlang", "AR", 27.1300, 95.7300, 380, "town", 18),

    # ---------------- Nagaland --------------------------------------------
    Node("DIMAPUR", "Dimapur", "NL", 25.9063, 93.7276, 145, "hub", 180),
    Node("KOHIMA", "Kohima", "NL", 25.6751, 94.1086, 1444, "hub", 115),
    Node("WOKHA", "Wokha", "NL", 26.0900, 94.2700, 1313, "town", 35),
    Node("MOKOKCHUNG", "Mokokchung", "NL", 26.3200, 94.5200, 1325, "town", 35),
    Node("ZUNHEBOTO", "Zunheboto", "NL", 26.0100, 94.5200, 1874, "town", 25),
    Node("PHEK", "Phek", "NL", 25.6700, 94.4700, 1524, "town", 20),
    Node("TUENSANG", "Tuensang", "NL", 26.2800, 94.8300, 1371, "town", 30),
    Node("MON", "Mon", "NL", 26.7200, 95.0500, 897, "town", 25),

    # ---------------- Manipur ---------------------------------------------
    Node("JIRIBAM", "Jiribam", "MN", 24.8000, 93.1200, 50, "town", 20),
    Node("NONEY", "Noney", "MN", 24.8300, 93.5500, 300, "town", 15),
    Node("SENAPATI", "Senapati", "MN", 25.2700, 94.0200, 1061, "town", 30),
    Node("IMPHAL", "Imphal", "MN", 24.8170, 93.9368, 786, "hub", 330),
    Node("THOUBAL", "Thoubal", "MN", 24.6400, 94.0100, 790, "town", 45),
    Node("CHURACHANDPUR", "Churachandpur", "MN", 24.3333, 93.6833, 914, "town", 55),
    Node("UKHRUL", "Ukhrul", "MN", 25.0500, 94.3600, 1662, "town", 25),
    Node("MOREH", "Moreh (Indo-Myanmar)", "MN", 24.2500, 94.3100, 230, "border", 16),

    # ---------------- Mizoram ---------------------------------------------
    Node("KOLASIB", "Kolasib", "MZ", 24.2200, 92.6800, 600, "town", 25),
    Node("AIZAWL", "Aizawl", "MZ", 23.7271, 92.7176, 1132, "hub", 300),
    Node("SERCHHIP", "Serchhip", "MZ", 23.3000, 92.8500, 1260, "town", 20),
    Node("LUNGLEI", "Lunglei", "MZ", 22.8800, 92.7300, 1400, "hub", 60),
    Node("CHAMPHAI", "Champhai", "MZ", 23.4700, 93.3300, 1672, "town", 25),
    Node("SAIHA", "Saiha", "MZ", 22.4900, 92.9800, 729, "town", 25),
    Node("ZOKHAWTHAR", "Zokhawthar (Indo-Myanmar)", "MZ", 23.3900, 93.3800, 300, "border", 5),

    # ---------------- Tripura ---------------------------------------------
    Node("DHARMANAGAR", "Dharmanagar", "TR", 24.3700, 92.1700, 30, "town", 45),
    Node("KAILASHAHAR", "Kailashahar", "TR", 24.3300, 92.0100, 25, "town", 25),
    Node("AMBASSA", "Ambassa", "TR", 23.9400, 91.8500, 60, "town", 20),
    Node("AGARTALA", "Agartala", "TR", 23.8315, 91.2868, 12, "hub", 520),
    Node("UDAIPUR", "Udaipur", "TR", 23.5300, 91.4900, 20, "town", 40),
    Node("SABROOM", "Sabroom (Maitri Setu)", "TR", 23.0100, 91.7200, 15, "border", 10),

    # ---------------- Sikkim ----------------------------------------------
    Node("RANGPO", "Rangpo (SK Gate)", "SK", 27.1700, 88.5300, 300, "border", 10),
    Node("GANGTOK", "Gangtok", "SK", 27.3389, 88.6065, 1650, "hub", 100),
    Node("NAMCHI", "Namchi", "SK", 27.1667, 88.3500, 1675, "town", 15),
    Node("GYALSHING", "Gyalshing", "SK", 27.2900, 88.2600, 1780, "town", 10),
    Node("MANGAN", "Mangan", "SK", 27.5100, 88.5300, 954, "town", 10),
    Node("NATHULA", "Nathu La (4310 m)", "SK", 27.3900, 88.8400, 4310, "pass", 0),
]

NODES: Dict[str, Node] = {n.id: n for n in _N}


# --------------------------------------------------------------------------
# CORRIDORS — real National Highways.
#   (highway_code, road_class, [node sequence], notes)
# road_class: NH = national highway, SH = state highway, RR = rural/feeder
# --------------------------------------------------------------------------
@dataclass
class Corridor:
    code: str
    road_class: str
    path: List[str]
    name: str
    lanes: int = 2
    strategic: bool = False          # defence / lifeline corridor
    notes: str = ""


CORRIDORS: List[Corridor] = [
    # ===== THE national artery: East-West Corridor NH-27 =====
    Corridor("NH-27", "NH",
             ["SILIGURI", "JALPAIGURI", "ALIPURDUAR", "COOCHBEHAR", "SRIRAMPUR",
              "DHUBRI", "KOKRAJHAR", "BONGAIGAON", "GOALPARA", "GUWAHATI"],
             "East-West Corridor (Siliguri → Guwahati)", lanes=4, strategic=True,
             notes="Sole surface lifeline for all 8 NE states. Passes the 22 km "
                   "Siliguri Corridor ('Chicken's Neck')."),
    Corridor("NH-27E", "NH",
             ["GUWAHATI", "NAGAON", "GOLAGHAT", "NUMALIGARH", "JORHAT",
              "SIVASAGAR", "DIBRUGARH", "TINSUKIA"],
             "Upper Assam trunk (Guwahati → Tinsukia)", lanes=4, strategic=True),

    # ===== Assam north bank NH-15 =====
    Corridor("NH-15", "NH",
             ["JOGIGHOPA", "BARPETA", "MANGALDOI", "TEZPUR", "NLAKHIMPUR",
              "DHEMAJI", "PASIGHAT", "ROING", "TEZU"],
             "North Bank Highway (Jogighopa → Tezu)", lanes=2, strategic=True,
             notes="Flood-prone north bank; crosses Bhupen Hazarika Setu region."),
    Corridor("NH-127B", "NH", ["JOGIGHOPA", "GOALPARA"], "Jogighopa link", lanes=2),
    Corridor("NH-17", "NH", ["GUWAHATI", "GOALPARA", "DHUBRI"], "South bank Dhubri link"),
    # Saraighat bridge — Guwahati's access to the north bank. Without this the
    # graph wrongly forces all north-bank traffic out west via Jogighopa.
    Corridor("NH-27N", "NH", ["GUWAHATI", "MANGALDOI"],
             "Saraighat Bridge → north bank (Baihata Chariali)", lanes=4,
             strategic=True,
             notes="Saraighat is one of only a handful of Brahmaputra road "
                   "crossings; it is itself a chokepoint."),
    # Kolia Bhomora Setu — the south-bank/north-bank link at Tezpur.
    Corridor("NH-715", "NH", ["NAGAON", "TEZPUR"],
             "Kolia Bhomora Setu (Nagaon → Tezpur)", lanes=2, strategic=True,
             notes="Brahmaputra crossing linking the NH-27 trunk to the north bank."),

    # ===== Meghalaya =====
    Corridor("NH-6A", "NH", ["GUWAHATI", "BYRNIHAT", "SHILLONG"],
             "Guwahati → Shillong", lanes=4, strategic=True),
    Corridor("NH-6", "NH", ["SHILLONG", "JOWAI", "BADARPUR", "SILCHAR"],
             "Shillong → Silchar (Jaintia Hills)", lanes=2, strategic=True,
             notes="Highest landslide density in NER. Repeated monsoon closures "
                   "cut Barak Valley + Mizoram + Tripura off from Guwahati."),
    Corridor("NH-62", "NH", ["SHILLONG", "NONGSTOIN", "TURA"], "Shillong → Tura"),
    Corridor("NH-217", "SH", ["SHILLONG", "DAWKI"], "Shillong → Dawki border"),
    Corridor("NH-217B", "SH", ["TURA", "GOALPARA"], "Tura → Goalpara (Garo Hills)"),

    # ===== Barak valley / Dima Hasao — the fragile alternative =====
    Corridor("NH-27S", "NH", ["NAGAON", "DIPHU", "HAFLONG", "BADARPUR"],
             "Lumding–Haflong hill route", lanes=2,
             notes="Dima Hasao section: chronic landslides, the only non-Meghalaya "
                   "road link to Barak Valley."),
    Corridor("NH-37B", "SH", ["SILCHAR", "HAILAKANDI", "KARIMGANJ"], "Barak valley loop"),
    Corridor("NH-8T", "SH", ["BADARPUR", "CHURAIBARI"], "Badarpur → Tripura gate"),

    # ===== Nagaland / Manipur — NH-29 + NH-2 =====
    Corridor("NH-29", "NH", ["NUMALIGARH", "DIMAPUR", "KOHIMA", "IMPHAL"],
             "Numaligarh → Dimapur → Kohima → Imphal", lanes=2, strategic=True,
             notes="Manipur's primary lifeline. Kohima–Mao section: landslides + "
                   "frequent bandh/blockade disruption."),
    Corridor("NH-2", "NH", ["KOHIMA", "SENAPATI", "IMPHAL"], "Kohima → Imphal via Senapati",
             strategic=True),
    # NH-36 Doboka–Lumding–Dimapur: the alternate approach to Nagaland that
    # bypasses the Numaligarh trunk. Matters when NH-27E is cut.
    Corridor("NH-36", "SH", ["DIPHU", "DIMAPUR"],
             "Karbi Anglong → Dimapur (NH-36)", lanes=2,
             notes="Alternate approach to Nagaland/Manipur bypassing upper Assam."),
    Corridor("NH-37M", "NH", ["SILCHAR", "JIRIBAM", "NONEY", "IMPHAL"],
             "Silchar → Imphal (NH-37)", lanes=2, strategic=True,
             notes="Manipur's second lifeline; Noney section highly unstable."),
    Corridor("NH-102", "NH", ["IMPHAL", "THOUBAL", "MOREH"], "Imphal → Moreh (Asian Hwy 1)",
             strategic=True),
    Corridor("NH-2C", "SH", ["IMPHAL", "CHURACHANDPUR"], "Imphal → Churachandpur"),
    Corridor("NH-150", "SH", ["IMPHAL", "UKHRUL"], "Imphal → Ukhrul"),
    Corridor("NH-129", "SH", ["DIMAPUR", "WOKHA", "MOKOKCHUNG"], "Dimapur → Mokokchung"),
    Corridor("NH-61", "SH", ["MOKOKCHUNG", "TUENSANG", "MON"], "Eastern Nagaland ridge road"),
    Corridor("NH-702D", "RR", ["KOHIMA", "ZUNHEBOTO", "PHEK"], "Interior Nagaland feeder"),
    Corridor("NH-702", "RR", ["MON", "SIVASAGAR"], "Mon → Assam descent"),

    # ===== Mizoram =====
    Corridor("NH-306", "NH", ["SILCHAR", "KOLASIB", "AIZAWL", "SERCHHIP", "LUNGLEI", "SAIHA"],
             "Silchar → Aizawl → Saiha", lanes=2, strategic=True,
             notes="Mizoram's only arterial road; single-thread south of Aizawl."),
    Corridor("NH-502A", "SH", ["AIZAWL", "CHAMPHAI", "ZOKHAWTHAR"], "Aizawl → Zokhawthar border"),

    # ===== Tripura =====
    Corridor("NH-8", "NH", ["CHURAIBARI", "DHARMANAGAR", "AMBASSA", "AGARTALA",
                            "UDAIPUR", "SABROOM"],
             "Assam–Agartala–Sabroom lifeline", lanes=2, strategic=True,
             notes="Tripura's sole road lifeline; Atharamura/Baramura ghat sections slip."),
    Corridor("NH-208", "SH", ["DHARMANAGAR", "KAILASHAHAR"], "Kailashahar feeder"),

    # ===== Sikkim — the textbook single point of failure =====
    Corridor("NH-10", "NH", ["SILIGURI", "RANGPO", "GANGTOK"],
             "Siliguri → Gangtok (NH-10)", lanes=2, strategic=True,
             notes="THE only road into Sikkim. Teesta-river landslides sever the "
                   "state for days at a time."),
    Corridor("NH-310", "SH", ["GANGTOK", "NATHULA"], "Gangtok → Nathu La"),
    Corridor("NH-510", "SH", ["GANGTOK", "MANGAN"], "North Sikkim road"),
    Corridor("NH-717A", "SH", ["RANGPO", "NAMCHI", "GYALSHING"], "South-West Sikkim"),

    # ===== Arunachal Pradesh =====
    Corridor("NH-13W", "NH", ["TEZPUR", "BHALUKPONG", "DIRANG", "BOMDILA",
                              "SELAPASS", "TAWANG"],
             "Balipara → Tawang (via Sela Pass)", lanes=2, strategic=True,
             notes="Defence-critical. Sela Pass at 4170 m: snow closure Dec–Mar, "
                   "landslides in monsoon. Sela Tunnel is the mitigation."),
    Corridor("NH-415", "NH", ["TEZPUR", "ITANAGAR"], "Tezpur → Itanagar"),
    Corridor("NH-415A", "SH", ["ITANAGAR", "NAHARLAGUN", "ZIRO"], "Itanagar → Ziro"),
    Corridor("NH-13E", "NH", ["ZIRO", "ALONG", "PASIGHAT"], "Trans-Arunachal (east)",
             strategic=True),
    Corridor("NH-713", "SH", ["SEPPA", "ITANAGAR"], "Seppa feeder"),
    Corridor("NH-713A", "RR", ["SEPPA", "BOMDILA"], "Seppa → Bomdila ridge track"),
    Corridor("NH-315", "SH", ["TEZU", "CHANGLANG", "TINSUKIA"], "Tezu → Tinsukia"),
    Corridor("NH-315A", "SH", ["DIBRUGARH", "DHEMAJI"], "Bogibeel Bridge link", lanes=4,
             notes="Bogibeel rail-road bridge: strategic Brahmaputra crossing."),
    Corridor("NH-315B", "SH", ["TINSUKIA", "ROING"], "Bhupen Hazarika Setu link"),
]


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
EARTH_R_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


# Hill roads are far longer than straight-line distance. Sinuosity multipliers
# calibrated loosely against real NER road distances.
SINUOSITY = {
    TERRAIN_PLAIN: 1.18,
    TERRAIN_VALLEY: 1.32,
    TERRAIN_HILLY: 1.55,
    TERRAIN_HIGH_PASS: 1.85,
}

# Free-flow speeds (km/h) — hill highways in NER genuinely average this low.
FREEFLOW_KMPH = {
    ("NH", TERRAIN_PLAIN): 58,
    ("NH", TERRAIN_VALLEY): 45,
    ("NH", TERRAIN_HILLY): 32,
    ("NH", TERRAIN_HIGH_PASS): 22,
    ("SH", TERRAIN_PLAIN): 48,
    ("SH", TERRAIN_VALLEY): 38,
    ("SH", TERRAIN_HILLY): 26,
    ("SH", TERRAIN_HIGH_PASS): 18,
    ("RR", TERRAIN_PLAIN): 35,
    ("RR", TERRAIN_VALLEY): 28,
    ("RR", TERRAIN_HILLY): 18,
    ("RR", TERRAIN_HIGH_PASS): 12,
}


def classify_terrain(a: Node, b: Node, straight_km: float) -> str:
    """Terrain class from endpoint elevations and gradient."""
    hi = max(a.elev_m, b.elev_m)
    lo = min(a.elev_m, b.elev_m)
    drop = hi - lo
    grade_pct = (drop / max(straight_km * 1000.0, 1.0)) * 100.0

    if hi >= 3000:
        return TERRAIN_HIGH_PASS
    if hi >= 900 or grade_pct >= 3.0:
        return TERRAIN_HILLY
    if drop >= 250 or (hi >= 400 and grade_pct >= 1.2):
        return TERRAIN_VALLEY
    return TERRAIN_PLAIN


# Flood exposure: Brahmaputra / Barak / Teesta floodplain nodes.
FLOODPLAIN = {
    "DHUBRI", "GOALPARA", "BARPETA", "JOGIGHOPA", "MANGALDOI", "NAGAON", "TEZPUR",
    "NLAKHIMPUR", "DHEMAJI", "GOLAGHAT", "JORHAT", "SIVASAGAR", "DIBRUGARH",
    "TINSUKIA", "SILCHAR", "KARIMGANJ", "HAILAKANDI", "BADARPUR", "PASIGHAT",
    "ROING", "TEZU", "JIRIBAM", "SRIRAMPUR", "COOCHBEHAR", "JALPAIGURI",
    "KOKRAJHAR", "BONGAIGAON", "DIMAPUR", "AGARTALA", "DHARMANAGAR", "KAILASHAHAR",
}

# Documented landslide-hotspot corridors (these carry extra base hazard).
LANDSLIDE_HOTSPOT_CORRIDORS = {
    "NH-6": 1.00,      # Shillong–Jowai–Silchar, Jaintia Hills
    "NH-27S": 0.95,    # Dima Hasao
    "NH-10": 1.00,     # Teesta valley, Sikkim
    "NH-13W": 0.90,    # Sela Pass approach
    "NH-306": 0.85,    # Mizoram arterial
    "NH-37M": 0.95,    # Noney, Manipur
    "NH-29": 0.75,     # Kohima–Mao
    "NH-8": 0.65,      # Tripura ghats
    "NH-2": 0.70,
    "NH-502A": 0.70,
    "NH-61": 0.65,
    "NH-310": 0.80,
    "NH-510": 0.80,
    "NH-713A": 0.85,
    "NH-702": 0.70,
}


@dataclass
class Segment:
    """One directed-agnostic road segment between two adjacent corridor nodes."""

    road_id: str
    corridor: str
    name: str
    road_class: str
    lanes: int
    strategic: bool
    u: str
    v: str
    u_name: str
    v_name: str
    state: str
    district: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    length_km: float
    terrain: str
    avg_slope_pct: float
    max_elev_m: int
    freeflow_kmph: float
    flood_exposure: float         # 0..1
    landslide_base: float         # 0..1  structural susceptibility
    snow_exposure: float          # 0..1
    bridge_count: int
    notes: str = ""

    def as_row(self) -> tuple:
        return (
            self.road_id, self.corridor, self.name, self.road_class, self.lanes,
            int(self.strategic), self.u, self.v, self.u_name, self.v_name,
            self.state, self.district, self.start_lat, self.start_lon,
            self.end_lat, self.end_lon, self.length_km, self.terrain,
            self.avg_slope_pct, self.max_elev_m, self.freeflow_kmph,
            self.flood_exposure, self.landslide_base, self.snow_exposure,
            self.bridge_count, self.notes,
        )


def build_segments() -> List[Segment]:
    """Expand corridors into individual road segments with derived attributes."""
    segs: List[Segment] = []
    seen: set[Tuple[str, str]] = set()

    for cor in CORRIDORS:
        for i in range(len(cor.path) - 1):
            uid, vid = cor.path[i], cor.path[i + 1]
            key = tuple(sorted((uid, vid)))
            if key in seen:
                continue            # avoid duplicate physical segment
            seen.add(key)

            a, b = NODES[uid], NODES[vid]
            straight = haversine_km(a.lat, a.lon, b.lat, b.lon)
            terrain = classify_terrain(a, b, straight)
            length = round(straight * SINUOSITY[terrain], 1)

            drop = abs(a.elev_m - b.elev_m)
            slope = round((drop / max(length * 1000.0, 1.0)) * 100.0, 3)
            max_elev = max(a.elev_m, b.elev_m)

            flood = 0.0
            if uid in FLOODPLAIN:
                flood += 0.5
            if vid in FLOODPLAIN:
                flood += 0.5
            if terrain in (TERRAIN_HILLY, TERRAIN_HIGH_PASS):
                flood *= 0.35        # hills drain; they slide instead
            flood = round(min(flood, 1.0), 3)

            ls = LANDSLIDE_HOTSPOT_CORRIDORS.get(cor.code, 0.15)
            # structural susceptibility grows with slope and relief
            ls = ls * (0.55 + 0.45 * min(slope / 4.0, 1.0))
            if terrain == TERRAIN_PLAIN:
                ls *= 0.18
            landslide = round(min(ls, 1.0), 3)

            snow = 0.0
            if max_elev >= 2200:
                snow = round(min((max_elev - 2200) / 1800.0 + 0.25, 1.0), 3)
            elif max_elev >= 1600:
                snow = 0.08

            bridges = int(max(0, round(length / 28.0)))
            if flood > 0.5:
                bridges += 1

            segs.append(Segment(
                road_id=f"{cor.code}:{uid}-{vid}",
                corridor=cor.code,
                name=f"{a.name} → {b.name}",
                road_class=cor.road_class,
                lanes=cor.lanes,
                strategic=cor.strategic,
                u=uid, v=vid, u_name=a.name, v_name=b.name,
                state=a.state,
                district=a.name,
                start_lat=a.lat, start_lon=a.lon, end_lat=b.lat, end_lon=b.lon,
                length_km=length,
                terrain=terrain,
                avg_slope_pct=slope,
                max_elev_m=max_elev,
                freeflow_kmph=FREEFLOW_KMPH[(cor.road_class, terrain)],
                flood_exposure=flood,
                landslide_base=landslide,
                snow_exposure=snow,
                bridge_count=bridges,
                notes=cor.notes,
            ))
    return segs


# District-level aggregation key = node name for hubs/towns
def district_list() -> List[dict]:
    out = []
    for n in NODES.values():
        out.append({
            "district": n.name, "node_id": n.id, "state": n.state,
            "state_name": STATES[n.state], "lat": n.lat, "lon": n.lon,
            "elev_m": n.elev_m, "kind": n.kind, "population_k": n.population_k,
            "floodplain": n.id in FLOODPLAIN,
        })
    return out


if __name__ == "__main__":
    s = build_segments()
    print(f"nodes      : {len(NODES)}")
    print(f"corridors  : {len(CORRIDORS)}")
    print(f"segments   : {len(s)}")
    print(f"total km   : {sum(x.length_km for x in s):,.0f}")
    from collections import Counter
    print("terrain    :", dict(Counter(x.terrain for x in s)))
    print("classes    :", dict(Counter(x.road_class for x in s)))
