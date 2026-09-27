"""Text normalisation and field parsing for business names and addresses.

Everything here is country-agnostic: the same rules and hand-written dictionaries are
applied to every record whatever its country label. All dictionaries below are written
by hand in code (no downloaded gazetteers / city lists).

Run as a script to normalise a split and cache it:
    python normalize.py --split train|test
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import jellyfish
import polars as pl
from unidecode import unidecode

import config as C

# =====================================================================================
# Hand-written dictionaries
# =====================================================================================

# Legal-form tokens (after cleaning; dots already removed so "l.l.c." -> "llc").
LEGAL_TOKENS = {
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "inc": "inc", "incorporated": "inc", "llc": "llc", "llp": "llp", "lp": "lp",
    "plc": "plc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "cie": "co", "tbk": "tbk", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "pllc": "llc", "ltda": "ltd", "opc": "opc",
    # Indic-script legal forms as they come out of unidecode (e.g. प्राइवेट लिमिटेड, प्रा. लि.)
    "limittedd": "ltd", "limittett": "ltd", "limitedd": "ltd", "limirrrrdd": "ltd",
    "praaivett": "pvt", "praiveett": "pvt", "praaibhett": "pvt", "piraiveett": "pvt",
    "praivrrrr": "pvt", "praivett": "pvt", "praa": "pvt", "li": "ltd",
    "elelpii": "llp", "elelpi": "llp",
    # French legal forms
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "sa": "sa",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "sca": "sca", "scs": "scs",
    "gie": "gie", "scm": "scm", "scea": "scea", "earl": "earl",
}
# Filler / honorific tokens removed from the name core.
NAME_STOP = {
    "the", "and", "of", "et", "de", "des", "du", "la", "le", "les", "d", "l", "a", "an",
    "ms", "mr", "mrs", "smt", "shri", "sri", "shree", "shrii", "messrs", "m", "s",
}
DOMAIN_RE = re.compile(
    r"^(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|in|co\.in|net|org|co|fr|biz|info|co\.uk|us|io|org\.in|net\.in)$")

# Address abbreviations -> canonical token (token level, after cleaning).
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ter": "terrace", "sq": "square", "trl": "trail",
    "pky": "parkway", "ctr": "center", "cntr": "center", "centre": "center",
    "mt": "mount", "ft": "fort", "pt": "point", "hts": "heights", "jct": "junction",
    "expy": "expressway", "fwy": "freeway", "tpke": "turnpike", "rte": "route",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "opp": "opposite", "nr": "near", "bh": "behind", "nagr": "nagar", "clny": "colony",
    "r": "rue", "all": "allee", "che": "chemin", "chem": "chemin", "ch": "chemin",
    "imp": "impasse", "fbg": "faubourg", "crs": "cours", "pte": "porte", "sq": "square",
    "rpt": "rondpoint", "lot": "lot", "res": "residence", "qu": "quai",
    "mg": "marg", "ngr": "nagar", "extn": "extension", "ext": "extension",
    "sec": "sector", "sect": "sector", "ph": "phase", "stn": "station",
    "bldg": "building", "apts": "apartments",
}
# Tokens dropped from the address core (house-number prefixes, unit designators, nulls).
ADDR_DROP = {
    "no", "num", "number", "nos", "h", "hno", "house", "door", "dno", "unit", "apt",
    "apartment", "ste", "suite", "suit", "fl", "flr", "floor", "rm", "room", "rom",
    "null", "none", "na", "nil", "n/a", "bis", "ter", "flat", "plot", "shop", "office",
    "etage", "bat", "batiment", "ndeg",
}
STREET_TYPES = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "court", "place", "highway",
    "parkway", "circle", "terrace", "square", "way", "trail", "loop", "pike", "alley",
    "route", "rue", "allee", "chemin", "impasse", "faubourg", "cours", "quai", "marg",
    "expressway", "freeway", "turnpike", "path", "row", "crescent", "grove", "plaza",
    "residence", "passage", "sentier", "voie", "cite", "promenade",
}
LANDMARK_WORDS = {"near", "opposite", "behind", "beside", "adjacent", "nxt"}
CITY_ALIAS = {
    "calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai",
    "bangalore": "bengaluru", "poona": "pune", "gurgaon": "gurugram",
    "trivandrum": "thiruvananthapuram", "baroda": "vadodara", "mysore": "mysuru",
    "mangalore": "mangaluru", "cochin": "kochi", "pondicherry": "puducherry",
    "banaras": "varanasi", "benares": "varanasi", "allahabad": "prayagraj",
    "orissa": "odisha", "belgaum": "belagavi", "hubli": "hubballi", "vizag": "visakhapatnam",
}
# State / province full names -> short code. Codes only need to be unique *within* a
# country because records are only ever compared inside the same country label.
STATE_NAMES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "washington dc": "dc",
    # India
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "or", "orissa": "or",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "andaman and nicobar islands": "an",
    "andaman and nicobar": "an", "chandigarh": "ch", "dadra and nagar haveli": "dn",
    "daman and diu": "dn", "dadra and nagar haveli and daman and diu": "dn", "delhi": "dl",
    "nct of delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "lakshadweep": "ld",
    "puducherry": "py",
}
# France: regions and their departements map to one region code, so a record giving the
# departement ("Nord") agrees with one giving the region ("Hauts-de-France").
FR_REGIONS = {
    "ara": ("Auvergne-Rhone-Alpes", "Ain", "Allier", "Ardeche", "Cantal", "Drome", "Isere", "Loire",
            "Haute-Loire", "Puy-de-Dome", "Rhone", "Savoie", "Haute-Savoie"),
    "bfc": ("Bourgogne-Franche-Comte", "Cote-d'Or", "Doubs", "Jura", "Nievre", "Haute-Saone",
            "Saone-et-Loire", "Yonne", "Territoire de Belfort"),
    "bre": ("Bretagne", "Cotes-d'Armor", "Finistere", "Ille-et-Vilaine", "Morbihan"),
    "cvl": ("Centre-Val de Loire", "Cher", "Eure-et-Loir", "Indre", "Indre-et-Loire", "Loir-et-Cher", "Loiret"),
    "cor": ("Corse", "Corse-du-Sud", "Haute-Corse"),
    "ges": ("Grand Est", "Ardennes", "Aube", "Marne", "Haute-Marne", "Meurthe-et-Moselle", "Meuse",
            "Moselle", "Bas-Rhin", "Haut-Rhin", "Vosges"),
    "hdf": ("Hauts-de-France", "Aisne", "Nord", "Oise", "Pas-de-Calais", "Somme"),
    "idf": ("Ile-de-France", "Paris", "Seine-et-Marne", "Yvelines", "Essonne", "Hauts-de-Seine",
            "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise"),
    "nor": ("Normandie", "Calvados", "Eure", "Manche", "Orne", "Seine-Maritime"),
    "naq": ("Nouvelle-Aquitaine", "Charente", "Charente-Maritime", "Correze", "Creuse", "Dordogne",
            "Gironde", "Landes", "Lot-et-Garonne", "Pyrenees-Atlantiques", "Deux-Sevres", "Vienne",
            "Haute-Vienne"),
    "occ": ("Occitanie", "Ariege", "Aude", "Aveyron", "Gard", "Haute-Garonne", "Gers", "Herault", "Lot",
            "Lozere", "Hautes-Pyrenees", "Pyrenees-Orientales", "Tarn", "Tarn-et-Garonne"),
    "pdl": ("Pays de la Loire", "Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendee"),
    "pac": ("Provence-Alpes-Cote d'Azur", "PACA", "Alpes-de-Haute-Provence", "Hautes-Alpes",
            "Alpes-Maritimes", "Bouches-du-Rhone", "Var", "Vaucluse"),
}
# Indian state names written in their own scripts (converted with the same cleaning below).
NATIVE_STATES = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn",
    "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "ts", "हरियाणा": "hr", "राजस्थान": "rj",
    "കേരളം": "kl", "बिहार": "br", "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb",
    "ଓଡ଼ିଶା": "or", "अসম": "as", "অসম": "as", "झारखंड": "jh", "छत्तीसगढ़": "cg", "उत्तराखंड": "uk",
    "हिमाचल प्रदेश": "hp", "गोवा": "ga", "पुडुचेरी": "py", "புதுச்சேரி": "py",
}
STATE_CODE_ALIAS = {"od": "or", "tg": "ts", "dd": "dn"}
STATE_CODES = set(STATE_NAMES.values()) | set(STATE_CODE_ALIAS)

# =====================================================================================
# Regexes
# =====================================================================================
_WS = re.compile(r"\s+")
_DOTTED_ACR = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?(?![a-z])")
_NAME_PUNCT = re.compile(r"[^a-z0-9 ]+")
_ADDR_PUNCT = re.compile(r"[^a-z0-9/\- ]+")
_HYPH_WORD = re.compile(r"(?<=[a-z]{2})-|-(?=[a-z]{2})|(?<=\s)-|-(?=\s)|^-|-$")
_SLASH_EDGE = re.compile(r"(?<=\s)/|/(?=\s)|^/|/$")
_ORDINAL = re.compile(r"^0*(\d+)(st|nd|rd|th|eme|er|e)$")
_NUMTOK = re.compile(r"\d")
_PIN6 = re.compile(r"^\d{6}$")
_PIN33 = re.compile(r"\b(\d{3}) (\d{3})\b")
_LEAD0 = re.compile(r"(?<![0-9])0+(?=\d)")


_SKEL_SUBS = (("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
              ("sh", "s"), ("ch", "c"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"),
              ("w", "v"), ("z", "j"), ("y", "i"))
_VOWELS = str.maketrans("", "", "aeiou")
_REPEAT = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Script-independent consonant skeleton of a token (first letter kept, vowels dropped,
    aspirates / sibilants merged, repeats collapsed): 'products' and the transliterated
    'proddktts' both -> 'prdkts'."""
    if not tok or not tok.isalpha():
        return tok
    for a, b in _SKEL_SUBS:
        tok = tok.replace(a, b)
    return _REPEAT.sub(r"\1", tok[0] + tok[1:].translate(_VOWELS))


def to_ascii(s: str) -> str:
    """NFKC-normalise and transliterate any script to lower-case ASCII."""
    if not s:
        return ""
    if not s.isascii():
        s = unidecode(unicodedata.normalize("NFKC", s).replace("N°", "No ").replace("n°", "no ")
                                                   .replace("°", " "))
    return s.lower()


# =====================================================================================
# Names
# =====================================================================================
def normalize_name(raw: str) -> dict:
    """Parse one business name into cleaned / core / sorted / nospace / legal-form fields."""
    s = to_ascii(raw).strip()
    s = s.replace("&", " and ").replace("+", " plus ").replace("'", "").replace("`", "")
    stripped = s.strip(" #@<>-_*~!\"()[]{}")
    is_domain = 0
    dm = DOMAIN_RE.match(stripped)
    if dm:
        is_domain = 1
        s = dm.group(1).replace("-", "")
    elif s.lstrip().startswith("#") and " " not in stripped:
        is_domain = 1                                           # hashtag style: #empireprogram
        s = stripped
    s = _DOTTED_ACR.sub(lambda m: m.group(0).replace(".", ""), s)   # l.l.c. -> llc
    s = _NAME_PUNCT.sub(" ", s)
    toks = s.split()
    clean = " ".join(toks)
    legal = sorted({LEGAL_TOKENS[t] for t in toks if t in LEGAL_TOKENS})
    if "pvt" in legal and "ltd" in legal:
        legal = [x for x in legal if x not in ("pvt", "ltd")] + ["pvt_ltd"]
    core_toks = [t for t in toks if t not in LEGAL_TOKENS and t not in NAME_STOP]
    if not core_toks:
        core_toks = [t for t in toks if t not in NAME_STOP] or toks
    # collapse immediate repeats ("aarvraj aarvraj apparels")
    dedup = [t for i, t in enumerate(core_toks) if i == 0 or t != core_toks[i - 1]]
    core = " ".join(dedup)
    first = dedup[0] if dedup else ""
    return {
        "name_clean": clean,
        "name_core": core,
        "name_sorted": " ".join(sorted(set(dedup))),
        "name_nospace": core.replace(" ", ""),
        "legal_form": "|".join(sorted(legal)),
        "is_domain": is_domain,
        "name_acronym": "".join(t[0] for t in dedup) if len(dedup) >= 2 else "",
        "name_phon": jellyfish.metaphone(first)[:6] if first.isalpha() else first,
        "name_first": first,
        "name_skel": " ".join(skeleton(t) for t in dedup),
    }


# =====================================================================================
# Addresses
# =====================================================================================
def _clean_segment(seg: str) -> list[str]:
    """Clean one comma-separated address segment into canonical tokens."""
    seg = _ADDR_PUNCT.sub(" ", seg)
    seg = _HYPH_WORD.sub(" ", seg)
    seg = _SLASH_EDGE.sub(" ", seg)
    out = []
    for t in seg.split():
        t = t.strip("-/")
        if not t:
            continue
        m = _ORDINAL.match(t)
        if m:
            out.append(m.group(1) + m.group(2))
            continue
        if _NUMTOK.search(t):
            t = _LEAD0.sub("", t)
        else:
            t = ADDR_ABBR.get(t, t)
            t = CITY_ALIAS.get(t, t)
        out.append(t)
    return out


def _seg_key(name: str) -> str:
    """Canonical form of a whole address segment, exactly as normalize_address sees it."""
    return " ".join(_clean_segment(to_ascii(name).replace("'", "")))


for _code, _names in FR_REGIONS.items():
    for _n in _names:
        STATE_NAMES[_seg_key(_n)] = _code
for _n, _code in NATIVE_STATES.items():
    STATE_NAMES[_seg_key(_n)] = _code


def _state_of(seg_toks: list[str]) -> str:
    """Return canonical state code if the whole segment is a state name / code, else ''."""
    if not seg_toks:
        return ""
    s = " ".join(seg_toks)
    if s in STATE_NAMES:
        return STATE_NAMES[s]
    if len(seg_toks) == 1 and s in STATE_CODES:
        return STATE_CODE_ALIAS.get(s, s)
    return ""


def normalize_address(raw: str) -> dict:
    """Parse one address into core text, house number, street, city, state, postcode."""
    s = to_ascii(raw)
    s = s.replace("b/h", " behind ").replace("&", " and ").replace("'", "")
    s = s.replace("#", " ").replace(".", " ").replace(";", ",").replace(":", " ")
    s = s.replace("(", " ").replace(")", " ")
    postcode = ""
    m = _PIN33.search(s)
    if m:
        postcode = m.group(1) + m.group(2)
        s = s[:m.start()] + postcode + s[m.end():]
    segs_raw = [x for x in s.split(",")]
    core_segs, landmark, state, all_nums = [], [], "", set()
    house, street_seg = "", None
    for seg in segs_raw:
        toks = _clean_segment(seg)
        if not toks:
            continue
        st = _state_of(toks)
        if st:
            state = state or st
            continue
        # split off landmark phrase ("near club florence")
        for i, t in enumerate(toks):
            if t in LANDMARK_WORDS or (t == "next" and i + 1 < len(toks) and toks[i + 1] == "to"):
                landmark.extend(toks[i + 1:])
                toks = toks[:i]
                break
        kept = []
        for t in toks:
            if t in ADDR_DROP:
                continue
            if _NUMTOK.search(t):
                if _PIN6.match(t):
                    postcode = postcode or t
                    continue
                all_nums.add(t)
                if not house and not _ORDINAL.match(t):
                    house = t
                    street_seg = len(core_segs)
            kept.append(t)
        if kept:
            core_segs.append(kept)
    # street tokens: segment holding the house number, else first segment with a street word
    if street_seg is None:
        for i, seg in enumerate(core_segs):
            if any(t in STREET_TYPES for t in seg):
                street_seg = i
                break
    street = ""
    if street_seg is not None:
        street = " ".join(t for t in core_segs[street_seg]
                          if t != house and t not in STREET_TYPES and not _NUMTOK.search(t))
    city = ""
    for i in range(len(core_segs) - 1, -1, -1):
        if i == street_seg:
            continue
        cand = [t for t in core_segs[i] if not _NUMTOK.search(t)]
        if cand:
            city = " ".join(cand)
            break
    core = " ".join(" ".join(seg) for seg in core_segs)
    return {
        "addr_core": core,
        "addr_segs": "|".join(" ".join(seg) for seg in core_segs),
        "addr_landmark": " ".join(landmark),
        "house_num": house,
        "all_nums": " ".join(sorted(all_nums)),
        "postcode": postcode,
        "street": street,
        "city": city,
        "state": state,
        "addr_empty": int(not core_segs),
    }


# =====================================================================================
# Batch / parallel driver
# =====================================================================================
NAME_COLS = ["name_clean", "name_core", "name_sorted", "name_nospace", "legal_form",
             "is_domain", "name_acronym", "name_phon", "name_first", "name_skel"]
ADDR_COLS = ["addr_core", "addr_segs", "addr_landmark", "house_num", "all_nums",
             "postcode", "street", "city", "state", "addr_empty"]
INT_COLS = {"is_domain", "addr_empty"}


def normalize_batch(args) -> pl.DataFrame:
    """Normalise a batch of (idx, names, addresses) and return a polars frame."""
    idx, names, addrs = args
    cols = {k: [] for k in NAME_COLS + ADDR_COLS}
    for n, a in zip(names, addrs):
        for k, v in normalize_name(n).items():
            cols[k].append(v)
        for k, v in normalize_address(a).items():
            cols[k].append(v)
    df = pl.DataFrame({"idx": idx, **cols})
    return df.with_columns([pl.col(c).cast(pl.Int8) for c in INT_COLS] +
                           [pl.col("idx").cast(pl.Int32)])


def norm_path(split: str):
    """Parquet path of the normalised table for a split."""
    return C.CACHE_DIR / f"{split}_norm.parquet"


def run(split: str, batch: int = 100_000) -> None:
    """Normalise every record of a split in parallel and write it to parquet."""
    import io_utils as io
    rec = io.load_records(split, ["idx", "name", "address"])
    n = rec.height
    jobs = ((rec["idx"][i:i + batch].to_list(), rec["name"][i:i + batch].to_list(),
             rec["address"][i:i + batch].to_list()) for i in range(0, n, batch))
    parts = []
    with ProcessPoolExecutor(C.N_JOBS) as ex:
        for k, df in enumerate(ex.map(normalize_batch, jobs)):
            parts.append(df)
            if k % 20 == 0:
                print(f"  {min((k + 1) * batch, n):,}/{n:,}", flush=True)
    out = pl.concat(parts)
    out.write_parquet(norm_path(split))
    print(out.head(5))


def load_norm(split: str, columns=None) -> pl.DataFrame:
    """Load the cached normalised table for a split."""
    return pl.read_parquet(norm_path(split), columns=columns)


if __name__ == "__main__":
    import io_utils as io
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=C.SPLITS, nargs="+", default=list(C.SPLITS))
    a = ap.parse_args()
    for sp in a.split:
        with io.stage(f"normalize {sp}"):
            run(sp)
