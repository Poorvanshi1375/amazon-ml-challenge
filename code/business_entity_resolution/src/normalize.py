"""Stage 1: text normalisation.

Produces, for every record:
  name_clean   - ascii, lowercase, punctuation-free name (all words kept)
  name_core    - name_clean without legal forms / honorifics
  name_alt     - the alias part after a marker such as "f/k/a", "dba" ("" if none)
  name_legal   - canonical legal-form tokens found in the name (space separated, sorted)
  addr_clean   - ascii, lowercase, punctuation-free address without "null"/URLs
  addr_comps   - the address split on commas, each part cleaned, joined with "|"
  house_no     - first token containing a digit ("" if none)
  postcode     - last standalone 4-6 digit token ("" if none) - country independent
  num_tokens   - all digit-bearing tokens, space separated

Only general language knowledge is hand-written here (legal-form words and
honorifics, including a French seed list). Street/state abbreviations are
learned from the training data (`mine_rewrites`), and so is a word dictionary
for names written in Indian scripts (`mine_word_dict`): matched training pairs
such as "सदर्न प्रॉपर्टीज प्राइवेट लिमिटेड" / "Southern Properties Private Limited"
teach "सदर्न" -> "southern" word by word, which character transliteration
("sdrn") cannot recover.
"""
import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz.distance import Levenshtein

# --- hand-written general language knowledge (documented in the methodology) ---
LEGAL_CANON = {
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "limited": "ltd", "ltd": "ltd", "private": "pvt",
    "pvt": "pvt", "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc",
    "pllc": "pllc", "pa": "pa", "na": "na",
    # French legal forms (seed list: no French training data exists)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "scp": "scp", "selarl": "selarl",
}
HONORIFICS = {"the", "shri", "shree", "sri", "m s", "ms", "messrs"}
ALIAS_RE = re.compile(r"\b(?:f/k/a|fka|d/b/a|dba|a/k/a|aka|formerly|trading as|t/a)\b", re.I)
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\S+\.(?:com|net|org|in|fr|co|io|biz|info)\b\S*)", re.I)
NULL_RE = re.compile(r"\b(?:null|none|nan|n/a)\b", re.I)
NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
DIGIT_RE = re.compile(r"\d")
TOKEN_STRIP = ".,;:()[]{}\"'`!?-_/\\|"
WORD_DICT = {}  # learned native-script word -> English word (set by prepare.py before normalising)


def set_word_dict(d):
    global WORD_DICT
    WORD_DICT = dict(d)


def translate_words(raw):
    """Replace non-ascii words found in the learned dictionary by their English word."""
    if raw.isascii() or not WORD_DICT:
        return raw
    out = []
    for t in raw.split():
        k = t.strip(TOKEN_STRIP).lower()
        out.append(WORD_DICT.get(k, t) if not k.isascii() else t)
    return " ".join(out)


def mine_word_dict(native_names, english_names, min_count=3, min_ratio=0.6):
    """Learn native-word -> English-word pairs from matched training names.

    Only pairs whose first name contains non-ascii words and has the same number
    of words as the second are used; words are aligned by position. A pair is kept
    when it is seen at least min_count times and is the dominant translation.
    """
    counts, totals = {}, {}
    for a, b in zip(native_names, english_names):
        if a.isascii():
            continue
        ta = [t.strip(TOKEN_STRIP).lower() for t in a.split()]
        tb = [t.strip(TOKEN_STRIP).lower() for t in b.split()]
        ta, tb = [t for t in ta if t], [t for t in tb if t]
        if len(ta) != len(tb) or not ta:
            continue
        for x, y in zip(ta, tb):
            if x.isascii() or not y.isascii() or not y.isalpha():
                continue
            counts[(x, y)] = counts.get((x, y), 0) + 1
            totals[x] = totals.get(x, 0) + 1
    d = {}
    for (x, y), c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if x not in d and c >= min_count and c / totals[x] >= min_ratio:
            d[x] = y
    return d
POSTCODE_RE = re.compile(r"^\d{4,6}$")


def to_ascii(s):
    """Transliterate any script to ascii (Devanagari, Telugu, accents...) and lowercase."""
    if not s.isascii():
        s = anyascii(unicodedata.normalize("NFKC", s))
    return s.lower()


def clean(s):
    """ascii + lowercase + '&'->'and' + punctuation to spaces + collapse spaces."""
    s = to_ascii(s).replace("&", " and ").replace("+", " ")
    s = NON_ALNUM_RE.sub(" ", s)
    return join_single_letters(s.split())


def join_single_letters(tokens):
    """'l l c' -> 'llc', 'p c' -> 'pc' (dotted abbreviations after punctuation removal)."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return " ".join(out)


def split_alias(raw):
    """'Orbilyra+ f/k/a Summit Council Inc' -> ('Orbilyra+ ', ' Summit Council Inc')."""
    parts = ALIAS_RE.split(raw, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (raw, "")


def name_fields(raw):
    raw = translate_words(raw)
    main, alt = split_alias(raw)
    name_clean = clean(raw if not alt else main + " " + alt)
    toks = name_clean.split()
    legal = sorted({LEGAL_CANON[t] for t in toks if t in LEGAL_CANON})
    core = [t for t in toks if t not in LEGAL_CANON and t not in HONORIFICS]
    alt_clean = clean(alt) if alt else ""
    alt_core = " ".join(t for t in alt_clean.split() if t not in LEGAL_CANON and t not in HONORIFICS)
    return name_clean, " ".join(core), alt_core, " ".join(legal)


def addr_fields(raw):
    raw = NULL_RE.sub(" ", URL_RE.sub(" ", raw))
    comps = [clean(c) for c in raw.split(",")]
    comps = [c for c in comps if c]
    addr_clean = " ".join(comps)
    nums, posts = [], []
    for ci, c in enumerate(comps):
        ct = c.split()
        for ti, t in enumerate(ct):
            if not DIGIT_RE.search(t):
                continue
            # a 5-6 digit number that does not start its address part (or is a
            # lone last part) is a postcode; house numbers lead their part
            if t.isdigit() and 5 <= len(t) <= 6 and (ti > 0 or (len(ct) == 1 and ci == len(comps) - 1 and ci > 0)):
                posts.append(t)
            else:
                nums.append(t)
    house = nums[0] if nums else ""
    postcode = posts[-1] if posts else ""
    return addr_clean, "|".join(comps), house, postcode, " ".join(nums + posts)


# --- France: no training labels exist, so French address conventions are written by hand
# (general language knowledge, documented in the methodology) ---
FR_ADMIN = {  # regions and departments: records alternate between the two, the city already locates them
    "hauts de france", "nouvelle aquitaine", "pays de la loire", "ile de france", "bretagne", "normandie",
    "grand est", "occitanie", "bourgogne franche comte", "auvergne rhone alpes", "centre val de loire",
    "provence alpes cote d azur", "corse", "nord pas de calais", "picardie", "aquitaine", "nord", "gironde",
    "loire atlantique", "pas de calais", "somme", "aisne", "oise", "vendee", "maine et loire", "sarthe",
    "mayenne", "landes", "dordogne", "lot et garonne", "pyrenees atlantiques", "charente", "charente maritime",
    "france"}
FR_TYPES = ["rue", "avenue", "boulevard", "allee", "impasse", "route", "chemin", "place", "cours", "quai",
            "cite", "square", "passage", "residence", "chaussee", "lotissement", "mail", "digue", "esplanade",
            "ruelle", "sente", "parvis", "venelle", "domaine", "promenade", "sentier", "parc", "clos", "hameau",
            "terrasse", "corniche", "carriere", "rangee", "quartier", "faubourg", "rond point"]
FR_ABBR = {"r": "rue", "av": "avenue", "ave": "avenue", "avn": "avenue", "bd": "boulevard", "blvd": "boulevard",
           "bld": "boulevard", "boul": "boulevard", "bvd": "boulevard", "all": "allee", "al": "allee",
           "allees": "allee", "imp": "impasse", "rte": "route", "ch": "chemin", "chem": "chemin", "pl": "place",
           "crs": "cours", "cour": "cours", "q": "quai", "qu": "quai", "qua": "quai", "sq": "square",
           "pass": "passage", "psg": "passage", "pas": "passage", "res": "residence", "resid": "residence",
           "chau": "chaussee", "lot": "lotissement", "espl": "esplanade", "parv": "parvis", "prom": "promenade",
           "rpt": "rond point", "terr": "terrasse", "cit": "cite", "fg": "faubourg", "fbg": "faubourg",
           "sen": "sente", "dom": "domaine", "ham": "hameau", "st": "saint", "ste": "sainte"}
FR_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "au", "aux", "et", "en"}
FR_NUM_PREFIX_RE = re.compile(r"\b(?:no|ndeg|n|num|numero)\s*(?=\d)")
FR_NUM_SUFFIX_RE = re.compile(r"\b(\d+)\s*(?:bis|ter|quater|[a-d])\b")
FR_DROP_RE = re.compile(r"\b(?:bp|cs|tsa|cedex|appt|appartement|apt|app|appart|etage|bat|batiment)"
                        r"\s*[0-9a-z]?\d*\b")
FR_GLUED_RE = re.compile(r"^([a-z]{3,})\d$")  # "chem1", "cour2", "espl1": a street type with a stray digit
FR_LEGAL = {"ei": "ei", "eirl": "eirl", "sca": "sca", "gie": "gie", "scop": "scop", "selas": "selas",
            "scs": "scs", "ets": "ets", "etablissement": "ets", "etablissements": "ets", "societe": "ste"}


def _fr_type(t):
    """Canonical French street type for a word in street position (abbreviation or 1-2 letter typo)."""
    if t in FR_ABBR:
        return FR_ABBR[t]
    if t in FR_TYPES or len(t) < 5:
        return t
    best = min(FR_TYPES, key=lambda c: Levenshtein.distance(t, c))
    d = Levenshtein.distance(t, best)
    return best if d <= 1 or (d == 2 and len(t) >= 6 and best[0] == t[0]) else t


def fr_addr_fields(raw):
    """French address: drop region/department parts, postal-box and apartment tails, number prefixes
    ("N°", "No") and suffixes (bis/ter), leading zeros; canonical street type; no articles."""
    raw = NULL_RE.sub(" ", URL_RE.sub(" ", raw))
    comps = []
    for c in raw.split(","):
        c = NON_ALNUM_RE.sub(" ", to_ascii(c).replace("&", " and ")).strip()
        if not c or " ".join(c.split()) in FR_ADMIN:
            continue
        c = FR_DROP_RE.sub(" ", FR_NUM_PREFIX_RE.sub(" ", c))
        c = re.sub(r"\b0+(\d)", r"\1", FR_NUM_SUFFIX_RE.sub(r"\1", c))
        toks = []
        for t in c.split():
            m = FR_GLUED_RE.match(t)
            toks.append(m.group(1) if m and _fr_type(m.group(1)) != m.group(1) else t)
        # street type: the word after the leading number, or the first word of the part
        i = 1 if toks and toks[0].isdigit() else 0
        if i < len(toks) and not toks[i].isdigit():
            toks[i] = _fr_type(toks[i])
        toks = [FR_ABBR.get(t, t) if t in ("st", "ste") else t for t in toks]
        toks = " ".join(toks).split()
        c = " ".join(t for t in toks if t not in FR_STOP)
        if c:
            comps.append(c)
    addr_clean = " ".join(comps)
    nums = [t for c in comps for t in c.split() if DIGIT_RE.search(t)]
    posts = [t for t in nums if t.isdigit() and len(t) == 5]
    nums = [t for t in nums if t not in posts]
    return addr_clean, "|".join(comps), nums[0] if nums else "", posts[-1] if posts else "", " ".join(nums + posts)


def fr_name_fields(raw):
    """French name: generic cleaning, plus French legal forms / "Etablissements" and the "(France)" tag
    treated as legal noise."""
    name_clean, core, alt, legal = name_fields(raw)
    toks = core.split()
    legal = sorted(set(legal.split()) | {FR_LEGAL[t] for t in toks if t in FR_LEGAL})
    core = " ".join(t for t in toks if t not in FR_LEGAL and t != "france") or core
    return name_clean, core, alt, " ".join(legal)


def _normalize_chunk(df):
    fr = (df.country.values == "France") if "country" in df else np.zeros(len(df), bool)
    n = [fr_name_fields(x) if f else name_fields(x) for x, f in zip(df.business_name.values, fr)]
    a = [fr_addr_fields(x) if f else addr_fields(x) for x, f in zip(df.business_address.values, fr)]
    out = pd.DataFrame(n, columns=["name_clean", "name_core", "name_alt", "name_legal"], index=df.index)
    out[["addr_clean", "addr_comps", "house_no", "postcode", "num_tokens"]] = pd.DataFrame(a, index=df.index)
    return out


def normalize_df(df, n_jobs=1, chunk=200_000):
    """Add the normalised columns to a source DataFrame (runs in parallel)."""
    chunks = [df.iloc[i:i + chunk] for i in range(0, len(df), chunk)]
    if n_jobs > 1 and len(chunks) > 1:
        with Pool(n_jobs) as p:
            parts = p.map(_normalize_chunk, chunks)
    else:
        parts = [_normalize_chunk(c) for c in chunks]
    return pd.concat([df, pd.concat(parts)], axis=1)


# --- learned abbreviation map (mined from matched training pairs) ---

def _is_abbrev(short, long_):
    """True if `short` could abbreviate `long_`: same first letter, ordered subsequence."""
    if len(short) >= len(long_) or short[0] != long_[0]:
        return False
    it = iter(long_)
    return all(c in it for c in short)


def _count_rewrites(pairs, min_count, min_ratio):
    counts, totals = {}, {}
    for x, y in pairs:
        s_, l_ = (x, y) if len(x) < len(y) else (y, x)
        if _is_abbrev(s_, l_):
            counts[(s_, l_)] = counts.get((s_, l_), 0) + 1
            totals[s_] = totals.get(s_, 0) + 1
    rewrites = {}
    for (s_, l_), c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if c >= min_count and s_ not in rewrites and c / totals[s_] >= min_ratio:
            rewrites[s_] = l_
    return rewrites


def mine_rewrites(left_comps, right_comps, min_count=30, min_ratio=0.5):
    """Learn abbreviation rewrites from matched training pairs (address side).

    Inputs are the `addr_comps` strings ("part|part|...") of the two records of
    each matched pair. Two maps are learned:
      comp_map  - whole address part -> canonical part ("oh"->"ohio",
                  "up"->"uttar pradesh", "mharastr"->"maharashtra"); applied only
                  when an entire part matches, so short codes never corrupt words
                  like French "de" or English "in".
      token_map - word -> word inside parts ("st"->"street", "rd"->"road"),
                  learned only from words that are not a whole part by themselves.
    A rewrite is kept when it is frequent, one side abbreviates the other (same
    first letter, ordered subsequence) and it is the dominant expansion.
    """
    comp_pairs, tok_pairs = [], []
    for a, b in zip(left_comps, right_comps):
        ca, cb = set(filter(None, a.split("|"))), set(filter(None, b.split("|")))
        ra, rb = ca - cb, cb - ca
        if len(ra) == 1 and len(rb) == 1:
            x, y = next(iter(ra)), next(iter(rb))
            if not DIGIT_RE.search(x + y):
                comp_pairs.append((x, y))
        whole = ca | cb
        ta = {t for c in ca for t in c.split()} - whole
        tb = {t for c in cb for t in c.split()} - whole
        xa, xb = ta - tb, tb - ta
        if len(xa) == 1 and len(xb) == 1:
            x, y = next(iter(xa)), next(iter(xb))
            if not DIGIT_RE.search(x + y):
                tok_pairs.append((x, y))
    return _count_rewrites(comp_pairs, min_count, min_ratio), _count_rewrites(tok_pairs, min_count, min_ratio)


def apply_rewrites(comps_series, comp_map, token_map):
    """Rewrite addr_comps strings; returns (new addr_comps, new addr_clean) lists."""
    new_comps, new_clean = [], []
    for s in comps_series:
        parts = []
        for c in s.split("|"):
            if not c:
                continue
            c = comp_map.get(c, c)
            parts.append(" ".join(token_map.get(t, t) for t in c.split()))
        new_comps.append("|".join(parts))
        new_clean.append(" ".join(parts))
    return new_comps, new_clean
