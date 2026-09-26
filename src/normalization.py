"""
Normalization Engine for Business Entity Resolution.
Preserves raw values and provides multiple canonical views:
- normalized (case, whitespace, punctuation, unicode)
- without legal suffixes (Pvt Ltd, LLC, Corp, SARL, SAS)
- compact (no whitespace, character-level comparison)
- token sets (with numbers and distinctive words preserved)
"""
import re
import unicodedata

LEGAL_TERMS = {
    'limited', 'ltd', 'private', 'pvt', 'llc', 'inc', 'corp', 'corporation',
    'llp', 'co', 'company', 'services', 'enterprises', 'holding', 'holdings',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'cie', 'lnc', 'd b a', 'dba'
}

COMMON_ADDR_TERMS = {
    'road', 'street', 'drive', 'avenue', 'lane', 'floor', 'city', 'unit',
    'block', 'near', 'behind', 'beside', 'opposite', 'cross', 'main', 'rd',
    'st', 'ave', 'dr', 'pl', 'blvd', 'rue', 'de', 'du', 'la', 'des', 'north',
    'south', 'east', 'west', 'india', 'state', 'district', 'nagar', 'colony'
}

def normalize_text(text):
    if not text:
        return ""
    # 1. Unicode NFKD normalization & ASCII folding
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8')
    # 2. Lowercase
    text = text.lower()
    # 3. Strip URLs / domains
    text = re.sub(r'https?://\S+|www\.\S+|\.(com|in|org|net|fr|co)\b', '', text)
    # 4. Standardize business prefixes
    text = re.sub(r'\bm/s\b|\bm\s+s\b|\bd/b/a\b|\bd\s+b\s+a\b|\bdba\b', ' ', text)
    # 5. Punctuation to whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 6. Collapse repeated whitespace
    return " ".join(text.split())

def normalize_business_name(raw_name):
    norm = normalize_text(raw_name)
    tokens = norm.split()
    no_suffix_tokens = [w for w in tokens if w not in LEGAL_TERMS and len(w) >= 3]
    norm_nosuffix = " ".join(no_suffix_tokens) if no_suffix_tokens else norm
    compact = re.sub(r'\s+', '', norm_nosuffix)
    return {
        'name_raw': raw_name or "",
        'name_norm': norm,
        'name_norm_nosuffix': norm_nosuffix,
        'name_tokens': tokens,
        'name_compact': compact
    }

def normalize_business_address(raw_addr):
    norm = normalize_text(raw_addr)
    tokens = norm.split()
    addr_tokens = [w for w in tokens if w not in COMMON_ADDR_TERMS and (len(w) >= 4 or any(c.isdigit() for c in w))]
    compact = re.sub(r'\s+', '', norm)
    numbers = set(w for w in tokens if any(c.isdigit() for c in w))
    return {
        'address_raw': raw_addr or "",
        'address_norm': norm,
        'address_tokens': addr_tokens,
        'address_compact': compact,
        'numbers': numbers
    }

def normalize_country(raw_country):
    return (raw_country or "").strip()

def normalize_record(entity_id, name, addr, country):
    n_views = normalize_business_name(name)
    a_views = normalize_business_address(addr)
    c_norm = normalize_country(country)
    return {
        'entity_id': entity_id,
        'country': c_norm,
        **n_views,
        **a_views
    }
