"""
Afaan Oromo Stemmer & Information Retrieval System
Based on: Debela Tesfaye & Ermias Abebe (2010) "Designing a Rule Based Stemmer for Afaan Oromo Text"
Dictionary: Tilahun Gamta (2004) Comprehensive Oromo-English Dictionary (COED)
"""

import json, re, math, os
from collections import defaultdict
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ─────────────────────────────────────────────
# STEMMER  (rule-based, context-sensitive)
# Follows the paper's architecture exactly
# ─────────────────────────────────────────────

VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz`')

STOP_WORDS = {
    'kana','sun','ani','inni','isaan','ise','akka','ana','fi','itti','irra',
    'irraa','irrati','irratti','keessa','keessaa','gara','jalaa','jala',
    'waliin','wajjin','malee','garuu','immoo','ammoo','yoo','yookaan',
    'ykn','hin','ni','isa','kan','kee','koo','isaa','isee','isaanii',
    'nuyi','nuti','isin','ati','inni','iseen','kun','suni','waan','kan',
    'maal','maali','eessa','eenyuu','akkam','yoom','meeqa','hagam','bara',
    'bira','dura','booda','erga','haga','hamma','osoo','yeroo','yeroo',
    'amma','har\'aa','ati','inni'
}

def count_vc_sequences(stem):
    """Count vowel-consonant sequences (measure) in the stem."""
    if not stem:
        return 0
    groups = []
    prev_type = None
    for ch in stem.lower():
        if ch in VOWELS:
            cur_type = 'V'
        elif ch in CONSONANTS:
            cur_type = 'C'
        else:
            continue
        if cur_type != prev_type:
            groups.append(cur_type)
            prev_type = cur_type
    # Count VC pairs
    count = 0
    for i in range(len(groups) - 1):
        if groups[i] == 'V' and groups[i+1] == 'C':
            count += 1
    return count

def ends_with_vowel(stem):
    s = stem.rstrip()
    return s and s[-1] in VOWELS

def ends_with_consonant(stem):
    s = stem.rstrip()
    return s and s[-1] in CONSONANTS

def ends_with_bgd(stem):
    s = stem.rstrip()
    return s and s[-1] in {'b','g','d'}

def ends_with_double_vowel(stem):
    s = stem.rstrip()
    return len(s) >= 2 and s[-1] in VOWELS and s[-2] in VOWELS

def ends_with_double_consonant(stem):
    s = stem.rstrip()
    return len(s) >= 2 and s[-1] in CONSONANTS and s[-2] in CONSONANTS

def strip_suffix(word, suffix):
    if word.endswith(suffix):
        return word[:-len(suffix)]
    return None

def apply_rule(word, suffix, substitution, measure_condition, extra_check=None):
    """Try to apply a stemming rule. Returns new stem or None."""
    stem_candidate = strip_suffix(word, suffix)
    if stem_candidate is None:
        return None
    measure = count_vc_sequences(stem_candidate)
    # Evaluate measure condition
    try:
        if not eval(f"{measure} {measure_condition}"):
            return None
    except:
        return None
    # Extra context-sensitive condition
    if extra_check and not extra_check(stem_candidate):
        return None
    # Apply substitution
    if substitution == '0':
        return stem_candidate
    elif substitution == '`':
        return stem_candidate + '`'
    elif substitution == 'd':
        return stem_candidate + 'd'
    else:
        return stem_candidate + substitution

# ── CLUSTER 1: Attached suffixes (case/postposition markers) ──
CLUSTER1_RULES = [
    # (suffix, substitution, measure_cond, extra_check)
    ('rrati',  '0', '>= 1', None),
    ('rratti', '0', '>= 1', None),
    ('rraa',   '0', '>= 1', None),
    ('irraa',  '0', '>= 1', None),
    ('irra',   '0', '>= 1', None),
    ('rratti', '0', '>= 1', None),
    ('keessa', '0', '>= 1', None),
    ('keessaa','0', '>= 1', None),
    ('tiif',   '0', '>= 1', None),
    ('dhaaf',  '0', '>= 1', None),
    ('dhan',   '0', '>= 1', None),
    ('dhaan',  '0', '>= 1', None),
    ('itti',   '0', '>= 1', None),
    ('tti',    '0', '>= 1', ends_with_vowel),
    ('tii',    '0', '>= 1', ends_with_consonant),
    ('niin',   '0', '>= 1', None),
    ('nii',    '0', '>= 1', None),
    ('nis',    '0', '>= 1', None),
    ('leen',   '0', '>= 1', None),
    ('llee',   '0', '>= 1', None),
]

# ── CLUSTER 2: Inflectional suffixes – number/gender ──
CLUSTER2_RULES = [
    ('oota',   '0', '>= 1', None),
    ('wwan',   '0', '>= 1', ends_with_double_vowel),
    ('ootaa',  '0', '>= 1', None),
    ('ootti',  '0', '>= 1', None),
    ('eeyyu',  '0', '>= 1', None),
    ('achuu',  '0', '>= 1', None),
    ('atuu',   '0', '>= 1', None),
    ('ittii',  '0', '>= 1', None),
    ('iidhaa', '0', '>= 1', None),
    ('eedha',  '0', '>= 1', None),
    ('aan',    '0', '>= 1', ends_with_vowel),
    ('een',    '0', '>= 1', ends_with_vowel),
    ('oon',    '0', '>= 1', ends_with_vowel),
    ('lee',    '0', '>= 1', ends_with_consonant),
    ('ooli',   '0', '>= 1', None),
]

# ── CLUSTER 3: Tense/voice/transitivity ──
CLUSTER3_RULES = [
    ('siisuu',  '0', '>= 1', None),
    ('siifachuu','0','>= 1', None),
    ('achuu',   '0', '>= 1', None),
    ('atuu',    '0', '>= 1', None),
    ('ifachuu', '0', '>= 1', None),
    ('ifamuu',  '0', '>= 1', None),
    ('amuu',    '0', '>= 1', None),
    ('amsuu',   '0', '>= 1', None),
    ('omuu',    '0', '>= 1', None),
    ('aawuu',   '0', '>= 1', None),
    ('oomuu',   '0', '>= 1', None),
    ('isuu',    '0', '>= 1', None),
    ('iisuu',   '0', '>= 1', None),
    ('uu',      '0', '>= 2', ends_with_consonant),
]

# ── CLUSTER 4: Special plural/case ──
CLUSTER4_RULES = [
    ('du',   '0', '>= 1', ends_with_bgd),
    ('du',   'd', '== 0', ends_with_bgd),
    ('di',   '0', '>= 1', ends_with_bgd),
    ('di',   'd', '== 0', ends_with_bgd),
    ('dan',  '0', '>= 1', ends_with_bgd),
    ('dan',  'd', '== 0', ends_with_bgd),
    ('wwan', '0', '>= 1', ends_with_double_vowel),
]

# ── CLUSTER 5: Derivational ──
CLUSTER5_RULES = [
    ('`aa',  '0', '>= 1', None),
    ("'e",   '0', '>= 1', None),
    ("'u",   '0', '>= 1', None),
    ("'ee",  '0', '>= 1', None),
    ('suu',  '0', '>= 1', None),
    ('sa',   '0', '>= 1', None),
    ('sse',  '0', '>= 1', None),
    ('nya',  '`', '>= 0', None),
    ("'aa",  '0', '>= 1', None),
    ('chuu', '0', '>= 1', None),
    ('muu',  '0', '>= 1', None),
    ('tuu',  '0', '>= 1', None),
    ('nuu',  '0', '>= 1', None),
]

# ── CLUSTER 6: Derivational – measure>=1, ends consonant ──
CLUSTER6_RULES = [
    ('eenya',  '0', '>= 1', ends_with_consonant),
    ('offaa',  '0', '>= 1', ends_with_consonant),
    ('annoo',  '0', '>= 1', ends_with_consonant),
    ('ummaa',  '0', '>= 1', ends_with_consonant),
    ('inni',   '0', '>= 1', ends_with_consonant),
    ('achuu',  '0', '>= 1', ends_with_consonant),
    ('ummii',  '0', '>= 1', ends_with_consonant),
    ('aatii',  '0', '>= 1', ends_with_consonant),
    ('iinsa',  '0', '>= 1', ends_with_consonant),
    ('ummaa',  '0', '>= 1', ends_with_consonant),
]

ALL_CLUSTERS = [CLUSTER1_RULES, CLUSTER2_RULES, CLUSTER3_RULES,
                CLUSTER4_RULES, CLUSTER6_RULES, CLUSTER5_RULES]

def apply_cluster_7_reduplication(word):
    """
    Handle words formed by reduplication of the first syllable (Oromo plural).
    E.g.: jabaa -> jajjabaa,  gabaabaa -> gaggabaabaa
    Algorithm from paper: if C1+R1==R2, delete R2 (i.e., remove the prefix dup).
    Practical: the reduplicated part is at the START, so we strip it.
    """
    if len(word) < 5:
        return word

    # Strategy: find the smallest prefix P (len 2-4) such that:
    # - word starts with P
    # - immediately after P, word continues with a string that starts with
    #   the last consonant of P doubled (jajj... = ja + jj...)
    #   OR with P itself repeated
    # Then the true stem starts at the end of the doubled-prefix chunk.

    for plen in range(2, 5):
        if len(word) < plen + 3:
            continue
        prefix = word[:plen]

        # Case A: simple repetition (prefixprefix...) e.g. gaga... from ga
        if word[plen:plen+plen] == prefix:
            return word[plen:]

        # Case B: CV prefix, then the last-C doubled (jajjabaa: ja + jj + abaa)
        # The stem is: prefix[:-1] (drop the trailing consonant of prefix) + word[plen+1:]
        # jajjabaa: prefix=ja, last_C=a? No, 'a' is vowel.
        # Actually ja ends in 'a' (vowel). Next = 'jj'. 
        # Let's look at what comes after prefix:
        after = word[plen:]
        last_char_prefix = prefix[-1]

        # Case B: prefix ends in vowel, after starts with repeated consonant
        if last_char_prefix in VOWELS and len(after) >= 2:
            if after[0] == after[1] and after[0] in CONSONANTS:
                # jajjabaa: prefix=ja, after=jjabaa -> stem = ja + babaa? No.
                # Paper: jajjabaa from jabaa. So stem = jabaa = prefix + after[1:]
                candidate = prefix + after[1:]  # ja + jabaa? = jajabaa... wrong
                # Actually: jabaa -> j(a)(j)(jabaa) -> jajjabaa
                # The duplication is: j (first C) + a (first V) = ja, then repeat the j -> jaj
                # So stem = word with first (plen+1) chars = 'jaj' removed -> 'abaa'
                # But that gives 'abaa' not 'jabaa'... 
                # Correct removal: remove 'jaj' (plen chars + 1 of the doubled C) 
                # -> word[plen+1:] = 'jabaa'... wait jajjabaa[3:] = jabaa YES!
                candidate2 = word[plen + 1:]  # skip prefix AND one of the doubled consonants
                if len(candidate2) >= 3:
                    return candidate2

        # Case C: prefix ends in consonant, after starts with that consonant (gagg: ga+gg)
        if last_char_prefix in CONSONANTS and len(after) >= 1 and after[0] == last_char_prefix:
            # gaggabaabaa: prefix=ga, last_C=a? No, 'a' is vowel again.
            pass

        # Case D: word starts C+V, next part is C+C+V+... (gaggabaabaa)
        # gaggabaabaa: g-a-g-g-a-b-a-a-b-a-a
        # Stem: gabaabaa. Prefix added: ga + gg (doubled g) -> gagg
        # So strip 'gag' (plen=2 'ga' + one 'g') = word[3:] = 'gabaabaa' YES
        if plen == 2 and word[0] in CONSONANTS and word[1] in VOWELS:
            C1, V1 = word[0], word[1]
            # Check if word[2] == C1 (start of doubled consonant)
            if len(word) > 4 and word[2] == C1 and word[3] == C1:
                # Double consonant -> gagg... -> remove gag -> gabaabaa
                return C1 + word[3:]  # put one C1 back + rest -> gabaabaa? 
                # word[3:] from gaggabaabaa = gabaabaa YES

    return word

def stem(word):
    """
    Main stemming function implementing the paper's iterative context-sensitive algorithm.
    Returns (stem, rules_applied) for transparency.
    """
    word = word.lower().strip()
    steps = []

    if not word or word in STOP_WORDS:
        return word, ['stop_word' if word in STOP_WORDS else 'empty']

    if len(word) <= 3:
        return word, ['too_short']

    current = word

    # Apply clusters in order (iterative)
    for cluster_idx, cluster in enumerate(ALL_CLUSTERS, 1):
        changed = True
        iterations = 0
        while changed and iterations < 3:
            changed = False
            for rule in cluster:
                suffix, substitution, measure_cond, extra_check = rule
                result = apply_rule(current, suffix, substitution, measure_cond, extra_check)
                if result and len(result) >= 2 and result != current:
                    steps.append(f"C{cluster_idx}: -{suffix}→'{result}'")
                    current = result
                    changed = True
                    break  # Restart cluster with new stem
            iterations += 1

    # Cluster 7: handle reduplication
    dedup = apply_cluster_7_reduplication(current)
    if dedup != current:
        steps.append(f"C7: dedup→'{dedup}'")
        current = dedup

    if not steps:
        steps = ['no_rule_applied']

    return current, steps


# ─────────────────────────────────────────────
# DOCUMENT CORPUS  (built-in Oromo sample texts)
# ─────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "title": "Seenaa Oromoo",
        "content": "Oromoon saboota Afriikaa keessaa tokko yoo ta'u, Itoophiyaa keessatti baay'inaan argamu. Afaan Oromoo afaanota Afrikaa keessaa afaan guddaa fi baay'ee dubbatamu keessaa tokko. Oromoon seenaa dheeraa fi aadaa bareedaa qabu. Sirni gadaa mootummaa dimokiraatawaa Oromoo yeroo dheeraa dura hundeeffame. Oromoon biyya isaanii irratti aangoo qabu."
    },
    {
        "id": 2,
        "title": "Barnootaa fi Baruumsa",
        "content": "Barnootni bu'ura guddinaa biyya kamiyyuu keessatti. Barattootni Afaan Oromootiin barachuu danda'u. Barsiisotni barumsa kennu barattootaaf. Manni baruumsa barattootaaf bakka barumsa mana. Baruumsa argachuun mirga barattootaa hunda. Oromoon baruumsaaf xiyyeeffannoo guddaa kennaa jiru."
    },
    {
        "id": 3,
        "title": "Qonnaa fi Dinagdee",
        "content": "Qonni bu'ura dinagdee Itoophiyaa. Qotiyyooleen lafa qotan. Beekaan qonnaan bulaa gargaaru. Omisha midhaanii dabaluu barbaachisa. Qonnaan bulaan biyya kana keessa hojjetu. Midhaaniin akka gaariitti guddatee galii argamsiisa."
    },
    {
        "id": 4,
        "title": "Fayyaa fi Dhukkuba",
        "content": "Fayyaan qabeenyaa namaa hundaa ol. Dhukkubni nama rakkisa. Yaalamuu fi fayyuun barbaachisaa dha. Dokterri dhukkubsataa yaaluu. Hospitaalli bakka namni yaalamuutti. Qorannoon dhukkuba adda baasuuf barbaachisaa. Namni fayyaa qabaachuuf nyaata gaarii nyaachuu qaba."
    },
    {
        "id": 5,
        "title": "Aadaa fi Duudhaa Oromoo",
        "content": "Aadaan Oromoo badhaadhaa fi daran bareedaa. Sirni Gadaa aadaa Oromoo keessatti bakka ol'aanaa qaba. Ateetee sirna aadaa dubartootaa Oromoo. Irreechaa ayyaana guddaa Oromoo. Kabajaan aadaa eeguu Oromoo biratti baay'ee barbaachisaa. Seenaan Oromoo haala adda ta'een dubbatama."
    },
    {
        "id": 6,
        "title": "Teknoolojii fi Ammayyeessuu",
        "content": "Teknoolojiin ammayyeessa addunyaa keessa waan hedduu jijjiire. Intarneetiin odeeffannoo argachuu salphaa godhe. Bilbilli harkaa namootaan gidduu walqunnamtii saffisaa godha. Itoophiyaanis teknoolojiin daran guddachaa jirti. Dargaggootni teknoolojii fayyadamuun baruumsa isaanii cimsuuf carraaqaa jiru."
    },
    {
        "id": 7,
        "title": "Haala Qilleensaa fi Naannoo",
        "content": "Naannoon nu marsu kabajamuu qaba. Bosoni addunyaa keessa xiqqaachaa jira. Roobni yeroon ro'u qonnaan bulaa gammachiisa. Bishaan dhugaatii namoota hedduuf rakkina. Haalli qilleensaa jijjiiramus midhagina naannoo irra daddaftee miidhaa geessisuu danda'a."
    },
    {
        "id": 8,
        "title": "Siyaasa fi Biyya Bulchuu",
        "content": "Mootummaan ummataa biyya bulcha. Miseensi paarlaamaa ummataan filamu. Mirgi ummataa kabajamuu qaba. Dimokiraasiin sirna biyya bulchuu ta'uu qaba. Filannoon hunda yoo gaggeeffamu mirga ummataati. Oromoon mirga ofiin of bulchuu qaba."
    },
]


# ─────────────────────────────────────────────
# LOAD DICTIONARY
# ─────────────────────────────────────────────

DICTIONARY = {}
dict_path = os.path.join(os.path.dirname(__file__), 'oromo_dictionary.json')
if os.path.exists(dict_path):
    with open(dict_path) as f:
        DICTIONARY = json.load(f)


# ─────────────────────────────────────────────
# TF-IDF RETRIEVAL ENGINE
# ─────────────────────────────────────────────

def tokenize(text):
    """Tokenize Oromo text."""
    # Oromo uses Latin script with special chars
    tokens = re.findall(r"[a-z'`]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def stem_tokens(tokens):
    return [stem(t)[0] for t in tokens]

class OromoIRSystem:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_lengths = {}
        self.idf = {}
        self.N = len(documents)
        self._build_index()

    def _build_index(self):
        doc_term_counts = {}
        for doc in self.documents:
            doc_id = doc['id']
            tokens = tokenize(doc['title'] + ' ' + doc['content'])
            stemmed = stem_tokens(tokens)
            counts = defaultdict(int)
            for t in stemmed:
                counts[t] += 1
            doc_term_counts[doc_id] = counts

        # Collect all terms
        all_terms = set()
        for counts in doc_term_counts.values():
            all_terms.update(counts.keys())

        # Build inverted index with TF-IDF
        for term in all_terms:
            df = sum(1 for counts in doc_term_counts.values() if term in counts)
            self.idf[term] = math.log((self.N + 1) / (df + 1)) + 1  # smooth IDF

        for doc in self.documents:
            doc_id = doc['id']
            counts = doc_term_counts[doc_id]
            total = sum(counts.values()) or 1
            vec_len = 0
            for term, cnt in counts.items():
                tf = 1 + math.log(cnt) if cnt > 0 else 0
                tfidf = tf * self.idf.get(term, 0)
                self.index[term][doc_id] = tfidf
                vec_len += tfidf ** 2
            self.doc_lengths[doc_id] = math.sqrt(vec_len) or 1

    def search(self, query, top_k=5):
        tokens = tokenize(query)
        stemmed = stem_tokens(tokens)

        if not stemmed:
            return []

        scores = defaultdict(float)
        query_vec = {}
        for term in stemmed:
            query_vec[term] = query_vec.get(term, 0) + 1

        q_norm = 0
        for term, cnt in query_vec.items():
            tf = 1 + math.log(cnt) if cnt > 0 else 0
            idf = self.idf.get(term, math.log((self.N + 1) / 1) + 1)
            q_tfidf = tf * idf
            query_vec[term] = q_tfidf
            q_norm += q_tfidf ** 2
            for doc_id, doc_tfidf in self.index.get(term, {}).items():
                scores[doc_id] += q_tfidf * doc_tfidf

        q_norm = math.sqrt(q_norm) or 1
        # Normalize
        for doc_id in scores:
            scores[doc_id] /= (q_norm * self.doc_lengths[doc_id])

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for doc_id, score in ranked:
            doc = next(d for d in self.documents if d['id'] == doc_id)
            results.append({
                'id': doc_id,
                'title': doc['title'],
                'snippet': doc['content'][:200] + '...',
                'score': round(score, 4),
                'matched_terms': stemmed
            })
        return results

ir_system = OromoIRSystem(SAMPLE_DOCUMENTS)


# ─────────────────────────────────────────────
# HTML TEMPLATE (modern GUI)
# ─────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="om">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Afaan Oromo NLP System</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3350;
    --accent: #5b8af5;
    --accent2: #7c5bf5;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
    --text: #e2e8f0;
    --muted: #8892a4;
    --radius: 12px;
    --font: 'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    min-height: 100vh;
  }

  /* Header */
  header {
    background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
  }
  .brand h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
  .brand p { font-size: 12px; color: var(--muted); }
  .badge {
    margin-left: auto;
    background: rgba(91,138,245,0.15);
    border: 1px solid rgba(91,138,245,0.3);
    color: var(--accent);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
  }

  /* Layout */
  .container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }

  /* Tabs */
  .tabs { display: flex; gap: 4px; margin-bottom: 28px; background: var(--surface); border-radius: var(--radius); padding: 4px; width: fit-content; }
  .tab {
    padding: 9px 20px;
    border-radius: 9px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: var(--muted);
    border: none;
    background: transparent;
    transition: all 0.2s;
  }
  .tab.active {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    box-shadow: 0 2px 12px rgba(91,138,245,0.4);
  }
  .tab:hover:not(.active) { color: var(--text); background: var(--surface2); }

  /* Panels */
  .panel { display: none; animation: fadeIn 0.3s ease; }
  .panel.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 16px;
  }

  /* Input / Controls */
  .input-row { display: flex; gap: 10px; margin-bottom: 16px; }
  input[type=text], textarea {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 9px;
    color: var(--text);
    font-size: 15px;
    font-family: var(--font);
    padding: 11px 16px;
    outline: none;
    transition: border-color 0.2s;
    width: 100%;
  }
  input[type=text]:focus, textarea:focus { border-color: var(--accent); }
  textarea { min-height: 100px; resize: vertical; }

  .btn {
    padding: 11px 22px;
    border-radius: 9px;
    border: none;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.2s;
    white-space: nowrap;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    box-shadow: 0 2px 12px rgba(91,138,245,0.35);
  }
  .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(91,138,245,0.5); }
  .btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

  /* Token chips */
  .token-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .token {
    padding: 6px 13px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .token-raw { background: rgba(250,204,21,0.12); border: 1px solid rgba(250,204,21,0.3); color: var(--yellow); }
  .token-stem { background: rgba(74,222,128,0.12); border: 1px solid rgba(74,222,128,0.3); color: var(--green); }
  .token-arrow { color: var(--muted); font-size: 11px; }

  /* Step trace */
  .step-trace {
    background: var(--surface2);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: var(--muted);
    margin-top: 10px;
  }
  .step-trace span { color: var(--accent); }

  /* Stats row */
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .stat {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    text-align: center;
  }
  .stat-num { font-size: 26px; font-weight: 700; color: var(--accent); }
  .stat-label { font-size: 12px; color: var(--muted); margin-top: 4px; }

  /* Search results */
  .result-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
  }
  .result-item:hover { border-color: var(--accent); }
  .result-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
  .result-title { font-size: 15px; font-weight: 600; }
  .result-score {
    background: rgba(91,138,245,0.15);
    color: var(--accent);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
  }
  .result-snippet { font-size: 13px; color: var(--muted); line-height: 1.6; }
  .result-terms { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
  .term-chip {
    background: rgba(74,222,128,0.1);
    color: var(--green);
    padding: 2px 9px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
  }

  /* Dictionary lookup */
  .dict-entry {
    background: var(--surface2);
    border-radius: var(--radius);
    padding: 20px;
    border-left: 3px solid var(--accent);
  }
  .dict-word { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
  .dict-pos {
    display: inline-block;
    background: rgba(124,91,245,0.15);
    color: var(--accent2);
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
  }
  .dict-def { font-size: 14px; color: var(--muted); line-height: 1.7; }

  /* Corpus panel */
  .doc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
  .doc-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .doc-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .doc-card-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; }
  .doc-card-preview { font-size: 12px; color: var(--muted); line-height: 1.5; }

  /* Add document */
  .add-form { display: flex; flex-direction: column; gap: 12px; }

  /* Alert */
  .alert { padding: 12px 16px; border-radius: 9px; font-size: 13px; margin-bottom: 12px; }
  .alert-info { background: rgba(91,138,245,0.1); border: 1px solid rgba(91,138,245,0.2); color: #93b4fc; }
  .alert-success { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.2); color: var(--green); }
  .alert-error { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.2); color: var(--red); }

  /* Loading */
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.2); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Responsive */
  @media (max-width: 600px) {
    .input-row { flex-direction: column; }
    .stats { grid-template-columns: repeat(2, 1fr); }
  }

  /* Highlight */
  mark { background: rgba(250,204,21,0.25); color: var(--yellow); border-radius: 3px; padding: 0 2px; }
  em { color: var(--accent); font-style: normal; }

  /* Empty state */
  .empty { text-align: center; padding: 48px 24px; color: var(--muted); }
  .empty-icon { font-size: 40px; margin-bottom: 12px; }

  hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
</style>
</head>
<body>

<header>
  <div class="logo">🌿</div>
  <div class="brand">
    <h1>Afaan Oromo NLP System</h1>
    <p>Rule-Based Stemmer · TF-IDF Retrieval · COED Dictionary</p>
  </div>
  <div class="badge">Qubee ✓</div>
</header>

<div class="container">
  <div class="tabs">
    <button class="tab active" onclick="switchTab('stemmer')">🔬 Stemmer</button>
    <button class="tab" onclick="switchTab('search')">🔍 Search</button>
    <button class="tab" onclick="switchTab('dictionary')">📖 Dictionary</button>
    <button class="tab" onclick="switchTab('corpus')">📚 Corpus</button>
    <button class="tab" onclick="switchTab('about')">ℹ️ About</button>
  </div>

  <!-- ── STEMMER PANEL ── -->
  <div id="panel-stemmer" class="panel active">
    <div class="card">
      <div class="card-title">Single Word Stemmer</div>
      <div class="input-row">
        <input type="text" id="stem-word" placeholder="Enter an Afaan Oromo word… e.g. barattootarratti" onkeydown="if(event.key==='Enter') stemWord()">
        <button class="btn btn-primary" onclick="stemWord()">Stem</button>
        <button class="btn btn-secondary" onclick="clearStemmer()">Clear</button>
      </div>
      <div id="stem-result"></div>
    </div>

    <div class="card">
      <div class="card-title">Batch Text Stemmer</div>
      <textarea id="batch-text" placeholder="Enter multiple Afaan Oromo sentences here…&#10;&#10;Example:&#10;Oromoon seenaa dheeraa qabu.&#10;Barattootni Afaan Oromootiin barachuu danda'u."></textarea>
      <div style="display:flex; gap:10px; margin-top:12px;">
        <button class="btn btn-primary" onclick="stemBatch()">Stem All Tokens</button>
        <button class="btn btn-secondary" onclick="loadSampleText()">Load Sample</button>
      </div>
      <div id="batch-result"></div>
    </div>
  </div>

  <!-- ── SEARCH PANEL ── -->
  <div id="panel-search" class="panel">
    <div class="card">
      <div class="card-title">Oromo Document Retrieval</div>
      <div class="input-row">
        <input type="text" id="search-query" placeholder="Search in Afaan Oromo… e.g. barumsa barnootaa" onkeydown="if(event.key==='Enter') runSearch()">
        <button class="btn btn-primary" onclick="runSearch()">Search</button>
      </div>
      <div class="alert alert-info">
        🔬 Queries are stemmed before matching — inflected forms will find related documents.
      </div>
      <div id="search-debug" style="display:none;" class="step-trace"></div>
    </div>
    <div id="search-results"></div>
  </div>

  <!-- ── DICTIONARY PANEL ── -->
  <div id="panel-dictionary" class="panel">
    <div class="card">
      <div class="card-title">COED Dictionary Lookup (Tilahun Gamta, 2004)</div>
      <div class="input-row">
        <input type="text" id="dict-word" placeholder="Look up an Afaan Oromo word…" onkeydown="if(event.key==='Enter') lookupWord()">
        <button class="btn btn-primary" onclick="lookupWord()">Look Up</button>
      </div>
      <p style="font-size:12px; color:var(--muted);">Dictionary contains <strong id="dict-size">loading…</strong> headwords. Try the stemmed form if the inflected form isn't found.</p>
    </div>
    <div id="dict-result"></div>

    <div class="card">
      <div class="card-title">Stem + Lookup (find base form first)</div>
      <div class="input-row">
        <input type="text" id="stem-lookup-word" placeholder="Enter inflected word to stem then look up…" onkeydown="if(event.key==='Enter') stemAndLookup()">
        <button class="btn btn-primary" onclick="stemAndLookup()">Stem & Look Up</button>
      </div>
      <div id="stem-lookup-result"></div>
    </div>
  </div>

  <!-- ── CORPUS PANEL ── -->
  <div id="panel-corpus" class="panel">
    <div class="stats" id="corpus-stats"></div>
    <div class="card">
      <div class="card-title">Document Collection</div>
      <div class="doc-grid" id="doc-grid"></div>
    </div>
    <div class="card">
      <div class="card-title">Add New Document</div>
      <div class="add-form">
        <input type="text" id="new-title" placeholder="Document title (Mata-duree)…">
        <textarea id="new-content" placeholder="Document content in Afaan Oromo (Qabiyyee)…"></textarea>
        <div>
          <button class="btn btn-primary" onclick="addDocument()">➕ Add to Corpus</button>
        </div>
      </div>
      <div id="add-result"></div>
    </div>
  </div>

  <!-- ── ABOUT PANEL ── -->
  <div id="panel-about" class="panel">
    <div class="card">
      <div class="card-title">About This System</div>
      <p style="line-height:1.8; font-size:14px; color:var(--muted);">
        This system implements a <strong style="color:var(--text)">rule-based, context-sensitive stemmer</strong> for Afaan Oromo,
        adapted from the algorithm described by <em>Debela Tesfaye & Ermias Abebe (2010)</em> in
        "Designing a Rule Based Stemmer for Afaan Oromo Text" (IJCL, Vol. 1, Issue 2).
      </p>
      <hr>
      <h3 style="font-size:14px; margin-bottom:10px;">Algorithm Architecture</h3>
      <p style="font-size:13px; color:var(--muted); line-height:1.7;">
        The stemmer uses 7 rule clusters applied iteratively:
      </p>
      <div style="margin-top: 12px; display:flex; flex-direction:column; gap:8px; font-size:13px;">
        <div class="step-trace">C1: Attached suffixes (case/postposition markers: -rra, -tti, -tiif…)</div>
        <div class="step-trace">C2: Inflectional – number/gender (-oota, -wwan, -een, -lee…)</div>
        <div class="step-trace">C3: Tense/voice/transitivity (-siisuu, -amuu, -achuu…)</div>
        <div class="step-trace">C4: Special plural/case (-du, -di, -dan for B/G/D-ending stems)</div>
        <div class="step-trace">C5: Derivational suffixes (-suu, -nya, -chuu…)</div>
        <div class="step-trace">C6: Derivational – requires consonant-ending stem (-eenya, -offaa, -annoo…)</div>
        <div class="step-trace">C7: Reduplication removal (handles plural-by-duplication: gaggabaabaa → gabaabaa)</div>
      </div>
      <hr>
      <h3 style="font-size:14px; margin-bottom:10px;">Dictionary Source</h3>
      <p style="font-size:13px; color:var(--muted); line-height:1.7;">
        Lexicon extracted from the <strong style="color:var(--text)">Comprehensive Oromo-English Dictionary (COED)</strong>
        by Tilahun Gamta / Xilaahun Gamtaa (2004, Karrayyuu Publishing, New York).
        Over 5,400 headwords with part-of-speech tags and English definitions.
      </p>
      <hr>
      <h3 style="font-size:14px; margin-bottom:10px;">Information Retrieval</h3>
      <p style="font-size:13px; color:var(--muted); line-height:1.7;">
        The search engine uses <strong style="color:var(--text)">TF-IDF with cosine similarity</strong>.
        Queries and documents are stemmed before indexing, enabling morphology-aware retrieval.
        For example, searching "barumsa" will match documents containing "barnoota", "barattootni", etc.
      </p>
    </div>
  </div>

</div>

<script>
// ── Tab switching ──
function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'corpus') loadCorpus();
  if (name === 'dictionary') loadDictSize();
}

// ── API helpers ──
async function api(endpoint, data) {
  const r = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  return r.json();
}

// ── STEMMER ──
async function stemWord() {
  const word = document.getElementById('stem-word').value.trim();
  if (!word) return;
  const el = document.getElementById('stem-result');
  el.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Stemming…</div>';
  const res = await api('/api/stem', {word});
  el.innerHTML = `
    <div style="margin-top:14px;">
      <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div class="token token-raw">${word}</div>
        <div style="color:var(--muted); font-size:18px;">→</div>
        <div class="token token-stem">${res.stem}</div>
        ${res.is_stop_word ? '<span style="color:var(--red); font-size:12px;">[stop word]</span>' : ''}
      </div>
      <div class="step-trace" style="margin-top:12px;">
        Steps: <span>${res.steps.join(' → ')}</span>
      </div>
      <div style="margin-top:12px; font-size:13px; color:var(--muted);">
        Measure (VC sequences in stem): <strong style="color:var(--text)">${res.measure}</strong>
        &nbsp;|&nbsp; Ends with: <strong style="color:var(--text)">${res.ends_with}</strong>
      </div>
    </div>`;
}

async function stemBatch() {
  const text = document.getElementById('batch-text').value.trim();
  if (!text) return;
  const el = document.getElementById('batch-result');
  el.innerHTML = '<div class="alert alert-info" style="margin-top:12px;"><span class="spinner"></span> Processing…</div>';
  const res = await api('/api/stem_batch', {text});
  if (!res.tokens || res.tokens.length === 0) {
    el.innerHTML = '<div class="alert alert-error" style="margin-top:12px;">No tokens found.</div>';
    return;
  }
  const chips = res.tokens.map(t =>
    `<div class="token ${t.is_stop ? 'token-raw' : 'token-stem'}" title="Steps: ${t.steps.join(' → ')}">
      ${t.original}${t.original !== t.stem && !t.is_stop ? ' <span class="token-arrow">→ '+t.stem+'</span>' : ''}
     </div>`
  ).join('');
  el.innerHTML = `
    <div style="margin-top:16px;">
      <div class="card-title">Results (${res.tokens.length} tokens, ${res.unique_stems} unique stems)</div>
      <div class="token-list">${chips}</div>
      <div class="step-trace" style="margin-top:12px;">
        Compression: ${res.tokens.length} words → ${res.unique_stems} unique stems
        (${Math.round((1 - res.unique_stems/res.tokens.length)*100)}% reduction)
      </div>
    </div>`;
}

function clearStemmer() {
  document.getElementById('stem-word').value = '';
  document.getElementById('stem-result').innerHTML = '';
}

function loadSampleText() {
  document.getElementById('batch-text').value = `Oromoon saboota Afriikaa keessaa tokko.
Barattootni Afaan Oromootiin barachuu danda'u.
Barsiisotni barumsa kennu barattootaaf.
Qonni bu'ura dinagdee Itoophiyaa.
Fayyaan qabeenyaa namaa hundaa ol.
Aadaan Oromoo badhaadhaa fi daran bareedaa.
Sirni gadaa mootummaa dimokiraatawaa Oromoo yeroo dheeraa dura hundeeffame.`;
}

// ── SEARCH ──
async function runSearch() {
  const query = document.getElementById('search-query').value.trim();
  if (!query) return;
  const el = document.getElementById('search-results');
  el.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Searching…</div>';
  const res = await api('/api/search', {query});

  // Show query debug
  const dbg = document.getElementById('search-debug');
  dbg.style.display = 'block';
  dbg.innerHTML = `Query tokens: <span>${res.query_tokens.join(', ')}</span> &nbsp;→&nbsp; Stems: <span>${res.query_stems.join(', ')}</span>`;

  if (!res.results || res.results.length === 0) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">🔍</div><div>No results found. Try different keywords.</div></div>';
    return;
  }
  el.innerHTML = `<div class="card-title" style="margin-bottom:8px;">${res.results.length} result${res.results.length>1?'s':''} for "<em>${query}</em>"</div>` +
    res.results.map((r, i) => `
      <div class="result-item">
        <div class="result-header">
          <div>
            <span style="color:var(--muted); font-size:12px; margin-right:8px;">#${i+1}</span>
            <span class="result-title">${r.title}</span>
          </div>
          <div class="result-score">Score: ${r.score}</div>
        </div>
        <div class="result-snippet">${r.snippet}</div>
        <div class="result-terms">
          ${r.matched_terms.map(t => `<span class="term-chip">${t}</span>`).join('')}
        </div>
      </div>`).join('');
}

// ── DICTIONARY ──
async function loadDictSize() {
  const res = await api('/api/dict_info', {});
  document.getElementById('dict-size').textContent = res.size.toLocaleString();
}

async function lookupWord() {
  const word = document.getElementById('dict-word').value.trim().toLowerCase();
  if (!word) return;
  const el = document.getElementById('dict-result');
  el.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Looking up…</div>';
  const res = await api('/api/lookup', {word});
  if (res.found) {
    el.innerHTML = `<div class="dict-entry">
      <div class="dict-word">${word}</div>
      <div class="dict-pos">${posLabel(res.pos)}</div>
      <div class="dict-def">${res.definition}</div>
    </div>`;
  } else {
    el.innerHTML = `<div class="alert alert-error">
      "<strong>${word}</strong>" not found in dictionary. 
      ${res.suggestions.length ? 'Did you mean: ' + res.suggestions.map(s=>`<strong>${s}</strong>`).join(', ') + '?' : 'Try the "Stem & Look Up" tool below.'}
    </div>`;
  }
}

async function stemAndLookup() {
  const word = document.getElementById('stem-lookup-word').value.trim();
  if (!word) return;
  const el = document.getElementById('stem-lookup-result');
  el.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Stemming then looking up…</div>';
  const res = await api('/api/stem_lookup', {word});
  let html = `<div class="step-trace" style="margin-bottom:12px;">
    "${word}" → stem: <span>${res.stem}</span> (steps: ${res.steps.join(' → ')})
  </div>`;
  if (res.found) {
    html += `<div class="dict-entry">
      <div class="dict-word">${res.stem} <span style="font-size:14px;color:var(--muted)">(from: ${word})</span></div>
      <div class="dict-pos">${posLabel(res.pos)}</div>
      <div class="dict-def">${res.definition}</div>
    </div>`;
  } else {
    html += `<div class="alert alert-error">Stem "<strong>${res.stem}</strong>" not found in dictionary either. The word may be a proper noun, compound, or very rare form.</div>`;
  }
  el.innerHTML = html;
}

function posLabel(pos) {
  const map = {n:'noun',tv:'transitive verb',iv:'intransitive verb',adj:'adjective',adv:'adverb',conj:'conjunction',prep:'preposition',pron:'pronoun',int:'interjection',imper:'imperative'};
  return map[pos] || pos;
}

// ── CORPUS ──
async function loadCorpus() {
  const res = await api('/api/corpus', {});
  const stats = document.getElementById('corpus-stats');
  stats.innerHTML = `
    <div class="stat"><div class="stat-num">${res.doc_count}</div><div class="stat-label">Documents</div></div>
    <div class="stat"><div class="stat-num">${res.total_tokens}</div><div class="stat-label">Tokens</div></div>
    <div class="stat"><div class="stat-num">${res.unique_stems}</div><div class="stat-label">Unique Stems</div></div>
    <div class="stat"><div class="stat-num">${res.vocab_size.toLocaleString()}</div><div class="stat-label">Dict Entries</div></div>`;

  const grid = document.getElementById('doc-grid');
  grid.innerHTML = res.documents.map(d => `
    <div class="doc-card" onclick="document.getElementById('search-query').value='${d.title.split(' ')[0]}'; switchTabDirect('search'); runSearch();">
      <div class="doc-card-title">${d.title}</div>
      <div class="doc-card-preview">${d.preview}</div>
    </div>`).join('');
}

function switchTabDirect(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.querySelectorAll('.tab').forEach(t => {
    if (t.getAttribute('onclick').includes(name)) t.classList.add('active');
  });
}

async function addDocument() {
  const title = document.getElementById('new-title').value.trim();
  const content = document.getElementById('new-content').value.trim();
  const el = document.getElementById('add-result');
  if (!title || !content) {
    el.innerHTML = '<div class="alert alert-error" style="margin-top:12px;">Please enter both title and content.</div>';
    return;
  }
  const res = await api('/api/add_document', {title, content});
  if (res.success) {
    el.innerHTML = `<div class="alert alert-success" style="margin-top:12px;">✓ Document "${title}" added! Corpus reindexed.</div>`;
    document.getElementById('new-title').value = '';
    document.getElementById('new-content').value = '';
    loadCorpus();
  }
}

// Init
loadDictSize();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stem', methods=['POST'])
def api_stem():
    word = request.json.get('word', '').strip().lower()
    s, steps = stem(word)
    m = count_vc_sequences(s)
    ew = s[-1] if s else ''
    return jsonify({
        'stem': s,
        'steps': steps,
        'measure': m,
        'ends_with': ew,
        'is_stop_word': word in STOP_WORDS
    })

@app.route('/api/stem_batch', methods=['POST'])
def api_stem_batch():
    text = request.json.get('text', '')
    tokens = re.findall(r"[a-z'`]+", text.lower())
    results = []
    all_stems = set()
    for t in tokens:
        s, steps = stem(t)
        is_stop = t in STOP_WORDS
        if not is_stop:
            all_stems.add(s)
        results.append({'original': t, 'stem': s, 'steps': steps, 'is_stop': is_stop})
    return jsonify({'tokens': results, 'unique_stems': len(all_stems)})

@app.route('/api/search', methods=['POST'])
def api_search():
    query = request.json.get('query', '')
    tokens = tokenize(query)
    stems = stem_tokens(tokens)
    results = ir_system.search(query)
    return jsonify({'results': results, 'query_tokens': tokens, 'query_stems': stems})

@app.route('/api/lookup', methods=['POST'])
def api_lookup():
    word = request.json.get('word', '').strip().lower()
    if word in DICTIONARY:
        e = DICTIONARY[word]
        return jsonify({'found': True, 'pos': e['pos'], 'definition': e['definition']})
    # Suggestions: words starting with same 3 chars
    prefix = word[:3] if len(word) >= 3 else word
    suggestions = [w for w in DICTIONARY if w.startswith(prefix) and w != word][:5]
    return jsonify({'found': False, 'suggestions': suggestions})

@app.route('/api/stem_lookup', methods=['POST'])
def api_stem_lookup():
    word = request.json.get('word', '').strip().lower()
    s, steps = stem(word)
    if s in DICTIONARY:
        e = DICTIONARY[s]
        return jsonify({'found': True, 'stem': s, 'steps': steps, 'pos': e['pos'], 'definition': e['definition']})
    return jsonify({'found': False, 'stem': s, 'steps': steps})

@app.route('/api/dict_info', methods=['POST'])
def api_dict_info():
    return jsonify({'size': len(DICTIONARY)})

@app.route('/api/corpus', methods=['POST'])
def api_corpus():
    docs = ir_system.documents
    total_tokens = sum(len(tokenize(d['title'] + ' ' + d['content'])) for d in docs)
    all_stems = set()
    for d in docs:
        tokens = tokenize(d['title'] + ' ' + d['content'])
        for t in tokens:
            all_stems.add(stem(t)[0])
    return jsonify({
        'doc_count': len(docs),
        'total_tokens': total_tokens,
        'unique_stems': len(all_stems),
        'vocab_size': len(DICTIONARY),
        'documents': [{'id': d['id'], 'title': d['title'], 'preview': d['content'][:120] + '…'} for d in docs]
    })

@app.route('/api/add_document', methods=['POST'])
def api_add_document():
    global ir_system
    title = request.json.get('title', '').strip()
    content = request.json.get('content', '').strip()
    if not title or not content:
        return jsonify({'success': False})
    new_id = max(d['id'] for d in ir_system.documents) + 1
    ir_system.documents.append({'id': new_id, 'title': title, 'content': content})
    ir_system = OromoIRSystem(ir_system.documents)
    return jsonify({'success': True, 'id': new_id})

if __name__ == '__main__':
    print("Starting Afaan Oromo NLP System on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
