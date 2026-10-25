"""Deterministic MinHash candidates, verified by token-shingle Jaccard.
Similarity grouping does not infer semantic topics or detect all paraphrases.
"""
import hashlib
import re
from collections import defaultdict
import numpy as np


def text_fingerprint(text):
    canonical = re.sub(r"\s+", " ", str(text)).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def shingle_set(text):
    words = re.findall(r"[^\s।!?.,;:]+", text)
    return {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))}


def group_documents(df, similarity=.9):
    df = df.copy().reset_index(drop=True)
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    df["text_hash"] = df.text.map(text_fingerprint)
    for column in ["topic_id", "parent_id", "prompt_id", "text_hash"]:
        if column not in df:
            continue
        seen = {}
        for i, value in enumerate(df[column].fillna("").astype(str)):
            if value.strip() and value.lower() not in {"unknown", "none", "nan"}:
                if value in seen:
                    union(i, seen[value])
                else:
                    seen[value] = i
    rng = np.random.default_rng(0)
    a = rng.integers(1, 2**32, 64, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**32, 64, dtype=np.uint64)
    buckets, shingles, matches = defaultdict(list), [], 0
    for i, text in enumerate(df.text):
        current = shingle_set(text)
        shingles.append(current)
        hashes = np.array([int.from_bytes(hashlib.blake2s(s.encode(), digest_size=4).digest(), "little") for s in sorted(current)], dtype=np.uint64)
        signature = ((hashes[:, None] * a + b) & np.uint64(2**32 - 1)).min(axis=0)
        candidates = set()
        keys = [(band, signature[band * 4:(band + 1) * 4].tobytes()) for band in range(16)]
        for key in keys:
            if len(buckets[key]) > 10000:
                raise ValueError("Excessive duplicate candidates. Deduplicate the corpus first.")
            candidates.update(buckets[key])
        for j in sorted(candidates):
            if root(i) != root(j) and len(current & shingles[j]) / len(current | shingles[j]) >= similarity:
                union(i, j)
                matches += 1
        for key in keys:
            buckets[key].append(i)
    df["group_id"] = [f"document_group_{root(i):06d}" for i in range(len(df))]
    real_topics = "topic_id" in df and df.topic_id.fillna("").astype(str).str.strip().ne("").all()
    return df, {"group_count": int(df.group_id.nunique()), "near_duplicate_links": matches,
        "jaccard_threshold": similarity, "grouping": "topic-and-document" if real_topics else "document-and-near-duplicate",
        "limitation": None if real_topics else "Genuine topic IDs unavailable; this is not topic-held-out. LSH is approximate; paraphrases may remain undetected."}
