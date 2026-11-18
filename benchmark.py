"""Evaluate chronological movie recommendations on a frozen MovieLens sample."""
import argparse
from collections import Counter
import csv
from datetime import UTC, datetime
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import requests
from scipy import sparse

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/ml-latest-small.zip"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
SHA = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
METHODS = ("popularity", "cosine", "pagerank")
DAY = 86400
K = 10


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def fingerprint(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def acquire():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(URL, stream=True, timeout=30) as response:
            response.raise_for_status()
            data = response.raw.read(2_000_001)
        if sha256(data).hexdigest() != SHA:
            raise ValueError("Official MovieLens archive changed")
        temporary = SOURCE.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(SOURCE)
    if fingerprint(SOURCE) != SHA:
        raise ValueError("MovieLens archive hash differs")


def load_movies():
    acquire()
    with zipfile.ZipFile(SOURCE) as archive:
        filename = "ml-latest-small/movies.csv"
        with archive.open(filename) as source, io.TextIOWrapper(source, encoding="utf-8", newline="") as text:
            movies = {int(r["movieId"]): {"title": r["title"], "genres": r["genres"].split("|")}
                      for r in csv.DictReader(text)}
        filename = "ml-latest-small/ratings.csv"
        with archive.open(filename) as source, io.TextIOWrapper(source, encoding="utf-8", newline="") as text:
            rows = sorted([(int(row["timestamp"]), int(row["userId"]), int(row["movieId"]), float(row["rating"]))
                          for row in csv.DictReader(text)])
    if len({(u, m) for _, u, m, _ in rows}) != len(rows):
        raise ValueError("Duplicate user and movie pair")
    if any(movie not in movies or not 0.5 <= rating <= 5 or timestamp <= 0 for timestamp, _, movie, rating in rows):
        raise ValueError("Invalid user rating")
    return rows, movies


def cutoffs(rows):
    timestamps = [row[0] for row in rows]
    return tuple(int(v) for v in np.quantile(timestamps, [0.70, 0.85], method="nearest"))


def first_targets(rows, start, end):
    selected = {}
    for timestamp, user, movie, rating in rows:
        if start <= timestamp < end and rating >= 4:
            selected.setdefault(user, movie)
    return selected


class Snapshot:
    def __init__(self, rows, as_of):
        self.cutoff = as_of
        self.history = [row for row in rows if row[0] < as_of]
        self.users = {user: i for i, user in enumerate(sorted({u for _, u, _, _ in self.history}))}
        self.movies = np.array(sorted({m for _, _, m, _ in self.history}))
        self.positions = {movie: i for i, movie in enumerate(self.movies)}
        self.seen = {u: set() for u in self.users}
        positives_i, positives_j = [], []
        self.popular = np.zeros(len(self.movies))
        for time, user, movie, rating in self.history:
            p = self.positions[movie]
            self.seen[user].add(p)
            if rating >= 4:
                positives_i.append(self.users[user])
                positives_j.append(p)
                self.popular[p] += 0.5 ** ((as_of - time) / (365.25 * DAY))
        shape = len(self.users), len(self.movies)
        data = np.ones(len(positives_i))
        self.likes = sparse.csr_array((data, (positives_i, positives_j)), shape=shape)
        normalized = self.likes.multiply(1 / np.sqrt(np.maximum(1, np.asarray(self.likes.sum(axis=0)))))
        self.cosine = (normalized.T @ normalized).tocsr()
        self.adjacency = sparse.bmat([[None, self.likes], [self.likes.T, None]], format="csr")
        self.degree = np.asarray(self.adjacency.sum(axis=1)).ravel()
        inverse = 1 / np.maximum(1, self.degree)
        self.transition = (self.adjacency @ sparse.diags(inverse)).tocsr()

    def score(self, user, method):
        if method not in METHODS:
            raise ValueError("Unknown method")
        if method == "popularity" or user not in self.users or self.degree[self.users[user]] == 0:
            return self.popular
        u = self.users[user]
        if method == "cosine":
            indices = self.likes[u:u + 1].indices
            return np.asarray(self.cosine[indices, :].sum(axis=0)).ravel()
        start = np.zeros(len(self.degree))
        start[u] = 1
        signal = start
        for _ in range(12):
            signal = 0.20 * start + 0.80 * (self.transition @ signal)
        return signal[len(self.users):]

    def recommend(self, user, method, k=K):
        remaining = np.ones(len(self.movies), dtype=bool)
        remaining[list(self.seen.get(user, ()))] = False
        allowed = np.flatnonzero(remaining)
        scores = self.score(user, method)[allowed]
        order = np.lexsort((self.movies[allowed], -self.popular[allowed], -scores))
        return self.movies[allowed[order[:k]]].tolist()


def evaluate(snapshot, targets):
    history_sizes = {user: len(history) for user, history in snapshot.seen.items()}
    slices = {}
    for user, movie in targets.items():
        count = history_sizes.get(user, 0)
        slices[user] = "new_user" if count == 0 else "1–20" if count <= 20 else "21+"
    eligible = sum(movie in snapshot.positions for movie in targets.values())
    counts = Counter(slices.values())
    methods, latency = {}, {}
    for method in METHODS:
        durations = []
        ranks = []
        by_slice = {key: [] for key in counts}
        suggestions = set()
        warm = []
        for user, expected in targets.items():
            start = perf_counter()
            predictions = snapshot.recommend(user, method)
            durations.append((perf_counter() - start) * 1000)
            suggestions.update(predictions)
            rank = predictions.index(expected) + 1 if expected in predictions else 0
            ranks.append(rank)
            by_slice[slices[user]].append(rank)
            if expected in snapshot.positions:
                warm.append(rank)
        def scores(values):
            hits = [rank for rank in values if rank]
            return {"users": len(values), "hits": len(hits), "recall_at_10": len(hits) / len(values),
                    "ndcg_at_10": sum(1 / np.log2(rank + 1) for rank in hits) / len(values)}
        methods[method] = {
            "all": scores(ranks), "known_items": scores(warm),
            "catalog_coverage": len(suggestions) / len(snapshot.movies),
            "unique_recommended": len(suggestions),
            "by_activity": {key: scores(values) for key, values in sorted(by_slice.items())},
        }
        latency[method] = {"median_ms": float(np.median(durations)), "p95_ms": float(np.quantile(durations, 0.95))}
    return {"targets": len(targets), "known_item_targets": eligible, "cold_item_targets": len(targets) - eligible,
            "history_ratings": len(snapshot.history), "catalog": len(snapshot.movies), "methods": methods}, latency


def run(output):
    rows, movies = load_movies()
    validation, test = cutoffs(rows)
    ends = rows[-1][0] + 1
    summary = {"source": {"url": URL, "sha256": SHA, "ratings": len(rows), "catalog_rows": len(movies)}, "k": K}
    timing = {}
    for name, start, end in (("validation", validation, test), ("test", test, ends)):
        snapshot = Snapshot(rows, start)
        targets = first_targets(rows, start, end)
        metrics, timing[name] = evaluate(snapshot, targets)
        summary[name] = {"cutoff": start, "cutoff_utc": datetime.fromtimestamp(start, UTC).isoformat(), **metrics}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", summary)
    write_json(output / "latency.json", timing)
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    run(args.out)
