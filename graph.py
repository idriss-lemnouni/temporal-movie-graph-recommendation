"""Import frozen histories and verify a snapshot-aware serving cache."""
import argparse
from itertools import batched
import json
import os
from pathlib import Path
from time import sleep
from uuid import uuid4

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
import redis

from benchmark import DAY, K, SHA, cutoffs, load_movies, write_json

SCHEMA = (
    ("MovieSnapshot", "key", "snapshot_key"),
    ("Film", "key", "film_key"),
    ("Viewer", "key", "viewer_key"),
    ("Genre", "name", "genre_name"),
)

MOVIES = """
UNWIND $batch AS row
MERGE (movie:Film {key: $snapshot + ':' + toString(row.id)})
SET movie.snapshot = $snapshot, movie.id = row.id, movie.title = row.title
FOREACH (name IN row.genres |
  MERGE (genre:Genre {name: name})
  MERGE (movie)-[:HAS_GENRE]->(genre)
)
"""

RATINGS = """
UNWIND $batch AS row
MATCH (movie:Film {key: $snapshot + ':' + toString(row.movie)})
MERGE (user:Viewer {key: $snapshot + ':' + toString(row.user)})
SET user.snapshot = $snapshot, user.id = row.user
MERGE (user)-[rating:RATED {key: $snapshot + ':' + toString(row.user) + ':' + toString(row.movie)}]->(movie)
SET rating.timestamp = row.time, rating.value = row.rating
"""

CHECK = """
MATCH (u:Viewer {snapshot: $snapshot})-[r:RATED]->(m:Film)
RETURN count(r) AS ratings, count(DISTINCT u) AS users, count(DISTINCT m) AS movies,
       max(r.timestamp) AS newest,
       sum(CASE WHEN r.timestamp >= $cutoff OR m.snapshot <> $snapshot THEN 1 ELSE 0 END) AS invalid
"""

POPULAR = """
MATCH (snapshot:MovieSnapshot {key: $snapshot, ready: true})
MATCH (m:Film {snapshot: $snapshot})<-[r:RATED]-(:Viewer)
WHERE NOT EXISTS {
  MATCH (:Viewer {key: snapshot.key + ':' + toString($user)})-[:RATED]->(m)
}
WITH m, sum(CASE WHEN r.value >= 4 THEN exp(log(0.5) * (snapshot.cutoff - r.timestamp) / $half_life) ELSE 0 END) AS score
RETURN m.id AS movie ORDER BY score DESC, movie LIMIT $limit
"""


def execute(driver, query, **params):
    return driver.execute_query(query, parameters_=params, database_="neo4j").records


def snapshot_key(cutoff):
    return f"{SHA}:{cutoff}"


def cache_key(prefix, token, user, limit):
    return f"{prefix}:v1:{token}:popularity:{user}:{limit}"


def prepare(driver):
    for label, field, constraint in SCHEMA:
        execute(driver, f"CREATE CONSTRAINT {constraint} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{field} IS UNIQUE")
    execute(driver, "CREATE CONSTRAINT rating_key IF NOT EXISTS FOR ()-[r:RATED]-() REQUIRE r.key IS UNIQUE")
    execute(driver, "CREATE INDEX film_snapshot IF NOT EXISTS FOR (f:Film) ON (f.snapshot)")
    execute(driver, "CREATE INDEX viewer_snapshot IF NOT EXISTS FOR (u:Viewer) ON (u.snapshot)")
    execute(driver, "CALL db.awaitIndexes(60)")


def import_snapshot(driver, events, films, cutoff):
    token = snapshot_key(cutoff)
    rows = [row for row in events if row[0] < cutoff]
    ids = sorted({movie for _, _, movie, _ in rows})
    execute(driver, """
        MERGE (snapshot:MovieSnapshot {key: $snapshot})
        ON CREATE SET snapshot.cutoff = $cutoff, snapshot.ready = false
    """, snapshot=token, cutoff=cutoff)
    for batch in batched(({"id": movie, **films[movie]} for movie in ids), 1000):
        execute(driver, MOVIES, snapshot=token, batch=list(batch))
    for batch in batched((dict(time=t, user=u, movie=m, rating=r) for t, u, m, r in rows), 1000):
        execute(driver, RATINGS, snapshot=token, batch=list(batch))
    proof = execute(driver, CHECK, snapshot=token, cutoff=cutoff)[0].data()
    if (proof["ratings"], proof["movies"], proof["users"], proof["newest"], proof["invalid"]) != (
        len(rows), len(ids), len({user for _, user, _, _ in rows}), rows[-1][0], 0
    ):
        raise RuntimeError(f"Graph snapshot does not match the local cutoff: {proof}")
    execute(driver, "MATCH (s:MovieSnapshot {key: $snapshot}) SET s.ready = true", snapshot=token)
    return proof


def recommend(driver, cache, cutoff, user, limit=K, prefix="movielens"):
    if limit < 1:
        raise ValueError("A positive limit is required")
    token = snapshot_key(cutoff)
    key = cache_key(prefix, token, user, limit)
    read = cache.get(key)
    if read is not None:
        return json.loads(read), True
    records = execute(driver, POPULAR, snapshot=token, user=user, limit=limit, half_life=365.25 * DAY)
    movies = [record["movie"] for record in records]
    if movies:
        cache.set(key, json.dumps(movies), ex=3600)
    return movies, False


def connect(uri):
    driver = GraphDatabase.driver(uri, auth=("neo4j", os.environ["MOVIELENS_NEO4J_PASSWORD"]), connection_timeout=4)
    for attempt in range(15):
        try:
            driver.verify_connectivity()
            return driver
        except (ServiceUnavailable, SessionExpired):
            if attempt == 14:
                driver.close()
                raise
            sleep(2)


def main(output, uri, redis_url):
    from benchmark import Snapshot

    events, films = load_movies()
    boundaries = cutoffs(events)
    baseline = [Snapshot(events, cutoff) for cutoff in boundaries]
    warm_user = min(set(baseline[0].users) & set(baseline[1].users))
    prefix = f"movielens-probe-{uuid4().hex}"
    with connect(uri) as driver, redis.Redis.from_url(redis_url) as cache:
        prepare(driver)
        version = execute(driver, "CALL dbms.components() YIELD versions RETURN versions[0] AS version")[0]["version"]
        redis_version = cache.info("server")["redis_version"]
        proof = {}
        cache_trace = []
        for model, cutoff in zip(baseline, boundaries):
            name = "validation" if cutoff == boundaries[0] else "test"
            proof[name] = import_snapshot(driver, events, films, cutoff)
            expected = model.recommend(warm_user, "popularity")
            received, cached = recommend(driver, cache, cutoff, warm_user, prefix=prefix)
            repeated, reused = recommend(driver, cache, cutoff, warm_user, prefix=prefix)
            if cached or not reused or received != expected or repeated != received:
                raise RuntimeError("Cache or ranking differs from the independently evaluated local snapshot")
            cache_trace.append({"snapshot": name, "miss_then_hit": True, "top_10_matches_cpu": True})
        if import_snapshot(driver, events, films, boundaries[-1]) != proof["test"]:
            raise RuntimeError("Reimporting the test cutoff changed the graph")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"neo4j": version, "redis": redis_version, "snapshots": proof, "cache": cache_trace,
                        "reimport_preserves_counts": True})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--neo4j", default="bolt://127.0.0.1:57687")
    cli.add_argument("--redis", default="redis://127.0.0.1:56379")
    args = cli.parse_args()
    main(args.out, args.neo4j, args.redis)
