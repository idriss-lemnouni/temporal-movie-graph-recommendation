import unittest

from benchmark import METHODS, Snapshot, first_targets
from graph import cache_key, snapshot_key


ROWS = [
    (10, 1, 10, 4.5),
    (11, 2, 10, 5),
    (12, 2, 20, 4.5),
    (20, 1, 20, 4),
    (21, 1, 30, 5),
    (22, 3, 30, 5),
]


class TemporalTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = Snapshot(ROWS, 20)

    def test_history_and_catalog_exclude_rating_at_cutoff_and_future_item(self):
        self.assertEqual(len(self.snapshot.history), 3)
        self.assertEqual(self.snapshot.movies.tolist(), [10, 20])
        for method in METHODS:
            self.assertEqual(self.snapshot.recommend(1, method), [20])

    def test_first_relevant_target_per_user_in_half_open_interval(self):
        self.assertEqual(first_targets(ROWS, 20, 22), {1: 20})
        self.assertEqual(first_targets(ROWS, 22, 23), {3: 30})

    def test_new_user_falls_back_to_past_popularity(self):
        self.assertEqual(self.snapshot.recommend(3, "popularity"), [10, 20])
        for method in METHODS:
            self.assertEqual(self.snapshot.recommend(3, method), [10, 20])

    def test_graph_scores_place_positive_neighbors_ahead_of_disconnected_movies(self):
        more = [(9, 4, 40, 5), *ROWS]
        snap = Snapshot(more, 20)
        for model in ("cosine", "pagerank"):
            self.assertGreater(snap.score(1, model)[snap.positions[20]],
                               snap.score(1, model)[snap.positions[40]])

    def test_serving_cache_key_varies_by_cutoff_and_top_k(self):
        one = cache_key("movies", snapshot_key(20), 1, 10)
        self.assertNotEqual(one, cache_key("movies", snapshot_key(21), 1, 10))
        self.assertNotEqual(one, cache_key("movies", snapshot_key(20), 1, 5))
