from collections import Counter

from hypothesis import given
from hypothesis.strategies import lists, permutations


# We need at least three candidates and at least three voters to have a
# paradox; anything less can only lead to victories or at worst ties.
@given(lists(permutations(["A", "B", "C"]), min_size=3))
def test_elections_are_transitive(election):
    all_candidates = {"A", "B", "C"}

    # First calculate the pairwise counts of how many prefer each candidate
    # to the other
    counts = Counter()
    for vote in election:
        for i in range(len(vote)):
            for j in range(i + 1, len(vote)):
                counts[(vote[i], vote[j])] += 1

    # Now look at which pairs of candidates one has a majority over the
    # other and store that.
    graph = {}
    for i in all_candidates:
        for j in all_candidates:
            if counts[(i, j)] > counts[(j, i)]:
                graph.setdefault(i, set()).add(j)

    # Now for each triple assert that it is transitive.
    for x in all_candidates:
        for y in graph.get(x, ()):
            for z in graph.get(y, ()):
                assert x not in graph.get(z, ())
