"""Procedural puzzle generation.

Every puzzle is produced from parameters plus a random seed, so the same seed
reproduces the same puzzle but no puzzle is stored or hard-coded anywhere.
Players can also override the ingredients (their own element names, integers,
or base set) and the generators build the poset around that input.
"""
import math
import random
import re
from itertools import combinations

from poset import MAX_NODES, Poset, PosetError, clean_label

MODES = ("random", "divisibility", "subset")


# ------------------------------------------------------------------ input helpers

def parse_str_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = [t for t in re.split(r"[,\s;]+", value.strip()) if t]
    else:
        items = [str(t) for t in value]
    return [clean_label(t) for t in items]


def parse_int_list(value):
    out = []
    for t in parse_str_list(value):
        try:
            n = int(t)
        except ValueError:
            raise PosetError(f"'{t}' is not a whole number.")
        if n < 1 or n > 10**6:
            raise PosetError("Integers must be between 1 and 1,000,000.")
        out.append(n)
    return out


def _default_labels(n):
    return [chr(65 + i) for i in range(n)]


def _isolated(poset):
    touched = set()
    for a, b in poset.covers:
        touched.update((a, b))
    return [n for n in poset.nodes if n not in touched]


# ------------------------------------------------------------------ generators

def random_poset(n, density, rng, labels=None):
    labels = labels or _default_labels(n)
    n = len(labels)
    density = min(max(density, 0.05), 0.9)
    best = None
    for _ in range(60):
        order = labels[:]
        rng.shuffle(order)  # a hidden linear extension guarantees acyclicity
        pairs = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < density]
        # attach any isolated element to a random neighbour
        touched = {x for pr in pairs for x in pr}
        for i, name in enumerate(order):
            if name not in touched:
                j = rng.choice([k for k in range(n) if k != i])
                pairs.append((order[min(i, j)], order[max(i, j)]))
                touched.update((name, order[j]))
        p = Poset(labels, pairs)
        if len(p.covers) >= max(2, n // 2 + 1):
            return p, pairs
        if best is None or len(p.covers) > len(best[0].covers):
            best = (p, pairs)
    return best


def _divisors(m):
    return [d for d in range(1, m + 1) if m % d == 0]


def divisibility_poset(n, rng, values=None):
    if values:
        vals = sorted(set(values))
    else:
        base = 210  # 2*3*5*7 has 16 divisors, enough for any allowed size
        for _ in range(400):
            m = rng.randint(6, 400)
            if len(_divisors(m)) >= n:
                base = m
                break
        ds = _divisors(base)
        vals = None
        for _ in range(40):
            cand = sorted(rng.sample(ds, min(n, len(ds))))
            pairs = [(str(a), str(b)) for a in cand for b in cand if a != b and b % a == 0]
            if pairs:
                p = Poset([str(v) for v in cand], pairs)
                if not _isolated(p):
                    vals = cand
                    break
        vals = vals or sorted(rng.sample(ds, min(n, len(ds))))
    labels = [str(v) for v in vals]
    pairs = [(str(a), str(b)) for a in vals for b in vals if a != b and b % a == 0]
    return Poset(labels, pairs), pairs


def _set_label(s):
    return "∅" if not s else "{" + ",".join(s) + "}"


def subset_poset(n, rng, elements=None):
    if elements:
        elems = list(dict.fromkeys(elements))
        if len(elems) > 5:
            raise PosetError("Use at most 5 base elements for the subset lattice.")
        if len(elems) < 2:
            raise PosetError("Use at least 2 base elements for the subset lattice.")
    else:
        k = max(2, math.ceil(math.log2(max(n, 2))))
        elems = [chr(97 + i) for i in range(k)]
    universe = [c for r in range(len(elems) + 1) for c in combinations(elems, r)]
    if elements:
        n = min(n, len(universe)) if n else len(universe)
    n = min(n, len(universe), MAX_NODES)
    chosen = None
    for _ in range(40):
        cand = rng.sample(universe, n) if n < len(universe) else list(universe)
        pairs = [(_set_label(a), _set_label(b)) for a in cand for b in cand if a != b and set(a) < set(b)]
        if pairs:
            p = Poset([_set_label(c) for c in cand], pairs)
            if not _isolated(p):
                chosen = cand
                break
    chosen = chosen or (rng.sample(universe, n) if n < len(universe) else list(universe))
    labels = [_set_label(c) for c in chosen]
    pairs = [(_set_label(a), _set_label(b)) for a in chosen for b in chosen if a != b and set(a) < set(b)]
    return Poset(labels, pairs), pairs


# ------------------------------------------------------------------ public API

def generate(mode, rng, size=6, density=0.35, labels=None, values=None, elements=None):
    if mode not in MODES:
        raise PosetError(f"Unknown mode '{mode}'.")
    size = max(3, min(int(size), MAX_NODES))

    if mode == "random":
        names = parse_str_list(labels)
        if names:
            if len(set(names)) != len(names):
                raise PosetError("Your custom element names must be unique.")
            names = names[:MAX_NODES]
            if len(names) < 3:
                raise PosetError("Give at least 3 element names, or leave the field empty.")
        poset, pairs = random_poset(size, float(density), rng, names or None)
        desc = ("Draw the Hasse diagram of the order generated by the pairs below. "
                "The list is not reduced: some pairs are implied by others.")
    elif mode == "divisibility":
        ints = parse_int_list(values)
        if ints and (len(set(ints)) < 3 or len(set(ints)) > MAX_NODES):
            raise PosetError(f"Give between 3 and {MAX_NODES} distinct integers, or leave the field empty.")
        poset, pairs = divisibility_poset(size, rng, ints or None)
        desc = "a ≤ b means that a divides b. Draw the Hasse diagram of this order."
    else:
        base = parse_str_list(elements)
        poset, pairs = subset_poset(size, rng, base or None)
        desc = "A ≤ B means A ⊆ B. Draw the Hasse diagram of these sets ordered by inclusion."

    return {"poset": poset, "given": pairs, "description": desc}


def stage_puzzle(stage, seed):
    """Campaign puzzle for a given stage: bigger and denser as the stage number grows."""
    rng = random.Random(seed)
    size = min(3 + stage, 14)
    if stage <= 2:
        mode = "random"
    else:
        mode = rng.choice(MODES)
    density = min(0.55, 0.22 + 0.03 * stage)
    out = generate(mode, rng, size=size, density=density)
    out["mode"] = mode
    return out
