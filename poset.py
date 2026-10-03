"""Core partial-order logic for the Hasse Diagram Builder.

Nothing in here knows about any particular puzzle. A puzzle is simply a set of
element names plus a list of (smaller, larger) pairs. From that we compute:

* the transitive closure (every relation a < b that follows from the pairs),
* the cover relations (a < b with nothing strictly between them) - these are
  exactly the edges of the Hasse diagram,
* a verdict on a diagram drawn by the player.
"""
import re

MAX_NODES = 16
MAX_LABEL = 12
MAX_PAIRS = 400
EPS = 8.0  # minimum vertical gap (board units) for two nodes to count as "different heights"

_RESERVED = {"__proto__", "constructor", "prototype"}
_OPS = re.compile(r"(<=|>=|≤|≥|->|→|<|>)")


class PosetError(ValueError):
    """Raised for any invalid user input; the message is safe to show to players."""


def clean_label(raw):
    s = str(raw).strip()
    if not s:
        raise PosetError("An element name is empty.")
    if len(s) > MAX_LABEL:
        raise PosetError(f"'{s[:MAX_LABEL]}…' is longer than {MAX_LABEL} characters.")
    if any(ord(c) < 32 for c in s):
        raise PosetError("Element names cannot contain control characters.")
    if s in _RESERVED:
        raise PosetError(f"'{s}' is a reserved word and cannot be used as an element name.")
    return s


class Poset:
    """A finite strict partial order built from generating pairs (lower, upper)."""

    def __init__(self, nodes, pairs):
        nodes = [clean_label(n) for n in nodes]
        if len(set(nodes)) != len(nodes):
            raise PosetError("Element names must be unique.")
        if len(nodes) < 2:
            raise PosetError("A puzzle needs at least 2 elements.")
        if len(nodes) > MAX_NODES:
            raise PosetError(f"At most {MAX_NODES} elements are supported (got {len(nodes)}).")

        known = set(nodes)
        seen, clean = set(), []
        for a, b in pairs:
            if a not in known or b not in known:
                raise PosetError(f"Relation uses unknown element: {a} , {b}.")
            if a == b or (a, b) in seen:
                continue  # reflexive pairs are implied in every poset; duplicates are noise
            seen.add((a, b))
            clean.append((a, b))
        if not clean:
            raise PosetError("Add at least one relation between two different elements.")
        if len(clean) > MAX_PAIRS:
            raise PosetError(f"At most {MAX_PAIRS} relations are supported (got {len(clean)}).")

        self.nodes = nodes
        self.pairs = clean

        adj = {n: [] for n in nodes}
        for a, b in clean:
            adj[a].append(b)

        cycle = self._find_cycle(adj)
        if cycle:
            raise PosetError(
                "These relations form a cycle: " + " ≤ ".join(cycle) +
                ". A partial order is antisymmetric, so this would force them all to be equal."
            )

        memo = {}

        def reach(u):
            if u not in memo:
                s = set()
                for v in adj[u]:
                    s.add(v)
                    s |= reach(v)
                memo[u] = s
            return memo[u]

        self.up = {n: set(reach(n)) for n in nodes}  # strictly greater elements
        self.down = {n: set() for n in nodes}        # strictly smaller elements
        for a, ups in self.up.items():
            for b in ups:
                self.down[b].add(a)

        self.covers = sorted(
            (a, b)
            for a in nodes
            for b in self.up[a]
            if not any(b in self.up[c] for c in self.up[a])
        )
        self.cover_set = set(self.covers)

    @staticmethod
    def _find_cycle(adj):
        color = {n: 0 for n in adj}
        stack = []

        def dfs(u):
            color[u] = 1
            stack.append(u)
            for v in adj[u]:
                if color[v] == 1:
                    return stack[stack.index(v):] + [v]
                if color[v] == 0:
                    found = dfs(v)
                    if found:
                        return found
            color[u] = 2
            stack.pop()
            return None

        for n in adj:
            if color[n] == 0:
                found = dfs(n)
                if found:
                    return found
        return None

    def less(self, a, b):
        return b in self.up[a]


# --------------------------------------------------------------------------- parsing

def _split_tokens(text):
    return [t for t in re.split(r"[,\s;]+", text.strip()) if t]


def parse_custom(nodes_in, rel_in):
    """Turn free-form player input into (nodes, pairs).

    Accepted relation syntax, one statement per line (or separated by ';'):
        a < b          a <= b        a ≤ b        a -> b
        b > a          b >= a
        a < b < c      (chains)
        a, b           (pair meaning a below b)
        # comment
    Elements may also be listed explicitly; any element mentioned in a relation
    is added automatically.
    """
    nodes, seen = [], set()

    def add(label):
        label = clean_label(label)
        if label not in seen:
            seen.add(label)
            nodes.append(label)
        return label

    if isinstance(nodes_in, str):
        tokens = _split_tokens(nodes_in)
    else:
        tokens = [str(t) for t in (nodes_in or [])]
    for t in tokens:
        add(t)

    pairs = []

    def add_pair(lo, hi):
        pairs.append((add(lo), add(hi)))

    def parse_line(line):
        line = line.split("#", 1)[0].strip()
        if not line:
            return
        if _OPS.search(line):
            parts = _OPS.split(line)
            for i in range(1, len(parts), 2):
                left, op, right = parts[i - 1].strip(), parts[i], parts[i + 1].strip()
                if not left or not right:
                    raise PosetError(f"Could not read the relation: '{line}'.")
                if op in (">", ">=", "≥"):
                    left, right = right, left
                add_pair(left, right)
            return
        if "," in line:
            bits = [b.strip() for b in line.split(",")]
        else:
            bits = line.split()
        if len(bits) != 2 or not all(bits):
            raise PosetError(
                f"Could not read the relation: '{line}'. Use forms like 'a < b', 'a <= b' or 'a, b'."
            )
        add_pair(bits[0], bits[1])

    if isinstance(rel_in, str):
        for line in re.split(r"[\n;]+", rel_in):
            parse_line(line)
    else:
        for item in rel_in or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                add_pair(item[0], item[1])
            elif isinstance(item, str):
                parse_line(item)
            else:
                raise PosetError("Relations must be strings or [lower, upper] pairs.")

    if len(nodes) > MAX_NODES:
        raise PosetError(f"At most {MAX_NODES} elements are supported (got {len(nodes)}).")
    return nodes, pairs


# --------------------------------------------------------------------------- validation

def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PosetError("Node positions must be numbers.")
    if v != v or v in (float("inf"), float("-inf")):
        raise PosetError("Node positions must be finite numbers.")
    return float(v)


def _read_diagram(poset, positions, edges):
    if not isinstance(positions, dict) or not isinstance(edges, list):
        raise PosetError("Malformed diagram.")
    ys = {}
    for n in poset.nodes:
        p = positions.get(n)
        if not isinstance(p, dict) or "y" not in p:
            raise PosetError(f"Element {n} has not been placed on the board.")
        ys[n] = _num(p["y"])
    clean = []
    for e in edges:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            raise PosetError("Each edge must be a pair of element names.")
        a, b = e
        if a not in ys or b not in ys or a == b:
            raise PosetError("An edge refers to an unknown element.")
        clean.append((a, b))
    return ys, clean


def check_diagram(poset, positions, edges):
    """Judge a player's diagram.

    ``positions`` maps element -> {x, y} in screen units (larger y = lower on screen);
    ``edges`` is a list of undirected [a, b] pairs. A diagram is solved when

    1. the drawn edges are exactly the cover relations,
    2. every edge is oriented correctly (the greater element is drawn higher),
    3. every relation a < b (not only covers) places a strictly below b.
    """
    ys, edge_list = _read_diagram(poset, positions, edges)

    statuses, messages = [], []
    drawn = set()
    for a, b in edge_list:
        key = frozenset((a, b))
        if key in drawn:
            statuses.append("duplicate")
            continue
        drawn.add(key)

        if abs(ys[a] - ys[b]) <= EPS:
            statuses.append("flat")
            messages.append(f"{a}–{b} is a horizontal line. Connected elements must sit at different heights.")
            continue
        lo, hi = (a, b) if ys[a] > ys[b] else (b, a)  # bigger y is lower on screen

        if (lo, hi) in poset.cover_set:
            statuses.append("ok")
        elif (hi, lo) in poset.cover_set:
            statuses.append("inverted")
            messages.append(f"{hi} is drawn above {lo}, but {hi} ≤ {lo}. Greater elements go higher.")
        elif poset.less(lo, hi) or poset.less(hi, lo):
            small, big = (lo, hi) if poset.less(lo, hi) else (hi, lo)
            mid = sorted(c for c in poset.up[small] if big in poset.up[c])
            statuses.append("redundant")
            messages.append(
                f"{small}–{big} is implied by transitivity (through {mid[0]}), so it is not an edge of a Hasse diagram."
            )
        else:
            statuses.append("unrelated")
            messages.append(f"{a} and {b} are incomparable, so they must not be joined.")

    missing = [(lo, hi) for lo, hi in poset.covers if frozenset((lo, hi)) not in drawn]

    placement = []
    for a in poset.nodes:
        for b in sorted(poset.up[a]):
            if not ys[a] > ys[b] + EPS:
                placement.append((a, b))
    for a, b in placement[:4]:
        messages.append(f"{b} must be placed higher than {a}, because {a} ≤ {b}.")
    if len(placement) > 4:
        messages.append(f"…and {len(placement) - 4} more height problems.")

    bad = sum(1 for s in statuses if s not in ("ok", "duplicate"))
    if missing:
        messages.append(
            f"{len(missing)} covering relation{'s are' if len(missing) != 1 else ' is'} still missing an edge."
        )

    return {
        "solved": not bad and not missing and not placement,
        "edge_status": statuses,
        "counts": {
            "ok": statuses.count("ok"),
            "bad": bad,
            "missing": len(missing),
            "placement": len(placement),
        },
        "messages": messages,
        "_missing": missing,
        "_placement": placement,
    }


def suggest_hint(poset, positions, edges):
    """Return one concrete next step for the player."""
    res = check_diagram(poset, positions, edges)
    if res["solved"]:
        return {"type": "none", "text": "Nothing to fix. This diagram is already correct."}
    if res["_missing"]:
        lo, hi = sorted(res["_missing"])[0]
        return {
            "type": "edge", "nodes": [lo, hi],
            "text": f"{hi} covers {lo}: nothing lies strictly between them. Join them with an edge.",
        }
    for (a, b), st in zip(edges, res["edge_status"]):
        if st not in ("ok", "duplicate"):
            return {"type": "remove", "nodes": [a, b],
                    "text": f"The edge {a}–{b} does not belong (or is drawn the wrong way round)."}
    if res["_placement"]:
        a, b = res["_placement"][0]
        return {"type": "move", "nodes": [a, b], "text": f"{b} should sit higher than {a}."}
    return {"type": "none", "text": "Look at the feedback list for what to adjust."}
