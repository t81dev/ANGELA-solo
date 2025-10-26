
class SharedGraph:
    """
    Υ Meta-Subjective Architecting: a minimal shared perspective graph.

    API (as per manifest "upcoming"):
      - add(view) -> view_id
      - diff(peer) -> Dict
      - merge(strategy) -> Dict
    """
    def __init__(self) -> None:
        self._graph = DiGraph()
        self._views: Dict[str, GraphView] = {}
        self._last_merge: Optional[Dict[str, Any]] = None

    def add(self, view: Dict[str, Any]) -> str:
        if not isinstance(view, dict):
            raise TypeError("view must be a dictionary")
        view_id = f"view_{uuid.uuid4().hex[:8]}"
        gv = GraphView(id=view_id, payload=view, ts=time.time())
        self._views[view_id] = gv

        # store nodes/edges if present, else stash payload as node
        nodes = view.get("nodes", [])
        edges = view.get("edges", [])
        if nodes and isinstance(nodes, list):
            for n in nodes:
                nid = n.get("id") or f"n_{uuid.uuid4().hex[:6]}"
                self._graph.add_node(nid, **{k: v for k, v in n.items() if k != "id"})
        else:
            self._graph.add_node(view_id, payload=view)
        if edges and isinstance(edges, list):
            for e in edges:
                src, dst = e.get("src"), e.get("dst")
                if src and dst:
                    self._graph.add_edge(src, dst, **{k: v for k, v in e.items() if k not in ("src", "dst")})
        return view_id

    def diff(self, peer: "SharedGraph") -> Dict[str, Any]:
        """Return a shallow, conflict-aware diff summary vs peer graph."""
        if not isinstance(peer, SharedGraph):
            raise TypeError("peer must be SharedGraph")

        self_nodes = set(self._graph.nodes())
        peer_nodes = set(peer._graph.nodes())
        added = list(self_nodes - peer_nodes)
        removed = list(peer_nodes - self_nodes)
        common = self_nodes & peer_nodes

        conflicts = []
        for n in common:
            a = self._graph.nodes[n]
            b = peer._graph.nodes[n]
            # simple attribute-level conflict detection
            for k in set(a.keys()) | set(b.keys()):
                if k in a and k in b and a[k] != b[k]:
                    conflicts.append({"node": n, "key": k, "left": a[k], "right": b[k]})

        return {"added": added, "removed": removed, "conflicts": conflicts, "ts": time.time()}

    def merge(self, strategy: str = "prefer_recent") -> Dict[str, Any]:
        """
        Merge internal views into a single perspective.
        Strategies:
          - prefer_recent (default): pick newer attribute values
          - prefer_majority: pick most frequent value (by view occurrence)
        """
        if strategy not in ("prefer_recent", "prefer_majority"):
            raise ValueError("Unsupported merge strategy")

        # Aggregate attributes from views
        attr_hist: Dict[Tuple[str, str], List[Tuple[Any, float]]] = defaultdict(list)
        for gv in self._views.values():
            payload = gv.payload
            nodes = payload.get("nodes") or [{"id": gv.id, **payload}]
            for n in nodes:
                nid = n.get("id") or gv.id
                for k, v in n.items():
                    if k == "id":
                        continue
                    attr_hist[(nid, k)].append((v, gv.ts))

        merged_nodes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for (nid, key), vals in attr_hist.items():
            if strategy == "prefer_recent":
                v = sorted(vals, key=lambda x: x[1], reverse=True)[0][0]
            else:  # prefer_majority
                counter = Counter([vv for vv, _ in vals])
                v = counter.most_common(1)[0][0]
            merged_nodes[nid][key] = v

        merged = {"nodes": [{"id": nid, **attrs} for nid, attrs in merged_nodes.items()], "strategy": strategy, "ts": time.time()}
        self._last_merge = merged
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Ethical Sandbox Containment (isolated what-if scenarios)
# ─────────────────────────────────────────────────────────────────────────────
