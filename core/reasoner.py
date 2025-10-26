
class Reasoner:
    def process(self, task: str, context: Dict[str, Any]) -> Any:
        return {"message": f"Processed: {task}", "context_hint": bool(context)}


# PATCH: Belief Conflict Tolerance in SharedGraph
def merge(self, strategy="default", tolerance_scoring=False):
    # existing merge logic ...
    if tolerance_scoring:
        for edge in self.graph.edges():
            self.graph.edges[edge]['confidence_delta'] = self._calculate_confidence_delta(edge)
    return self.graph


def vote_on_conflict_resolution(self, conflicts):
    votes = {c: self._score_conflict(c) > 0.5 for c in conflicts}
    return votes


### ANGELA UPGRADE: SharedGraph.ingest_events
# ingest_events monkeypatch
def __ANGELA__SharedGraph_ingest_events(*args, **kwargs):

# args: (self, events, *, source_peer, strategy='append_reconcile', clock=None)
    clock = dict(clock or {})
    applied = 0
    conflicts = 0
    # simple in-memory dedupe set
    if not hasattr(self, '_seen_event_hashes'):
        self._seen_event_hashes = set()
    for ev in events or []:
        blob = json.dumps(ev, sort_keys=True).encode('utf-8')
        h = hashlib.sha256(blob).hexdigest()
        if h in self._seen_event_hashes:
            continue
        # conflict stub: if same key present with different value -> conflict++
        if hasattr(self, '_event_index'):
            key = ev.get('id') or h
            if key in self._event_index:
                conflicts += 1
        else:
            self._event_index = {}
        key = ev.get('id') or h
        self._event_index[key] = ev
        self._seen_event_hashes.add(h)
        applied += 1
        # bump vector clock
        clock[source_peer] = int(clock.get(source_peer, 0)) + 1
    return {"applied": applied, "conflicts": conflicts, "new_clock": clock}

try:
    SharedGraph.ingest_events = __ANGELA__SharedGraph_ingest_events
except Exception as _e:
    # class may not exist; define minimal class
    class SharedGraph:  # type: ignore
        pass
    SharedGraph.ingest_events = __ANGELA__SharedGraph_ingest_events

# --- flat-layout bootstrap ---
import sys
import types
import importlib
import importlib.util
import importlib.machinery
import importlib.abc
