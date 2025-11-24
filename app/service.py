from dataclasses import dataclass
from typing import List, Tuple
from app.schemas import InferenceRequest, TraceItem
from app.nn.modules import default_modules

@dataclass
class Pipeline:
    modules: list

    def run(self, req: InferenceRequest) -> Tuple[str, List[TraceItem]]:
        trace: List[TraceItem] = []
        outputs: List[str] = []
        for m in self.modules:
            ok, s = m.decide(req)
            out = m.run(req) if ok else ""
            trace.append(TraceItem(module=m.name, accepted=ok, output=out, score=s))
            if ok and m.name != "fallback":
                outputs.append(out)
                break  # one-shot routing; fallback handles the rest
        if not outputs:
            # only fallback accepted
            outputs = [t.output for t in trace if t.module == "fallback"]
        reply = "\n".join(filter(None, outputs))
        return reply, trace

_pipeline: Pipeline | None = None

def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(modules=default_modules())
    return _pipeline