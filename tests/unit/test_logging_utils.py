"""Per-call fields passed to ``log_event`` must survive into the JSON line -- a prior bug had
``_Adapter.process`` looking for a top-level ``fields`` kwarg while ``log_event`` nested them
inside ``extra``, so every round/phase/metric field was silently dropped from events.jsonl.
"""

import io
import json

import pytest

from ssfl.logging_utils import ForbiddenLogFieldError, bind, configure_logging, log_event


def _last_line(stream: io.StringIO) -> dict:
    return json.loads(stream.getvalue().strip().splitlines()[-1])


def test_log_event_includes_bound_and_call_fields():
    stream = io.StringIO()
    logger = bind(configure_logging(stream=stream), run_id="r1", algorithm="ssfl")
    log_event(logger, "aggregate", round=2, phase="train", valid_rate=0.9, rejected_count=1)
    line = _last_line(stream)
    assert line["run_id"] == "r1"
    assert line["algorithm"] == "ssfl"
    assert line["round"] == 2
    assert line["phase"] == "train"
    assert line["valid_rate"] == 0.9
    assert line["rejected_count"] == 1


def test_log_event_call_fields_vary_across_calls():
    stream = io.StringIO()
    logger = bind(configure_logging(stream=stream), run_id="r1")
    log_event(logger, "aggregate", round=1)
    log_event(logger, "aggregate", round=2)
    lines = [json.loads(x) for x in stream.getvalue().strip().splitlines()]
    assert [line["round"] for line in lines] == [1, 2]


def test_log_event_rejects_forbidden_field():
    stream = io.StringIO()
    logger = bind(configure_logging(stream=stream), run_id="r1")
    with pytest.raises(ForbiddenLogFieldError):
        log_event(logger, "leak", weights=[1, 2, 3])


def test_bind_rejects_forbidden_field():
    with pytest.raises(ForbiddenLogFieldError):
        bind(configure_logging(stream=io.StringIO()), state_dict="x")
