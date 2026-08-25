from ssfl.telemetry import filter_batch_events


def test_batch_event_filter_preserves_higher_level_events() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def callback(event: str, fields: dict[str, object]) -> None:
        events.append((event, fields))

    filtered = filter_batch_events(callback, log_every_batch=False)
    assert filtered is not None
    filtered("training_batch", {"batch": 1})
    filtered("prediction_batch", {"batch": 2})
    filtered("training_epoch", {"epoch": 1})
    filtered("client_phase_end", {"round": 1})

    assert events == [
        ("training_epoch", {"epoch": 1}),
        ("client_phase_end", {"round": 1}),
    ]


def test_batch_event_filter_can_be_disabled_for_debugging() -> None:
    events: list[str] = []

    def callback(event: str, fields: dict[str, object]) -> None:
        del fields
        events.append(event)

    unfiltered = filter_batch_events(callback, log_every_batch=True)
    assert unfiltered is callback
    unfiltered("evaluation_batch", {"batch": 1})

    assert events == ["evaluation_batch"]
