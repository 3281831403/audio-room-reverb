"""Red tests for the audio-room-reverb exercise.

Every test deliberately calls an unimplemented stub, so the initial suite is
expected to report exactly 15 failures and no collection/import errors.
"""

import pytest

from reverb.room import (
    ReverbError,
    apply_distance,
    convolve,
    early_reflections,
    late_tail,
    make_ir,
    mix_dry_wet,
    reverb_batch,
    rt60,
)

SURFACES = ("floor", "ceiling", "front", "back", "left", "right")


def _room(size=(2.0, 2.0, 2.0), absorption=0.2):
    """Build a valid six-surface room description."""
    return {
        "size": list(size),
        "materials": {surface: absorption for surface in SURFACES},
    }


# ---------------------------------------------------------------------------
# One test per defect (10)
# ---------------------------------------------------------------------------


def test_rt60_uses_volume_and_surface_area():
    """Defect 1: volume-only RT60 makes compact rooms sound too large."""
    compact = _room((2.0, 2.0, 2.0), 0.2)  # V=8, S=24
    slab = _room((1.0, 2.0, 4.0), 0.2)  # V=8, S=28

    assert rt60(compact, 0.2) == pytest.approx(0.161 * 8.0 / (24.0 * 0.2), abs=1e-6)
    assert rt60(slab, 0.2) == pytest.approx(0.161 * 8.0 / (28.0 * 0.2), abs=1e-6)


def test_early_reflections_are_sorted_by_delay():
    """Defect 2: unsorted reflections can arrive before the direct sound."""
    room = _room((4.0, 3.0, 2.5), 0.25)
    source = [0.4, 0.6, 0.3]
    listener = [3.2, 2.1, 1.7]

    reflections = early_reflections(room, source, listener, 2)
    delays = [reflection["delay"] for reflection in reflections]

    assert reflections[0] == {"delay": 0.0, "gain": 1.0, "surface": "direct"}
    assert delays == sorted(delays)
    assert all(delay >= 0.0 for delay in delays)


def test_image_sources_respect_max_order():
    """Defect 3: unbounded image sources explode exponentially."""
    room = _room((3.0, 2.5, 2.0), 0.3)
    source = [0.5, 0.5, 0.5]
    listener = [2.0, 1.5, 1.2]

    first_order = early_reflections(room, source, listener, 1)
    second_order = early_reflections(room, source, listener, 2)

    assert len(first_order) == 1 + 6
    assert len(second_order) == 1 + 6 + 18


def test_late_tail_is_truncated_at_rt60():
    """Defect 4: a fixed five-second tail ignores the actual RT60."""
    room = _room((3.0, 3.0, 3.0), 0.1)

    tail = late_tail(room, 0.5, 1000, 5000)
    repeat = late_tail(room, 0.5, 1000, 5000)

    assert len(tail) == 500
    assert any(abs(sample) > 0.0 for sample in tail)
    assert tail == repeat


def test_convolve_keeps_full_output_length():
    """Defect 5: dropping one IR length truncates the convolution tail."""
    samples = [1.0, 2.0, 3.0]
    ir = [0.5, 0.5]

    result = convolve(samples, ir)

    assert len(result) == 4
    assert result == pytest.approx([0.5, 1.5, 2.5, 1.5], abs=1e-6)


def test_dry_wet_ratio_is_clamped():
    """Defect 6: ratio=2 doubles the level instead of clamping to wet."""
    dry = [1.0, -1.0]
    wet = [2.0, -2.0]

    assert mix_dry_wet(dry, wet, 2.0) == pytest.approx(wet, abs=1e-6)
    assert mix_dry_wet(dry, wet, 1.5) == pytest.approx(wet, abs=1e-6)
    assert mix_dry_wet(dry, wet, -3.0) == pytest.approx(dry, abs=1e-6)
    assert mix_dry_wet(dry, wet, 0.25) == pytest.approx([1.25, -1.25], abs=1e-6)


def test_distance_gain_is_unity_at_reference():
    """Defect 7: a 0.5 gain at the reference distance loses 6 dB."""
    samples = [1.0, -2.0, 0.5]

    assert apply_distance(samples, 2.0, 2.0) == pytest.approx(samples, abs=1e-6)
    assert apply_distance(samples, 1.0, 2.0) == pytest.approx(samples, abs=1e-6)
    assert apply_distance(samples, 4.0, 2.0) == pytest.approx(
        [0.5, -1.0, 0.25], abs=1e-6
    )


def test_absorption_is_clamped_to_unit_interval():
    """Defect 8: a negative absorption produces a reflection gain above 1."""
    room = _room((2.0, 2.0, 2.0), -0.5)
    source = [0.005, 1.0, 1.0]
    listener = [0.015, 1.0, 1.0]

    reflections = early_reflections(room, source, listener, 1)

    assert len(reflections) == 1 + 6
    assert all(0.0 <= reflection["gain"] <= 1.0 + 1e-9 for reflection in reflections)


def test_batch_generates_a_separate_ir_per_room():
    """Defect 9: sharing one IR bleeds one room's reverb into another room."""
    rooms = {
        "booth": _room((2.0, 2.0, 2.0), 0.9),
        "hall": _room((8.0, 5.0, 4.0), 0.1),
    }
    tracks = [
        {
            "name": "voice",
            "samples": [1.0, 0.5, 0.0],
            "room": "booth",
            "source": [0.5, 0.5, 0.5],
            "listener": [1.5, 1.5, 1.5],
        },
        {
            "name": "music",
            "samples": [0.5, 0.25, 0.0],
            "room": "hall",
            "source": [1.0, 1.0, 1.0],
            "listener": [7.0, 4.0, 3.0],
        },
    ]

    result = reverb_batch(tracks, rooms, {"sample_rate": 1000, "ratio": 0.4})
    rows = result["tracks"]

    assert [row["name"] for row in rows] == ["voice", "music"]
    assert rows[0]["ir"] != rows[1]["ir"]
    assert len(rows[0]["ir"]) != len(rows[1]["ir"])


def test_batch_rejects_sample_rate_mismatch():
    """Defect 10: silently mixing sample rates shifts the resulting pitch."""
    room = _room((3.0, 2.5, 2.0), 0.3)
    rooms = {"room": room}
    tracks = [
        {
            "name": "valid",
            "samples": [1.0, 0.0],
            "room": "room",
            "source": [0.5, 0.5, 0.5],
            "listener": [2.0, 1.5, 1.2],
            "sample_rate": 8000,
        },
        {
            "name": "mismatch",
            "samples": [1.0, 0.0],
            "room": "room",
            "source": [0.5, 0.5, 0.5],
            "listener": [2.0, 1.5, 1.2],
            "sample_rate": 16000,
        },
    ]

    with pytest.raises(ReverbError):
        reverb_batch(tracks, rooms, {"sample_rate": 8000})


# ---------------------------------------------------------------------------
# Combination / edge tests (5)
# ---------------------------------------------------------------------------


def test_max_order_zero_returns_direct_only():
    """Combination: zero-order image sources reduce to the direct path."""
    room = _room((3.0, 2.0, 2.5), 0.2)

    reflections = early_reflections(room, [0.5, 0.5, 0.5], [2.0, 1.5, 1.8], 0)

    assert reflections == [{"delay": 0.0, "gain": 1.0, "surface": "direct"}]


def test_zero_wet_ratio_returns_dry_signal():
    """Combination: a 0.0 wet ratio is an exact dry pass-through."""
    dry = [1.0, -0.25, 0.5]
    wet = [0.2, 0.3, -0.4]

    assert mix_dry_wet(dry, wet, 0.0) == pytest.approx(dry, abs=1e-6)


def test_empty_ir_convolution_returns_input():
    """Combination: an empty IR cannot silently erase the dry signal."""
    samples = [0.25, -0.5, 0.75]

    assert convolve(samples, []) == pytest.approx(samples, abs=1e-6)
    assert convolve([], [1.0, 2.0]) == []


def test_empty_room_raises_reverb_error():
    """Combination: an empty room description is not processable."""
    with pytest.raises(ReverbError):
        make_ir({}, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 8000)


def test_batch_preserves_track_order():
    """Combination: successful batches keep the exact input order."""
    room = _room((3.0, 2.5, 2.0), 0.3)
    names = ["intro", "loop", "outro"]
    tracks = [
        {
            "name": name,
            "samples": [1.0, 0.0, 0.5],
            "room": "room",
            "source": [0.5, 0.5, 0.5],
            "listener": [2.0, 1.5, 1.2],
        }
        for name in names
    ]

    result = reverb_batch(tracks, {"room": room}, {"sample_rate": 1000})

    assert [row["name"] for row in result["tracks"]] == names
