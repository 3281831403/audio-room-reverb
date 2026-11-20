"""Room acoustics and convolution primitives for the audio-room-reverb task.

Conventions
-----------
* Coordinates are metres in a right-handed x/y/z space.
* ``room`` is ``{"size": [x, y, z], "materials": {surface: absorption}}``.
  The six surface names are floor, ceiling, front, back, left and right.
* RT60 values are seconds; sample rates are integer Hz.
* Dry/wet ratios are clamped to the inclusive range 0.0..1.0.
* Numeric assertions use an absolute tolerance of 1e-6 unless stated otherwise.
"""

import math

SPEED_OF_SOUND = 343.0
SABINE_CONSTANT = 0.161
TAIL_SEED = 0x9E3779B9
DEFAULT_MAX_ORDER = 2
SURFACES = ("floor", "ceiling", "front", "back", "left", "right")

# Axis -> (negative face, positive face).  x is width, y is depth, z is height.
_AXIS_FACES = (
    ("left", "right"),
    ("front", "back"),
    ("floor", "ceiling"),
)

_MASK64 = (1 << 64) - 1


class ReverbError(Exception):
    """Raised for invalid rooms, geometry, sample rates, ratios or batches."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value):
    return _is_number(value) and math.isfinite(float(value))


def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _room_size(room):
    """Validate *room* and return its three positive dimensions."""
    if not isinstance(room, dict):
        raise ReverbError("room must be a mapping")
    size = room.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ReverbError("room size must contain exactly three dimensions")
    dims = []
    for value in size:
        if not _finite_number(value) or float(value) <= 0.0:
            raise ReverbError("room dimensions must be positive finite numbers")
        dims.append(float(value))
    materials = room.get("materials", {})
    if materials is None:
        materials = {}
    if not isinstance(materials, dict):
        raise ReverbError("room materials must be a mapping")
    for surface in SURFACES:
        if surface in materials and not _finite_number(materials[surface]):
            raise ReverbError("absorption for %r must be numeric" % surface)
    return tuple(dims)


def _room_materials(room):
    materials = room.get("materials", {})
    return materials if isinstance(materials, dict) else {}


def _raw_alpha(room, surface):
    materials = _room_materials(room)
    value = materials.get(surface, 0.0)
    if not _finite_number(value):
        raise ReverbError("absorption for %r must be numeric" % surface)
    return float(value)


def _alpha(room, surface):
    """Absorption clamped to the physical 0.0..1.0 interval."""
    return _clamp(_raw_alpha(room, surface), 0.0, 1.0)


def _surface_areas(dims):
    x, y, z = dims
    return {
        "floor": x * y,
        "ceiling": x * y,
        "front": x * z,
        "back": x * z,
        "left": y * z,
        "right": y * z,
    }


def _vector(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ReverbError("%s must be a three-element vector" % label)
    result = []
    for component in value:
        if not _finite_number(component):
            raise ReverbError("%s components must be finite numbers" % label)
        result.append(float(component))
    return result


def _positive_sample_rate(sample_rate):
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        raise ReverbError("sample_rate must be a positive integer")
    return sample_rate


# ---------------------------------------------------------------------------
# Deterministic noise helpers
# ---------------------------------------------------------------------------


def _mix64(value):
    """One round of the SplitMix64 finalizer (pure integer arithmetic)."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _quantize(value):
    """Stable integer fingerprint for a float used as hashing input."""
    return int(round(float(value) * 1000000.0)) & _MASK64


def _tail_seed(dims, decay, sample_rate):
    seed = TAIL_SEED
    for value in (dims[0], dims[1], dims[2], decay, sample_rate):
        seed = _mix64(seed ^ _quantize(value))
    return seed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rt60(room, absorption) -> float:
    """Return the Sabine reverberation time RT60 in seconds.

    Contract:
      * Both the volume ``V = x*y*z`` and the total surface area
        ``S = 2*(x*y + x*z + y*z)`` are used:
        ``RT60 = 0.161 * V / (S * alpha_bar)``.
      * When ``absorption`` is a number it is the mean absorption coefficient.
        When it is ``None``, ``alpha_bar`` is the area-weighted mean of the
        six values in ``room["materials"]`` (missing surfaces count as 0.0).
      * Individual absorption values are clamped to 0.0..1.0 before use.
      * The result is in seconds and must be finite and non-negative.
      * An empty/malformed room, non-positive dimensions, or a resulting
        ``alpha_bar`` of zero raises ``ReverbError``.
      * Numeric tolerance: 1e-6.

    This function must use both V and S: volume alone makes small rooms sound
    like cathedrals because it ignores the comparatively large surface area.
    """
    x, y, z = _room_size(room)

    if absorption is None:
        areas = _surface_areas((x, y, z))
        total_area = sum(areas.values())
        weighted = sum(areas[s] * _alpha(room, s) for s in SURFACES)
        alpha_bar = weighted / total_area
    else:
        if not _finite_number(absorption):
            raise ReverbError("absorption must be a finite number or None")
        alpha_bar = _clamp(float(absorption), 0.0, 1.0)

    if alpha_bar <= 0.0:
        raise ReverbError("mean absorption must be greater than zero")

    volume = x * y * z
    total_area = 2.0 * (x * y + x * z + y * z)
    result = SABINE_CONSTANT * volume / (total_area * alpha_bar)
    if not math.isfinite(result) or result < 0.0:
        raise ReverbError("RT60 is not finite")
    return result


def _image_position(index, length, source_position):
    """Image-source coordinate for signed reflection index along one axis."""
    offset = source_position if index % 2 == 0 else length - source_position
    return index * length + offset


def _bounce_faces(axis, index):
    """Face names visited along one axis, source-outwards order."""
    negative, positive = _AXIS_FACES[axis]
    count = abs(index)
    if count == 0:
        return []
    first = positive if index > 0 else negative
    second = negative if index > 0 else positive
    return [first if step % 2 == 0 else second for step in range(count)]


def early_reflections(room, source, listener, max_order) -> list[dict]:
    """Return image-source early reflections sorted by increasing delay.

    Contract:
      * ``source`` and ``listener`` are ``[x, y, z]`` points in metres.
      * Every result item is a dict with exactly these keys:
        ``{"delay": float, "gain": float, "surface": str}``.
      * The direct path is always included first with ``delay == 0.0``,
        ``gain == 1.0`` and ``surface == "direct"``.
      * For images with reflection order ``1..max_order``:
        ``delay = path_length / 343.0`` seconds and
        ``gain = (1.0 / (1.0 + path_length)) *
        product(sqrt(1.0 - clamp(alpha_face, 0.0, 1.0)))``.
        ``surface`` is the final face name; multi-bounce paths join face names
        with ``">"``.
      * At most one image exists for each signed axis combination whose
        absolute axis indices sum to at most ``max_order``. Consequently
        ``max_order == 1`` yields 1 direct + 6 first-order paths, and
        ``max_order == 0`` yields only the direct path.
      * ``max_order`` must be a non-negative integer; otherwise, or for an
        empty/malformed room or bad vector, raise ``ReverbError``.
      * The returned list is sorted ascending by delay so no reflection can
        precede the direct sound. Numeric tolerance: 1e-6.
    """
    dims = _room_size(room)
    if (
        not isinstance(max_order, int)
        or isinstance(max_order, bool)
        or max_order < 0
    ):
        raise ReverbError("max_order must be a non-negative integer")
    src = _vector(source, "source")
    lis = _vector(listener, "listener")

    reflections = [{"delay": 0.0, "gain": 1.0, "surface": "direct"}]

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                indices = (nx, ny, nz)
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > max_order:
                    continue

                image = []
                faces = []
                for axis, index in enumerate(indices):
                    length = dims[axis]
                    image.append(
                        _image_position(index, length, src[axis])
                    )
                    faces.extend(_bounce_faces(axis, index))

                path_length = math.sqrt(
                    sum((image[axis] - lis[axis]) ** 2 for axis in range(3))
                )
                gain = 1.0 / (1.0 + path_length)
                for face in faces:
                    gain *= math.sqrt(1.0 - _alpha(room, face))

                reflections.append(
                    {
                        "delay": path_length / SPEED_OF_SOUND,
                        "gain": gain,
                        "surface": ">".join(faces),
                    }
                )

    reflections.sort(
        key=lambda item: (
            item["delay"],
            0 if item["surface"] == "direct" else 1,
        )
    )
    return reflections


def late_tail(room, rt60, sample_rate, length) -> list[float]:
    """Return a deterministic exponentially decaying noise tail.

    Contract:
      * ``rt60`` is the decay time in seconds and ``sample_rate`` is in Hz.
      * ``length`` is the requested maximum number of samples. The returned
        list has exactly ``min(length, max(1, round(rt60 * sample_rate)))``
        samples, so it never extends beyond one RT60 (a fixed five-second tail
        is incorrect). ``length == 0`` returns ``[]``.
      * The noise is generated by an integer hash seeded by the module
        constant ``TAIL_SEED`` together with the room dimensions, ``rt60`` and
        ``sample_rate``. It is fully deterministic (no ``random`` module and no
        sleeping), so identical calls return identical lists.
      * The envelope decays by 60 dB over ``rt60`` seconds.
      * Invalid/negative ``length``, non-positive ``rt60`` or sample rate, or
        an empty/malformed room raise ``ReverbError``.
      * Numeric tolerance: 1e-6.
    """
    dims = _room_size(room)

    if not _finite_number(rt60) or float(rt60) <= 0.0:
        raise ReverbError("rt60 must be a positive finite number")
    decay = float(rt60)

    if not _finite_number(sample_rate) or float(sample_rate) <= 0.0:
        raise ReverbError("sample_rate must be a positive number")
    rate = float(sample_rate)

    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ReverbError("length must be a non-negative integer")
    if length == 0:
        return []

    sample_count = min(length, max(1, int(round(decay * rate))))

    seed = _tail_seed(dims, decay, rate)
    tail = []
    for i in range(sample_count):
        hashed = _mix64(seed ^ (((i + 1) * 0x9E3779B97F4A7C15) & _MASK64))
        noise = (hashed >> 11) * (1.0 / float(1 << 53)) * 2.0 - 1.0
        time = i / rate
        envelope = 10.0 ** (-3.0 * time / decay)
        tail.append(noise * envelope)
    return tail


def make_ir(room, source, listener, sample_rate) -> list[float]:
    """Synthesize a mono room impulse response (IR).

    Contract:
      * ``sample_rate`` is in Hz and must be a positive integer.
      * Early reflections come from
        ``early_reflections(room, source, listener, DEFAULT_MAX_ORDER)``.
      * The late tail comes from
        ``late_tail(room, T60, sample_rate,
        max(1, round(T60 * sample_rate)))`` where ``T60`` is the room's Sabine
        RT60 in seconds.
      * The direct sound is at sample 0; each reflection is placed at
        ``round(delay * sample_rate)``; the tail starts after the last early
        reflection. Total length is
        ``1 + max(1, round(max_delay * sample_rate)) +
        max(1, round(T60 * sample_rate))``, with at most one sample of
        placement/rounding error.
      * The IR is not normalized.
      * An empty/malformed room, invalid vectors, or a non-positive sample rate
        raise ``ReverbError``. Numeric tolerance: 1e-6.
    """
    rate = _positive_sample_rate(sample_rate)
    dims = _room_size(room)

    reflections = early_reflections(room, source, listener, DEFAULT_MAX_ORDER)
    decay = rt60(room, None)

    max_delay = max(item["delay"] for item in reflections)
    reflection_span = max(1, int(round(max_delay * rate)))
    tail_length = max(1, int(round(decay * rate)))
    tail = late_tail(room, decay, rate, tail_length)

    total = 1 + reflection_span + tail_length
    ir = [0.0] * total

    # The direct path sits at sample zero with unit gain.
    ir[0] = 1.0
    for item in reflections:
        if item["surface"] == "direct":
            continue
        index = int(round(item["delay"] * rate))
        if 0 <= index < total:
            ir[index] += item["gain"]

    tail_start = 1 + reflection_span
    for offset, value in enumerate(tail):
        index = tail_start + offset
        if index < total:
            ir[index] += value

    return ir


def convolve(samples, ir) -> list[float]:
    """Return the full linear convolution of two finite sequences.

    Contract:
      * ``out[n] = sum_k samples[k] * ir[n-k]`` over valid indices.
      * The output length is exactly ``len(samples) + len(ir) - 1`` for two
        non-empty inputs; the final IR-length part of the tail must not be
        dropped.
      * An empty ``ir`` returns a copy of ``samples``; an empty ``samples``
        returns ``[]`` (including when both are empty).
      * No normalization or truncation is applied.
      * Numeric tolerance: 1e-6.
    """
    samples = list(samples)
    ir = list(ir)

    if not samples:
        return []
    if not ir:
        return samples

    output = [0.0] * (len(samples) + len(ir) - 1)
    for i, sample in enumerate(samples):
        if sample == 0:
            continue
        for j, coefficient in enumerate(ir):
            output[i + j] += sample * coefficient
    return output


def mix_dry_wet(dry, wet, ratio) -> list[float]:
    """Return the linear dry/wet mix ``(1-ratio)*dry + ratio*wet``.

    Contract:
      * ``ratio`` is clamped to 0.0..1.0 before mixing. Thus 2.0 behaves as
        1.0, -0.5 behaves as 0.0, and ``ratio == 0`` returns the dry signal.
      * ``dry`` and ``wet`` must have equal lengths; the result has that same
        length. Mismatched lengths raise ``ReverbError``.
      * A non-finite/non-numeric ratio raises ``ReverbError``.
      * Numeric tolerance: 1e-6.
    """
    if not _finite_number(ratio):
        raise ReverbError("ratio must be a finite number")
    if not isinstance(dry, (list, tuple)) or not isinstance(wet, (list, tuple)):
        raise ReverbError("dry and wet must be sequences")
    if len(dry) != len(wet):
        raise ReverbError("dry and wet must have equal lengths")

    mix = _clamp(float(ratio), 0.0, 1.0)
    return [
        (1.0 - mix) * float(dry_sample) + mix * float(wet_sample)
        for dry_sample, wet_sample in zip(dry, wet)
    ]


def apply_distance(samples, distance, ref_distance) -> list[float]:
    """Apply inverse-distance amplitude attenuation.

    Contract:
      * Amplitude gain is 1.0 when ``distance <= ref_distance`` (the reference
        distance is the 0 dB point and must not be attenuated to 0.5).
      * Beyond the reference distance the gain is
        ``ref_distance / distance``; twice the reference distance therefore
        gives 0.5 (-6.02 dB). Nearer sources are not boosted above 1.0.
      * ``distance`` must be finite and non-negative; ``ref_distance`` must be
        finite and strictly positive. Invalid values raise ``ReverbError``.
      * The returned list has the same length as ``samples``.
      * Numeric tolerance: 1e-6.
    """
    if not _finite_number(distance) or float(distance) < 0.0:
        raise ReverbError("distance must be a non-negative finite number")
    if not _finite_number(ref_distance) or float(ref_distance) <= 0.0:
        raise ReverbError("ref_distance must be a positive finite number")
    if not isinstance(samples, (list, tuple)):
        raise ReverbError("samples must be a sequence")

    current = float(distance)
    reference = float(ref_distance)
    gain = 1.0 if current <= reference else reference / current
    return [float(sample) * gain for sample in samples]


def reverb_batch(tracks, rooms, config) -> dict:
    """Process several tracks atomically and preserve their input order.

    Contract:
      * ``tracks`` is a list of dicts with required keys ``name``, ``samples``,
        ``room``, ``source`` and ``listener``. Optional keys are
        ``sample_rate`` (must match ``config["sample_rate"]`` when present),
        ``ratio`` (defaults to ``config.get("ratio", 0.5)``), ``distance``
        (default 0.0) and ``ref_distance`` (default 1.0).
      * ``rooms`` maps room names to room dicts. ``config`` must contain a
        positive integer ``sample_rate`` in Hz.
      * Each track gets its own freshly generated IR for its own room:
        ``ir = make_ir(room, source, listener, sample_rate)``,
        ``wet = convolve(samples, ir)``,
        ``result = apply_distance(mix_dry_wet(samples, wet, ratio), ...)``.
        IRs must never be shared or cached between tracks/rooms.
      * Every track, room and sample rate is validated before any synthesis.
        If any track is invalid (including a sample-rate mismatch), raise
        ``ReverbError`` and return no partial result: the batch rolls back as
        a whole.
      * Return ``{"sample_rate": sr, "tracks": [{"name", "room", "ir",
        "samples"}, ...]}`` where the inner ``samples`` is the processed
        signal and the list is in the same order as the input.
      * Numeric tolerance: 1e-6.
    """
    if not isinstance(config, dict):
        raise ReverbError("config must be a mapping")
    sample_rate = config.get("sample_rate")
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        raise ReverbError("config sample_rate must be a positive integer")

    if not isinstance(rooms, dict):
        raise ReverbError("rooms must be a mapping")
    if not isinstance(tracks, list):
        raise ReverbError("tracks must be a list")

    default_ratio = config.get("ratio", 0.5)
    if not _finite_number(default_ratio):
        raise ReverbError("config ratio must be a finite number")

    # Validate every room up front, before any synthesis happens.
    for name, room in rooms.items():
        _room_size(room)
        rt60(room, None)
        if not isinstance(name, str):
            raise ReverbError("room names must be strings")

    # Validate every track up front as well; this makes the batch atomic.
    plans = []
    for track in tracks:
        if not isinstance(track, dict):
            raise ReverbError("each track must be a mapping")
        for key in ("name", "samples", "room", "source", "listener"):
            if key not in track:
                raise ReverbError("track is missing required key %r" % key)

        samples = track["samples"]
        if not isinstance(samples, (list, tuple)):
            raise ReverbError("track samples must be a sequence")
        for sample in samples:
            if not _finite_number(sample):
                raise ReverbError("track samples must be finite numbers")

        room_name = track["room"]
        if not isinstance(room_name, str) or room_name not in rooms:
            raise ReverbError("track references unknown room %r" % (room_name,))

        source = _vector(track["source"], "source")
        listener = _vector(track["listener"], "listener")

        if "sample_rate" in track:
            track_rate = track["sample_rate"]
            if (
                not isinstance(track_rate, int)
                or isinstance(track_rate, bool)
                or track_rate != sample_rate
            ):
                raise ReverbError(
                    "track sample_rate %r does not match config sample_rate %r"
                    % (track_rate, sample_rate)
                )

        ratio = track.get("ratio", default_ratio)
        if not _finite_number(ratio):
            raise ReverbError("track ratio must be a finite number")

        distance = track.get("distance", 0.0)
        if not _finite_number(distance) or float(distance) < 0.0:
            raise ReverbError("track distance must be a non-negative number")

        ref_distance = track.get("ref_distance", 1.0)
        if not _finite_number(ref_distance) or float(ref_distance) <= 0.0:
            raise ReverbError("track ref_distance must be positive")

        plans.append(
            {
                "name": track["name"],
                "samples": list(samples),
                "room": room_name,
                "source": source,
                "listener": listener,
                "ratio": float(ratio),
                "distance": float(distance),
                "ref_distance": float(ref_distance),
            }
        )

    # All inputs are valid: synthesize, one independent IR per track.
    results = []
    for plan in plans:
        room = rooms[plan["room"]]
        ir = make_ir(room, plan["source"], plan["listener"], sample_rate)
        wet = convolve(plan["samples"], ir)
        # Zero-pad the dry signal to the convolution length so the mix keeps
        # the full reverb tail instead of truncating it.
        dry = plan["samples"] + [0.0] * (len(wet) - len(plan["samples"]))
        mixed = mix_dry_wet(dry, wet, plan["ratio"])
        processed = apply_distance(mixed, plan["distance"], plan["ref_distance"])
        results.append(
            {
                "name": plan["name"],
                "room": plan["room"],
                "ir": ir,
                "samples": processed,
            }
        )

    return {"sample_rate": sample_rate, "tracks": results}
