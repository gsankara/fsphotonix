"""Measurement and characterisation of an assembled system.

These are the numbers a paper or a design review actually needs: the end-to-end
transfer matrix, the optical power budget, cross-talk between output channels,
fan-out uniformity, and the chromatic walk-off of a WDM source through a
dispersive hologram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import engine as eng
from .fields import FieldSet
from .grid import Grid
from .system.optical_system import OpticalSystem, Trace


def transfer_matrix(system: OpticalSystem, source_stage: str, detector_stage: str,
                    n_in: int | None = None, probe: float = 1.0) -> np.ndarray:
    """Probe each emitter in turn and record the detector response.

    The result is the machine's real linear map, including every loss and
    aberration in the model.  Reprogramming the holograms and re-measuring is the
    standard way to check that a design does what the maths asked for.
    """
    src = system[source_stage].source
    n = n_in if n_in is not None else src.n_emitters
    saved = src.get_drive() if hasattr(src, "get_drive") else None
    rows = []
    try:
        for j in range(n):
            e = np.zeros(n)
            e[j] = probe
            src.set_drive(e)
            out = system.run(keep_fields=False).outputs[detector_stage].vector()
            rows.append(out / probe)
    finally:
        if saved is not None:
            src.set_drive(saved)
    return np.asarray(rows, dtype=float).T  # (n_out, n_in)


def crosstalk_db(T: np.ndarray) -> np.ndarray:
    """Per-input cross-talk: strongest unintended output vs the intended one [dB]."""
    T = np.abs(np.asarray(T, dtype=float))
    out = np.empty(T.shape[1])
    for j in range(T.shape[1]):
        col = T[:, j].copy()
        i = int(np.argmax(col))
        want = col[i]
        col[i] = 0.0
        other = col.max() if col.size > 1 else 0.0
        out[j] = 10 * np.log10(max(other, 1e-300) / max(want, 1e-300))
    return out


def uniformity(values) -> float:
    """``(max-min)/(max+min)`` -- the standard fan-out uniformity figure."""
    v = np.abs(np.asarray(values, dtype=float).ravel())
    if v.size == 0 or v.max() <= 0:
        return 1.0
    return float((v.max() - v.min()) / (v.max() + v.min()))


def power_budget(trace: Trace) -> dict:
    """Optical power at each plane, plus the loss of each hop, in dB."""
    p = {k: v.total_power() for k, v in trace.fields.items()}
    names = trace.order
    loss = {}
    for a, b in zip(names[:-1], names[1:]):
        if p.get(a, 0) > 0:
            loss[f"{a}->{b}"] = 10 * np.log10(max(p[b], 1e-300) / p[a])
    detected = {k: float(np.sum(v.optical_power)) if v.optical_power is not None else np.nan
                for k, v in trace.outputs.items()}
    return {"power_W": p, "hop_loss_dB": loss, "detected_W": detected}


def diffraction_efficiency(modulator, wavelength: float, order=(1, 0)) -> dict:
    """Per-region efficiency of a chosen order, and where the rest of the light goes."""
    out = {}
    for r in modulator.regions:
        orders = r.orders(wavelength)
        tot = sum(o.efficiency for _, o in orders) or 1.0
        want = sum(o.efficiency for _, o in orders if (o.mx, o.my) == tuple(order))
        zero = sum(o.efficiency for _, o in orders if (o.mx, o.my) == (0, 0))
        out[r.label or f"r{len(out)}"] = {
            "target_order": want, "zero_order": zero,
            "other": tot - want - zero, "accounted": tot,
        }
    return out


def chromatic_walkoff(grating, wavelengths, z: float) -> np.ndarray:
    """Screen displacement of a grating's first order versus wavelength [m].

    A diffractive deflector is dispersive: this is the number that decides whether
    a WDM bank can share one hologram or needs per-wavelength correction.
    """
    out = []
    for lam in wavelengths:
        os_ = [o for o in grating.orders(lam) if (o.mx, o.my) != (0, 0)]
        if not os_:
            out.append((np.nan, np.nan))
            continue
        o = os_[0]
        out.append(o.screen_position(lam, z))
    return np.asarray(out, dtype=float)


def spot_metrics(fields: FieldSet, positions, radius: float) -> dict:
    """Power, centroid error and peak irradiance around each expected spot."""
    I = eng.to_numpy(fields.mean_intensity())
    X, Y = (eng.to_numpy(a) for a in fields.grid.meshgrid())
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    power, cerr, peak = [], [], []
    for px, py in pos:
        m = ((X - px) ** 2 + (Y - py) ** 2) <= radius**2
        w = I[m]
        s = float(w.sum())
        power.append(s * fields.grid.cell_area)
        if s > 0:
            cx = float((I[m] * X[m]).sum() / s)
            cy = float((I[m] * Y[m]).sum() / s)
            cerr.append(float(np.hypot(cx - px, cy - py)))
        else:
            cerr.append(np.nan)
        peak.append(float(w.max()) if w.size else 0.0)
    return {"power_W": np.asarray(power), "centroid_error_m": np.asarray(cerr),
            "peak_irradiance": np.asarray(peak),
            "uniformity": uniformity(power),
            "collected_fraction": float(np.sum(power) / max(fields.total_power(), 1e-300))}


def illumination_map(fields: FieldSet, modulator) -> dict:
    """Where each channel's light actually lands on the modulator panel.

    Returns the ``(n_channels, n_regions)`` power matrix, plus the power that misses
    every region -- the light that falls in the guard gaps between the holograms or
    off the panel altogether.  This is the measurement that turns the illumination
    footprint into a number: widen the collimated spot and the off-diagonal terms
    and the spill both grow.
    """
    grid = fields.grid
    masks = []
    for r in modulator.regions:
        masks.append(eng.to_numpy(r.aperture.mask(grid, r.center[0], r.center[1])).ravel())
    M = np.asarray(masks) if masks else np.zeros((0, grid.size))
    union = (M.sum(axis=0) > 0).astype(float) if len(M) else np.zeros(grid.size)

    P = np.zeros((len(fields.channels), len(modulator.regions)))
    total = np.zeros(len(fields.channels))
    spill = np.zeros(len(fields.channels))
    for i, ch in enumerate(fields.channels):
        I = (eng.to_numpy(eng.xp.abs(ch.u) ** 2).ravel() * grid.cell_area
             * ch.waveform.mean_power())
        total[i] = I.sum()
        if len(M):
            P[i] = M @ I
        spill[i] = total[i] - float(union @ I)
    return {
        "power_W": P,
        "incident_W": total,
        "spill_W": spill,
        "spill_fraction": spill / np.maximum(total, 1e-300),
        "region_labels": [r.label for r in modulator.regions],
    }


def illumination_crosstalk(fields: FieldSet, modulator, mapping=None) -> dict:
    """Illumination cross-talk of a source array onto a modulator panel.

    ``mapping[i]`` is the region that channel ``i`` is *supposed* to illuminate
    (default: channel ``i`` -> region ``i``).  The figure of merit is the ratio of
    the power an **input** puts on foreign regions to the power it puts on its own,
    because light landing on a neighbour's hologram is diffracted to that
    neighbour's screen targets and is then indistinguishable from signal.

    Results are aggregated *per input*, not per channel: several channels can serve
    one input (a main beam plus the lenslet's stray background, or several WDM
    lines), and only their sum is physically meaningful -- a per-channel ratio would
    report the same figure for a 1% and a 10% stray component.
    ``per_channel_crosstalk_db`` keeps the unaggregated view.
    """
    m = illumination_map(fields, modulator)
    P = m["power_W"]
    n_ch, n_rg = P.shape
    if mapping is None:
        mapping = [i % n_rg for i in range(n_ch)] if n_rg else []
    mapping = list(mapping)

    own_ch = np.array([P[i, mapping[i]] for i in range(n_ch)]) if n_rg else np.zeros(n_ch)
    foreign_ch = P.sum(axis=1) - own_ch
    with np.errstate(divide="ignore", invalid="ignore"):
        ch_db = 10 * np.log10(np.where(own_ch > 0,
                                       foreign_ch / np.maximum(own_ch, 1e-300), np.nan))

    inputs = sorted(set(mapping))
    own = np.zeros(len(inputs))
    foreign = np.zeros(len(inputs))
    incident = np.zeros(len(inputs))
    for k, inp in enumerate(inputs):
        idx = [i for i, v in enumerate(mapping) if v == inp]
        own[k] = own_ch[idx].sum()
        foreign[k] = foreign_ch[idx].sum()
        incident[k] = m["incident_W"][idx].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        xt_db = 10 * np.log10(np.where(own > 0, foreign / np.maximum(own, 1e-300), np.nan))
    return {
        **m,
        "mapping": mapping,
        "inputs": inputs,
        "own_W": own,
        "foreign_W": foreign,
        "crosstalk_db": xt_db,
        "per_channel_crosstalk_db": ch_db,
        "worst_crosstalk_db": float(np.nanmax(xt_db)) if len(inputs) else float("nan"),
        "capture_efficiency": own / np.maximum(incident, 1e-300),
    }


def illumination_sweep(source_factory, modulator, grid, diameters, mapping=None) -> dict:
    """Sweep the illuminated footprint and report the capture/cross-talk trade-off.

    ``source_factory(diameter) -> LightSource``.  The result is the curve that
    decides the collimator design: capture efficiency rises with footprint until the
    hologram aperture is filled, then flattens, while cross-talk keeps climbing once
    the footprint approaches the region pitch.
    """
    out = {"diameter_m": [], "capture": [], "crosstalk_db": [], "spill": []}
    for d in diameters:
        src = source_factory(float(d))
        f = src.emit(grid)
        r = illumination_crosstalk(f, modulator, mapping=mapping)
        out["diameter_m"].append(float(d))
        out["capture"].append(float(np.mean(r["capture_efficiency"])))
        out["crosstalk_db"].append(float(np.nanmean(r["crosstalk_db"])))
        out["spill"].append(float(np.mean(r["spill_fraction"])))
    return {k: np.asarray(v) for k, v in out.items()}


def cgh_gap_report(modulator, z: float, wavelength: float | None = None,
                   guard_waists: float = 2.0) -> dict:
    """Whether the two hologram gaps are big enough, and what they cost.

    Combines :meth:`ModulatorArray.gap_report` with the detector-side consequence:
    a sub-aperture of width ``a`` produces a spot of width ``~lambda*z/a`` on the
    screen, so shrinking the sub-apertures (larger intra-input gap, more targets per
    input) blurs the spots until neighbouring screen targets overlap.  ``ok`` is true
    when the required guard (``guard_waists`` spot widths) fits inside the target
    pitch.
    """
    lam = wavelength or modulator.default_wavelength
    g = modulator.gap_report(z=z, wavelength=lam)
    spot = g["spot_size_um"] * 1e-6
    tp = g["target_pitch_um"] * 1e-6 if g["target_pitch_um"] == g["target_pitch_um"] \
        else float("nan")
    need = guard_waists * spot
    return {
        **g,
        "required_target_pitch_um": need * 1e6,
        "detector_guard_ok": bool(tp > need) if tp == tp else None,
        "aperture_loss_db": (10 * np.log10(max(g["subaperture_um"], 1e-9)
                                          / max(g["subaperture_um"]
                                                + g["intra_input_gap_um"], 1e-9))),
        "illumination_guard_um": g["inter_input_gap_um"],
    }


def energy_check(fields_in: FieldSet, fields_out: FieldSet) -> dict:
    """Sanity check that a propagation step conserved power."""
    a, b = fields_in.total_power(), fields_out.total_power()
    return {"in_W": a, "out_W": b,
            "ratio": b / a if a > 0 else np.nan,
            "loss_dB": 10 * np.log10(max(b, 1e-300) / max(a, 1e-300))}
