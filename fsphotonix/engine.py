"""Numeric backend and FFT helpers.

The whole simulator goes through this module for array creation and FFTs so that
a CUDA backend can be swapped in globally:

    import fsphotonix as fp
    fp.engine.use_backend("cupy")      # everything downstream runs on GPU

It also provides :func:`zoom_dft2`, a Bluestein chirp-z transform that evaluates
a 2-D DFT on an *arbitrary uniform output grid*.  That is what makes
"small modulator, big screen" geometries tractable: the detection plane sampling
is chosen independently of the modulator sampling, with no zero padding.
"""

from __future__ import annotations

import numpy as _np

_BACKEND = "numpy"
xp = _np  # active array module
COMPLEX = _np.complex128
REAL = _np.float64


def use_backend(name: str = "numpy") -> str:
    """Select ``'numpy'`` or ``'cupy'`` as the array backend."""
    global xp, _BACKEND
    if name == "numpy":
        xp, _BACKEND = _np, "numpy"
    elif name in ("cupy", "gpu"):
        import cupy  # noqa: PLC0415  (optional dependency)

        xp, _BACKEND = cupy, "cupy"
    else:
        raise ValueError(f"unknown backend {name!r}")
    return _BACKEND


def backend() -> str:
    return _BACKEND


def use_precision(single: bool = False) -> None:
    """Switch between complex128 (default) and complex64 working precision."""
    global COMPLEX, REAL
    COMPLEX = _np.complex64 if single else _np.complex128
    REAL = _np.float32 if single else _np.float64


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype)


def to_numpy(a):
    """Bring an array back to host memory (no-op on the numpy backend)."""
    get = getattr(a, "get", None)
    return get() if callable(get) else _np.asarray(a)


def zeros(shape, dtype=None):
    return xp.zeros(shape, dtype=dtype if dtype is not None else COMPLEX)


def fft2(a):
    return xp.fft.fft2(a)


def ifft2(a):
    return xp.fft.ifft2(a)


def fftshift(a):
    return xp.fft.fftshift(a)


def ifftshift(a):
    return xp.fft.ifftshift(a)


def fftfreq(n, d):
    return xp.fft.fftfreq(n, d=d)


# --------------------------------------------------------------------------- #
# Chirp-z / zoom DFT
# --------------------------------------------------------------------------- #
def _next_fast_len(n: int) -> int:
    """Smallest power of two >= n (portable across backends)."""
    return 1 << int(n - 1).bit_length()


def czt(a, m: int, w: complex, a0: complex = 1.0, axis: int = -1):
    """Chirp-z transform ``X[k] = sum_n a[n] * a0**n * w**(n*k)``, k = 0..m-1.

    Bluestein's algorithm: O((n+m) log(n+m)) instead of O(n*m).
    """
    a = xp.moveaxis(xp.asarray(a), axis, -1)
    n = int(a.shape[-1])
    nfft = _next_fast_len(n + m - 1)

    # chirp c[j] = w**(-j^2/2) sampled for j = -(n-1) .. (m-1)
    j = _np.arange(-(n - 1), m, dtype=_np.float64)
    c_seq = asarray(w ** (-(j**2) / 2.0), dtype=COMPLEX)

    nn = _np.arange(n, dtype=_np.float64)
    g = a * asarray((a0**nn) * (w ** (nn**2 / 2.0)), dtype=COMPLEX)

    conv = xp.fft.ifft(
        xp.fft.fft(g, nfft, axis=-1) * xp.fft.fft(c_seq, nfft), axis=-1
    )[..., n - 1 : n - 1 + m]

    kk = _np.arange(m, dtype=_np.float64)
    out = conv * asarray(w ** (kk**2 / 2.0), dtype=COMPLEX)
    return xp.moveaxis(out, -1, axis)


def dtft_axis(a, x0: float, dx: float, f_out, axis: int = -1):
    """Evaluate ``sum_n a[n] exp(-2j*pi*f*x_n)`` on a uniform ``f_out`` grid.

    ``f_out`` must be uniformly spaced; ``x_n = x0 + n*dx``.
    """
    f_out = _np.asarray(to_numpy(f_out), dtype=_np.float64)
    m = int(f_out.shape[0])
    f0 = float(f_out[0])
    df = float(f_out[1] - f_out[0]) if m > 1 else 0.0
    # exp(-2j pi (f0 + k df)(x0 + n dx))
    #   = exp(-2j pi f_k x0) * exp(-2j pi f0 n dx) * exp(-2j pi df dx n k)
    out = czt(
        a,
        m,
        w=_np.exp(-2j * _np.pi * df * dx),
        a0=_np.exp(-2j * _np.pi * f0 * dx),
        axis=axis,
    )
    phase = asarray(_np.exp(-2j * _np.pi * f_out * x0), dtype=COMPLEX)
    shape = [1] * out.ndim
    shape[axis if axis >= 0 else out.ndim + axis] = m
    return out * phase.reshape(shape)


def zoom_dft2(u, x_in, y_in, fx_out, fy_out):
    """2-D DTFT of ``u`` evaluated on the uniform output frequency grid.

    ``x_in``/``y_in`` are the input sample coordinate vectors, ``fx_out``/
    ``fy_out`` the (uniform) output spatial-frequency vectors.  ``u`` is indexed
    ``[y, x]``.  Returns the *unscaled* sum; multiply by ``dx*dy`` for the
    continuous-integral approximation.
    """
    dx = float(x_in[1] - x_in[0])
    dy = float(y_in[1] - y_in[0])
    tmp = dtft_axis(u, float(x_in[0]), dx, fx_out, axis=-1)
    return dtft_axis(tmp, float(y_in[0]), dy, fy_out, axis=-2)
