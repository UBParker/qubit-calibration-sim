"""Tests for the Bloch sphere drawing helpers (viz/bloch.py)."""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402
from PIL import Image, ImageSequence  # noqa: E402

from calibration_utils.style import KET_E, KET_F, KET_G, theme_context  # noqa: E402
from viz import bloch  # noqa: E402
from viz.bloch import (  # noqa: E402
    bloch_axes,
    draw_state,
    leakage_bar,
    save_gif,
    update_leakage_bar,
)


def _rotation_through_minus_y(n: int) -> np.ndarray:
    """|g> to |e> under an x pulse: (0, -sin t, cos t)."""
    t = np.linspace(0, np.pi, n)
    return np.stack([np.zeros_like(t), -np.sin(t), np.cos(t)], axis=1)


def test_bloch_axes_renders_dark():
    with theme_context("dark") as p:
        fig = plt.figure(figsize=(3, 3))
        ax = bloch_axes(fig, 111, theme="dark")
        assert isinstance(ax, Axes3D)
        fig.canvas.draw()
        labels = {t.get_text() for t in ax.texts}
        assert {KET_G, KET_E, "+x", "+y"} <= labels
        assert to_hex(fig.get_facecolor()) == p.bg.lower()
        # The rendered image is mostly the dark background, not matplotlib's default white.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        assert rgba[..., :3].mean() < 80
        plt.close(fig)


@pytest.mark.parametrize(("elev", "azim"), [(28.0, -40.0), (10.0, 25.0)])
def test_bloch_axes_view_angles(elev, azim):
    with theme_context("dark"):
        fig = plt.figure(figsize=(3, 3))
        ax = bloch_axes(fig, 111, theme="dark", elev=elev, azim=azim)
        assert ax.elev == elev and ax.azim == azim
        plt.close(fig)


def test_bloch_axes_default_view():
    with theme_context("dark"):
        fig = plt.figure(figsize=(3, 3))
        ax = bloch_axes(fig, 111)
        assert (ax.elev, ax.azim) == (bloch.VIEW_ELEV, bloch.VIEW_AZIM) == (28.0, -40.0)
        plt.close(fig)


def test_draw_state_returns_artists():
    with theme_context("dark") as p:
        fig = plt.figure(figsize=(3, 3))
        ax = bloch_axes(fig, 111, theme="dark")
        path = _rotation_through_minus_y(12)
        artists = draw_state(ax, path[-1], p.fit, trail=path)
        assert len(artists) >= 2
        for a in artists:
            assert a.axes is ax
        fig.canvas.draw()
        no_trail = draw_state(ax, path[0], p.fit)
        assert len(no_trail) >= 1
        plt.close(fig)


def test_leakage_bar_label():
    with theme_context("dark"):
        fig, ax = plt.subplots(figsize=(1, 2))
        artists = leakage_bar(ax, 3.2e-3, "dark")
        fig.canvas.draw()
        texts = " ".join(t.get_text() for t in ax.texts)
        assert KET_F in texts and "10" in texts
        n_patches, n_texts = len(ax.patches), len(ax.texts)
        bar, value = update_leakage_bar(artists, 4.0e-2)
        assert len(ax.patches) == n_patches and len(ax.texts) == n_texts
        assert np.isclose(bar.get_height(), 4.0e-2 - 1e-5)
        assert "4.0" in value.get_text()
        update_leakage_bar(artists, 0.0)
        assert "<" in value.get_text()
        plt.close(fig)


def _render_gif(path):
    with theme_context("dark") as p:
        fig = plt.figure(figsize=(3, 3))
        ax = bloch_axes(fig, 111, theme="dark")
        path_vecs = _rotation_through_minus_y(10)
        state = {"artists": []}

        def update(i):
            for a in state["artists"]:
                a.remove()
            state["artists"] = draw_state(ax, path_vecs[i], p.fit, trail=path_vecs[: i + 1])
            return state["artists"]

        anim = FuncAnimation(fig, update, frames=10, blit=False)
        try:
            return save_gif(anim, path, fps=10)
        finally:
            plt.close(fig)


@pytest.mark.slow  # renders a GIF only for a GIF size check
def test_save_gif_small(tmp_path):
    a = _render_gif(tmp_path / "a.gif")
    b = _render_gif(tmp_path / "b.gif")
    assert a.exists() and a.stat().st_size < 3 * 1024 * 1024
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.slow  # renders a GIF only for a GIF size limit check
def test_save_gif_rejects_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(bloch, "MAX_GIF_BYTES", 1)
    out = tmp_path / "big.gif"
    with pytest.raises(ValueError, match="3 MB"):
        _render_gif(out)
    assert not out.exists()
    assert plt.get_fignums() == []


def _render_gif_with_source(path):
    """Render a short animation with a fixed label.

    Returns the GIF path, the label's pixel box (x0, y0, x1, y1) in image coordinates, and the
    last frame as matplotlib renders it at the GIF dpi.
    """
    with theme_context("dark") as p:
        fig = plt.figure(figsize=(3, 3))
        ax = bloch_axes(fig, 111, theme="dark")
        label = fig.text(0.04, 0.97, "STATIC LABEL", color=p.fg, fontsize=9, va="top")
        path_vecs = _rotation_through_minus_y(12)
        state = {"artists": []}

        def update(i):
            for a in state["artists"]:
                a.remove()
            state["artists"] = draw_state(ax, path_vecs[i], p.fit, trail=path_vecs[: i + 1])
            return state["artists"]

        anim = FuncAnimation(fig, update, frames=len(path_vecs), blit=False)
        try:
            save_gif(anim, path, fps=10)
            # The figure now shows the last frame; render it the way the writer does.
            dpi = bloch.GIF_DPI
            buf = io.BytesIO()
            fig.savefig(buf, format="rgba", dpi=dpi)
            w, h = (int(round(v * dpi)) for v in fig.get_size_inches())
            last = np.asarray(Image.frombuffer("RGBA", (w, h), buf.getbuffer(), "raw", "RGBA", 0, 1))
            ext = label.get_window_extent(fig.canvas.get_renderer())
            scale = dpi / fig.dpi
            box = (
                int(ext.x0 * scale),
                h - int(np.ceil(ext.y1 * scale)),
                int(np.ceil(ext.x1 * scale)),
                h - int(ext.y0 * scale),
            )
        finally:
            plt.close(fig)
    return path, box, last[..., :3]


def _composited_frames(path):
    with Image.open(path) as im:
        return [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)]


def test_save_gif_same_animation_gives_identical_bytes(tmp_path):
    a, _, _ = _render_gif_with_source(tmp_path / "a.gif")
    b, _, _ = _render_gif_with_source(tmp_path / "b.gif")
    assert a.read_bytes() == b.read_bytes()


def test_quantize_frames_share_one_palette():
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)) for _ in range(7)]
    out = bloch.quantize_frames(frames)
    assert all(f.mode == "P" for f in out)
    assert len({bytes(f.getpalette()) for f in out}) == 1
    # At most 255 colours, so index 255 stays free for transparency.
    assert max(int(np.asarray(f).max()) for f in out) < 255


def test_save_gif_frames_share_one_palette(tmp_path):
    """Every pixel of every frame takes its colour from one shared set of at most 255 colours.

    With ``optimize=True`` Pillow may store a frame's delta with a compacted local colour table,
    but the colours in it are entries of the shared palette, so colours cannot drift between
    frames. Per-frame adaptive palettes would give far more than 255 distinct colours in total.
    """
    path, _, _ = _render_gif_with_source(tmp_path / "a.gif")
    frames = _composited_frames(path)
    assert len(frames) > 5
    colours = np.unique(np.concatenate([f.reshape(-1, 3) for f in frames]), axis=0)
    assert len(colours) <= 255


def test_save_gif_static_region_unchanged(tmp_path):
    path, (x0, y0, x1, y1), _ = _render_gif_with_source(tmp_path / "a.gif")
    region = [f[y0:y1, x0:x1] for f in _composited_frames(path)]
    assert region[0].std() > 5  # the label is really there
    for r in region[1:]:
        np.testing.assert_array_equal(r, region[0])


def test_save_gif_last_frame_matches_source(tmp_path):
    path, _, last = _render_gif_with_source(tmp_path / "a.gif")
    decoded = _composited_frames(path)[-1].astype(int)
    assert decoded.shape == last.shape
    diff = np.abs(decoded - last.astype(int))
    assert diff.mean() < 2.0
    assert np.percentile(diff, 99) < 20
