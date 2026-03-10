"""Backend helpers: fig to base64, etc."""

import base64
import io


def fig_to_b64(fig):
    """Convert matplotlib figure to base64 PNG."""
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
