"""Entry point for hosting (Hugging Face Spaces, Render, etc.) and for a local run on port 7860.
The Flask app itself (all pages) lives in app/app.py."""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "app"))
from app import app  # noqa: E402


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
