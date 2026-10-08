"""Backward-compatible Streamlit entry point; the application lives in app.py."""

# Execute app.py as the script on every Streamlit rerun. A plain `import app`
# would be cached in sys.modules and could skip the UI on subsequent reruns.
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
