"""Injects outputs/dashboard_data.json into dashboard_template.html -> NBA_Game_Predictor.html"""
import json
from pathlib import Path

root = Path(__file__).parent
data = json.loads((root / "outputs" / "dashboard_data.json").read_text(encoding="utf-8"))
html = (root / "dashboard_template.html").read_text(encoding="utf-8")
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
out = root / "NBA_Game_Predictor.html"
out.write_text(html.replace("__DATA__", payload), encoding="utf-8", newline="\n")  # same line endings on every OS
print("Wrote", out, f"({out.stat().st_size / 1024:.0f} KB)")
