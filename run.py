"""Run the desktop application from the repository root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vehicle_instance_search"))

if __name__ == "__main__":
    from app import InstanceSearchApp

    InstanceSearchApp().mainloop()
