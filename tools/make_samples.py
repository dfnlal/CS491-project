"""Write the synthetic sample photos to samples/ (python tools/make_samples.py)."""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from threesweep import synthetic  # noqa: E402


def main():
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    for name, maker in (
        ("cylinder", synthetic.cylinder),
        ("cone", synthetic.cone),
        ("bent_pipe", synthetic.bent_pipe),
        ("box", synthetic.box),
    ):
        img = maker()[0]
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(out / f"synthetic_{name}.png"))
        print("wrote", out / f"synthetic_{name}.png")


if __name__ == "__main__":
    main()
