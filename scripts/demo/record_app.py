"""Record screenshots of the running Gradio app for the demo GIF (uses Playwright).

Start the app first (python app.py), then:

    pip install playwright && playwright install chromium
    python scripts/demo/record_app.py --out captures

Uploads the sample models, renames the classes, picks the "Quick look" preset, clicks
Generate and screenshots until the gallery appears. Output: a*.png, b*.png, c0.png, clicks.json
"""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860/")
    ap.add_argument("--models", nargs="+", default=[str(ROOT / "examples/models" / f) for f in
                                                   ("pipe_flange.obj", "l_bracket.obj", "pipe_tee.obj", "samples.mtl")])
    ap.add_argument("--classes", nargs="+", default=["flange", "bracket", "tee"])
    ap.add_argument("--out", default="captures")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds to wait for the render")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clicks = {}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        pg.goto(args.url, wait_until="domcontentloaded")
        pg.wait_for_timeout(5000)
        pg.set_input_files("input[type=file] >> nth=0", args.models)
        pg.wait_for_timeout(5000)
        pg.screenshot(path=out / "a1.png")

        # rename classes: double-click the "class name" cell of each row
        head = pg.locator("text='class name' >> visible=true").first.bounding_box()
        for i, name in enumerate(args.classes[:3]):
            x, y = head["x"] + 60, head["y"] + head["height"] / 2 + 37 * (i + 1)
            pg.mouse.dblclick(x, y)
            pg.wait_for_timeout(400)
            pg.keyboard.press("Control+A")
            pg.keyboard.type(name, delay=60)
            pg.keyboard.press("Enter")
            pg.wait_for_timeout(500)
            clicks[f"a{2 + i}"] = [x, y]
            pg.screenshot(path=out / f"a{2 + i}.png")
        pg.mouse.click(640, 150)

        preset = pg.get_by_text("Quick look")
        box = preset.bounding_box()
        preset.click()
        pg.wait_for_timeout(1200)
        clicks["a5"] = [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2]
        pg.screenshot(path=out / "a5.png")

        top = pg.evaluate("() => document.querySelector('#generate-btn').getBoundingClientRect().top + window.scrollY")
        pg.evaluate(f"window.scrollTo(0, {top} - 70)")
        pg.wait_for_timeout(700)
        g = pg.locator("#generate-btn").bounding_box()
        clicks["b00"] = [g["x"] + g["width"] / 2, g["y"] + g["height"] / 2]
        pg.screenshot(path=out / "b00.png")
        pg.locator("#generate-btn").click()

        t0, i = time.time(), 1
        while time.time() - t0 < args.timeout:
            pg.screenshot(path=out / f"b{i:03d}.png")
            i += 1
            if pg.locator(".thumbnail-item").count() > 0:
                break
            pg.wait_for_timeout(max(250, int((time.time() - t0) * 10)))  # slow down on long renders
        pg.wait_for_timeout(2500)
        pg.screenshot(path=out / "c0.png")
        (out / "clicks.json").write_text(json.dumps(clicks))
        browser.close()
    print(f"{i} progress frames saved to {out}")


if __name__ == "__main__":
    main()
