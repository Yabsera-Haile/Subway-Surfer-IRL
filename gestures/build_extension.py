import io
import os
import shutil
import subprocess
import tarfile
import urllib.request

from pose import ensure_model

EXT = "extension"
TASKS_VISION = "1.0.1"
TARBALL = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{TASKS_VISION}.tgz"
VENDOR = {
    "package/vision_bundle.mjs": "vendor/mediapipe/vision_bundle.js",
    "package/wasm/vision_wasm_internal.js": "vendor/mediapipe/wasm/vision_wasm_internal.js",
    "package/wasm/vision_wasm_internal.wasm": "vendor/mediapipe/wasm/vision_wasm_internal.wasm",
}


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  {src} -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


def fetch_vendor():
    missing = [d for d in VENDOR.values() if not os.path.exists(os.path.join(EXT, d))]
    if not missing:
        print(f"  MediaPipe tasks-vision {TASKS_VISION} already present")
        return
    print(f"  downloading MediaPipe tasks-vision {TASKS_VISION} ...")
    with urllib.request.urlopen(TARBALL) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member, dst in VENDOR.items():
            out = os.path.join(EXT, dst)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with tar.extractfile(member) as f, open(out, "wb") as g:
                g.write(f.read())
            print(f"  {member} -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    if not os.path.exists("models/baseline.json"):
        raise SystemExit("models/baseline.json missing - run gestures/baseline.py first")
    print("models:")
    for name in ("baseline", "cnn", "gru"):
        src = f"models/{name}.json"
        if os.path.exists(src):
            copy(src, f"{EXT}/models/{name}.json")
        else:
            print(f"  {src} missing - its option in the panel won't load (train it with gestures/deep.py {name})")
    copy(ensure_model(), f"{EXT}/models/pose_landmarker_lite.task")
    print("vendor:")
    fetch_vendor()

    print("golden check (JavaScript pipeline vs Python):")
    if shutil.which("node") is None:
        print("  node not found - skipped. Install Node to verify the port.")
        return
    if not os.path.exists("tests/golden_baseline.json"):
        raise SystemExit("  tests/golden_baseline.json missing - run gestures/test_pipeline.py --export")
    res = subprocess.run(["node", f"{EXT}/test/golden.test.js"], capture_output=True, text=True)
    print("  " + res.stdout.strip().replace("\n", "\n  "))
    if res.returncode:
        raise SystemExit("the extension's pipeline no longer matches Python - don't ship this build")
    print(f"\nready: load {os.path.abspath(EXT)} at chrome://extensions (Developer mode -> Load unpacked)")


if __name__ == "__main__":
    main()
