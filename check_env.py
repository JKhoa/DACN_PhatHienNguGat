import importlib

REQUIRED = ["cv2", "mediapipe", "numpy", "scipy"]


def main():
    missing = []
    for module in REQUIRED:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)

    if missing:
        print("Thieu package:", ", ".join(missing))
        raise SystemExit(1)

    print("Moi package da san sang.")


if __name__ == "__main__":
    main()
