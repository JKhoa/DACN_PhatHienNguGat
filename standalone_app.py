"""
Root launcher for the Sleepy Detection app.
Delegates to yolo-sleepy-allinone-final/standalone_app.py so you can run:
    python standalone_app.py [args]
from the repository root.
"""
import os
import sys
import runpy

HERE = os.path.dirname(__file__)
TARGET = os.path.join(HERE, "yolo-sleepy-allinone-final", "standalone_app.py")

if __name__ == "__main__":
    if not os.path.exists(TARGET):
        sys.stderr.write("ERROR: Cannot find target app script at: %s\n" % TARGET)
        sys.exit(1)
    # Preserve original args after the script path
    sys.argv = [TARGET] + sys.argv[1:]
    runpy.run_path(TARGET, run_name="__main__")
