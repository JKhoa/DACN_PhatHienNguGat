# Integrated Desktop App Files

This repository has been reorganized to keep the runtime desktop app focused.

## Runtime Entry Points (kept at repo root)
- `start_desktop_ui.py` - starts Vite desktop UI
- `start_python_backend.py` - starts Flask+SocketIO backend (`python-backend/server_with_tracking_backup.py`)
- `standalone_app.py` - launches `yolo-sleepy-allinone-final/gui_app.py`
- `check_env.py` - optional environment check helper

## Consolidated Desktop Backend (single source of truth)
Folder:
- `Desktop UI for Drowsiness Detection/python-backend/`

Core files:
- `server_with_tracking_backup.py`
- `yolo_detector.py`
- `drowsiness_logger.py`
- `db_helper.py`

Additional backend/support files:
- `server.py`, `server_enhanced.py`, `server_backup.py`, `server_simple.py`, `server_simple_backup.py`
- `report_generator.py`, `seed_sample_data.py`, `debug_server.py`, `check_model_info.py`, `quick_test_logging.py`

## Archived Non-App Python Files
Archived folder:
- `ARCHIVE_NON_APP_PY/`

Contains:
- duplicate files (`*_0.py`, `*_1.py`, ...)
- root legacy backend duplicates
- training/data/test utility scripts from repo root
- vendor-like unrelated Python files that were copied into root

## Notes
- Runtime now uses the Desktop UI backend folder as the main backend code location.
- If you need any archived scripts later (training/testing), restore from `ARCHIVE_NON_APP_PY`.
