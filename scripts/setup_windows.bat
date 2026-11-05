@echo off
setlocal
cd /d "%~dp0\.."
set "PIP_CACHE_DIR=%CD%\.pip-cache"
echo AuthenticVision local GUI setup
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 x64 first.
  pause
  exit /b 1
)
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
if errorlevel 1 (
  echo CPU PyTorch installation failed. Read the error above.
  pause
  exit /b 1
)
python -m pip install -r requirements-local.txt
if errorlevel 1 (
  echo Setup failed. Read the error above; do not delete your model checkpoint.
  pause
  exit /b 1
)
if not exist outputs mkdir outputs
echo.
echo Setup complete. Copy your cloud-trained best_model.pt into the outputs folder.
echo Then run scripts\run_app_windows.bat
pause
