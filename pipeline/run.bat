@echo off
cd /d "%~dp0"
echo ============================================================
echo  Revised leakage-safe benchmark (v2) - laptop edition
echo  Keep the laptop plugged in. Progress: outputs_v2\log.txt
echo  To stop cleanly: create an empty file called STOP in outputs_v2
echo  To resume: double-click run.bat again (finished units are skipped)
echo ============================================================
set PY=
py -3.12 --version >nul 2>nul && set PY=py -3.12
if not defined PY (py -3.11 --version >nul 2>nul && set PY=py -3.11)
if not defined PY (python --version >nul 2>nul && set PY=python)
if not defined PY (
  echo No working Python found. Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause & exit /b 1
)
if exist venv\Scripts\activate.bat (
  call venv\Scripts\activate.bat
  set PY=python
)
%PY% -c "import sklearn, imblearn, xgboost, shap, pandas, scipy" >nul 2>nul
if errorlevel 1 (
  echo Installing the required packages into a local environment. This takes a few minutes ^(log: setup_log.txt^).
  if not exist venv (%PY% -m venv venv)
  call venv\Scripts\activate.bat
  set PY=python
  python -m pip install --upgrade pip > setup_log.txt 2>&1
  pip install -r requirements.txt >> setup_log.txt 2>&1
  if errorlevel 1 (
    echo PACKAGE INSTALL FAILED. Send me the file setup_log.txt from this folder.
    pause & exit /b 1
  )
)
echo Using: %PY%
%PY% --version
echo Starting the run...
%PY% -u run_revised.py 1>> run_stdout.txt 2>> run_stderr.txt
echo.
echo Run finished or stopped. Building the analysis tables...
%PY% analyze_results.py 1>> analysis_stdout.txt 2>> analysis_stderr.txt
echo Running the leakage audit...
%PY% leakage_audit.py 1>> audit_stdout.txt 2>> audit_stderr.txt
echo Running the extra analyses (field ablations, calibration slope, decision curves, operating points)...
%PY% extra_analysis.py 1>> extra_stdout.txt 2>> extra_stderr.txt
echo Drawing the figures...
%PY% make_figures.py 1>> figures_stdout.txt 2>> figures_stderr.txt
echo Done. Tables are in outputs_v2\analysis, figures in outputs_v2\figures
pause
