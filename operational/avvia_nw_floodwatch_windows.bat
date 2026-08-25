@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ============================================================
  echo NW FloodWatch
  echo ERRORE: ambiente Python .venv non trovato.
  echo Eseguire prima setup_nw_floodwatch_windows.bat
  echo ============================================================
  pause
  exit /b 1
)

".venv\Scripts\python.exe" nw_flood_watch.py --open
set RC=%ERRORLEVEL%

echo.
if not "%RC%"=="0" (
  echo ============================================================
  echo NW FloodWatch terminato con errore %RC%.
  echo Consultare nw_floodwatch_output\NW_FloodWatch_pipeline_latest.log
  echo se presente.
  echo ============================================================
) else (
  echo ============================================================
  echo NW FloodWatch completato.
  echo Bollettino piu recente:
  echo nw_floodwatch_output\LATEST_NW_FloodWatch_Bollettino.pdf
  echo ============================================================
)
pause
exit /b %RC%
