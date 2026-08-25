@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo NW FloodWatch - setup Windows

echo Cartella: %CD%
echo ============================================================

set "PY_CMD="
where py >nul 2>&1
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.13"
  if not defined PY_CMD py -3.12 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.12"
  if not defined PY_CMD py -3 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3"
)
if not defined PY_CMD (
  where python >nul 2>&1 && set "PY_CMD=python"
)

if not defined PY_CMD (
  echo.
  echo ERRORE: Python non trovato.
  echo Installare Python 3.12 o 3.13 a 64 bit e ripetere il setup.
  echo Durante l'installazione selezionare "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

%PY_CMD% -c "import sys,struct; ok=(3,12) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P')*8==64; print('Python:', sys.version.split()[0], '-', struct.calcsize('P')*8, 'bit'); raise SystemExit(0 if ok else 2)"
if errorlevel 1 (
  echo.
  echo ERRORE: usare Python 3.12 o 3.13 a 64 bit.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Creazione ambiente virtuale .venv...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :fail
)

echo.
echo Aggiornamento pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installazione dipendenze...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Verifica pacchetto...
".venv\Scripts\python.exe" verifica_pacchetto_multiplatform.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo INSTALLAZIONE COMPLETATA

echo Per l'uso quotidiano eseguire:
echo   avvia_nw_floodwatch_windows.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo ============================================================
echo INSTALLAZIONE FALLITA

echo Leggere il messaggio di errore sopra.
echo ============================================================
pause
exit /b 1
