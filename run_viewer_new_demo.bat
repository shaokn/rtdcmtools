@echo off
setlocal

rem ================================================================
rem EDIT THIS LINE: set the full path to your Python interpreter.
rem A virtual environment is recommended. Examples:
rem   C:\path\to\rtdcmtools\.venv\Scripts\python.exe
rem   C:\Python311\python.exe
rem Keep the quotation marks when the path contains spaces.
rem ================================================================
set "PYTHON_EXE=C:\EDIT_THIS_PATH\python.exe"

set "PORT=8768"

rem To let other computers on the same network connect, append --host 0.0.0.0
rem to the command at the bottom, then open http://<this machine's IP>:%PORT%/
rem there. The firewall must allow the port, and the viewer has no login.

if not exist "%PYTHON_EXE%" (
    echo Python was not found:
    echo   %PYTHON_EXE%
    echo.
    echo Edit PYTHON_EXE near the top of this BAT file, then run it again.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo Starting RT DICOM Viewer with an empty case library...
echo Open: http://127.0.0.1:%PORT%/?source=dicom
echo Press Ctrl+C in this window to stop the server.
echo.

"%PYTHON_EXE%" -u "%~dp0server_new.py" --port %PORT% --default-source dicom

echo.
echo The viewer has stopped.
pause
endlocal
