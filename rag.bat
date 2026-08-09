@echo off
REM =============================================================================
REM rag.bat — Windows shiv launcher.
REM First run: creates the venv + installs deps. After that: just runs cli.py.
REM
REM Usage:
REM     rag                                (no args -> "rag start": bring up
REM                                          the whole stack and open the GUI)
REM     rag ask "What was YaRN about?"
REM     rag ingest --markdown .\samples
REM     rag eval
REM     rag serve / start / open / status / tray / stats / etc.
REM
REM Drop a shortcut to this file on your taskbar / Start menu / desktop and
REM double-click -> the full rag stack comes up and the browser opens.
REM =============================================================================
setlocal
set "REPO=%~dp0"
set "VENV=%REPO%.venv"
set "PYEXE=%VENV%\Scripts\python.exe"

if not exist "%PYEXE%" (
    echo [rag] first run — creating venv and installing deps. This takes a few minutes.
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [rag] venv creation failed. Is Python 3.12 on PATH?
        exit /b 1
    )
    call "%VENV%\Scripts\activate.bat"
    pip install --upgrade pip >nul
    pip install -r "%REPO%requirements.txt"
    if errorlevel 1 (
        echo [rag] pip install failed. See output above.
        exit /b 1
    )
    echo [rag] setup complete.
)

call "%VENV%\Scripts\activate.bat" >nul
REM No args -> default to "start" so a desktop shortcut just works.
if "%*"=="" (
    python "%REPO%cli.py" start
) else (
    python "%REPO%cli.py" %*
)
endlocal
