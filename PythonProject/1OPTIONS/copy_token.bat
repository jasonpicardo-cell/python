@echo off
setlocal

set "SRC=\\192.168.1.6\jason\PycharmProjects\PythonProject\1OPTIONS\.zerodha_token"
set "DEST=C:\Users\jjpic\git\python\PythonProject\1OPTIONS\.zerodha_token"

if not exist "C:\Users\jjpic\git\python\PythonProject\1OPTIONS" mkdir "C:\Users\jjpic\git\python\PythonProject\1OPTIONS"

type "%SRC%" > "%DEST%"
if errorlevel 1 (
    echo [ERROR] Copy failed.
    pause
    exit /b 1
)

echo [OK] Copied .zerodha_token to %DEST%
timeout /t 3 >nul