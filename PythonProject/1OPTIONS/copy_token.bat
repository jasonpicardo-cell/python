@echo off
setlocal

set "LAST=6"
if not "%~1"=="" set "LAST=%~1"

set "SRC=\\192.168.1.%LAST%\jason\PycharmProjects\PythonProject\1OPTIONS\.zerodha_token"
set "DESTDIR=C:\Users\jjpic\git\python\PythonProject\1OPTIONS"
set "DEST=%DESTDIR%\.zerodha_token"

if not exist "%DESTDIR%" mkdir "%DESTDIR%"

echo Copying from %SRC%
type "%SRC%" > "%DEST%"
if errorlevel 1 (
    echo [ERROR] Copy failed.
    pause
    exit /b 1
)

echo [OK] Copied .zerodha_token to %DEST%
timeout /t 3 >nul