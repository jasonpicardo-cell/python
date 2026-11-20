@echo off
setlocal

set "LAST=6"
if not "%~1"=="" set "LAST=%~1"

set "HOST=192.168.1.%LAST%"
set "SRC=\\%HOST%\jason\PycharmProjects\PythonProject\1OPTIONS\.zerodha_token"
set "DESTDIR=C:\Users\jjpic\git\python\PythonProject\1OPTIONS"
set "DEST=%DESTDIR%\.zerodha_token"

if not exist "%DESTDIR%" mkdir "%DESTDIR%"

REM Connect to the Mac share if not already connected
net use "\\%HOST%\jason" >nul 2>&1
if errorlevel 1 (
    echo Connecting to Mac share...
    net use "\\%HOST%\jason" /user:jason * /persist:yes
    if errorlevel 1 (
        echo [ERROR] Login to Mac failed.
        pause
        exit /b 1
    )
)

echo Copying from %SRC%
type "%SRC%" > "%DEST%"
if errorlevel 1 (
    echo [ERROR] Copy failed.
    del "%DEST%" >nul 2>&1
    pause
    exit /b 1
)

echo [OK] Copied .zerodha_token to %DEST%
timeout /t 3 >nul