@echo off
chcp 65001 > nul

echo ====================================
echo   Dynamic Model AI - Setup & Execution
echo ====================================

echo Installing package in editable mode...
pip install -e .
if errorlevel 1 goto ERROR

echo.
echo Starting Dynamic Model AI CLI...
call dmai
if errorlevel 1 goto ERROR

echo.
echo ------------------------------------
echo Process completed successfully.
echo ------------------------------------
pause
exit /b 0

:ERROR
echo.
echo ====================================
echo [ERROR] An error occurred during execution.
echo ====================================
pause
exit /b 1