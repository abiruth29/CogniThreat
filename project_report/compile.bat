@echo off
REM CogniThreat IEEE Report Compilation Script
REM Windows Batch File for Easy PDF Generation

echo ================================================
echo CogniThreat IEEE Report Compiler
echo ================================================
echo.

REM Check if LaTeX is installed
where pdflatex >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pdflatex not found!
    echo.
    echo Please install MiKTeX or TeX Live:
    echo   - MiKTeX: https://miktex.org/download
    echo   - TeX Live: https://www.tug.org/texlive/
    echo.
    pause
    exit /b 1
)

echo [1/4] First compilation pass...
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
if %errorlevel% neq 0 (
    echo [ERROR] First compilation failed!
    echo Check CogniThreat_IEEE_Report.log for details
    pause
    exit /b 1
)

echo [2/4] Processing bibliography...
bibtex CogniThreat_IEEE_Report
if %errorlevel% neq 0 (
    echo [WARNING] Bibliography processing had issues
    echo This is normal if no citations changed
)

echo [3/4] Second compilation pass...
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
if %errorlevel% neq 0 (
    echo [ERROR] Second compilation failed!
    pause
    exit /b 1
)

echo [4/4] Final compilation pass...
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
if %errorlevel% neq 0 (
    echo [ERROR] Final compilation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo SUCCESS! PDF Generated Successfully
echo ================================================
echo.
echo Output: CogniThreat_IEEE_Report.pdf
echo.

REM Check if PDF exists and show size
if exist CogniThreat_IEEE_Report.pdf (
    for %%A in (CogniThreat_IEEE_Report.pdf) do (
        echo File size: %%~zA bytes
    )
    echo.
    
    REM Ask if user wants to open PDF
    set /p OPEN="Open PDF now? (Y/N): "
    if /i "%OPEN%"=="Y" (
        start CogniThreat_IEEE_Report.pdf
    )
) else (
    echo [ERROR] PDF file not found!
    echo Check CogniThreat_IEEE_Report.log for errors
)

echo.
echo Cleaning auxiliary files...
del /Q CogniThreat_IEEE_Report.aux 2>nul
del /Q CogniThreat_IEEE_Report.log 2>nul
del /Q CogniThreat_IEEE_Report.out 2>nul
del /Q CogniThreat_IEEE_Report.bbl 2>nul
del /Q CogniThreat_IEEE_Report.blg 2>nul

echo Done!
echo.
pause
