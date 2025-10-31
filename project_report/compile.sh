#!/bin/bash
# CogniThreat IEEE Report Compilation Script
# Linux/macOS Shell Script for Easy PDF Generation

echo "================================================"
echo "CogniThreat IEEE Report Compiler"
echo "================================================"
echo ""

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "[ERROR] pdflatex not found!"
    echo ""
    echo "Please install TeX Live:"
    echo "  - Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  - macOS: brew install --cask mactex"
    echo "  - Or visit: https://www.tug.org/texlive/"
    echo ""
    exit 1
fi

echo "[1/4] First compilation pass..."
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
if [ $? -ne 0 ]; then
    echo "[ERROR] First compilation failed!"
    echo "Check CogniThreat_IEEE_Report.log for details"
    exit 1
fi

echo "[2/4] Processing bibliography..."
bibtex CogniThreat_IEEE_Report
if [ $? -ne 0 ]; then
    echo "[WARNING] Bibliography processing had issues"
    echo "This is normal if no citations changed"
fi

echo "[3/4] Second compilation pass..."
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
if [ $? -ne 0 ]; then
    echo "[ERROR] Second compilation failed!"
    exit 1
fi

echo "[4/4] Final compilation pass..."
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
if [ $? -ne 0 ]; then
    echo "[ERROR] Final compilation failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "SUCCESS! PDF Generated Successfully"
echo "================================================"
echo ""
echo "Output: CogniThreat_IEEE_Report.pdf"
echo ""

# Check if PDF exists and show size
if [ -f "CogniThreat_IEEE_Report.pdf" ]; then
    SIZE=$(du -h CogniThreat_IEEE_Report.pdf | cut -f1)
    echo "File size: $SIZE"
    echo ""
    
    # Ask if user wants to open PDF
    read -p "Open PDF now? (y/n): " OPEN
    if [ "$OPEN" = "y" ] || [ "$OPEN" = "Y" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            open CogniThreat_IEEE_Report.pdf
        else
            # Linux
            xdg-open CogniThreat_IEEE_Report.pdf
        fi
    fi
else
    echo "[ERROR] PDF file not found!"
    echo "Check CogniThreat_IEEE_Report.log for errors"
    exit 1
fi

echo ""
echo "Cleaning auxiliary files..."
rm -f CogniThreat_IEEE_Report.aux
rm -f CogniThreat_IEEE_Report.log
rm -f CogniThreat_IEEE_Report.out
rm -f CogniThreat_IEEE_Report.bbl
rm -f CogniThreat_IEEE_Report.blg

echo "Done!"
echo ""
