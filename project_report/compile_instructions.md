# LaTeX Compilation Instructions for CogniThreat IEEE Report

## 📋 Prerequisites

### Required Software

#### Windows
**Option 1: MiKTeX (Recommended)**
1. Download from: https://miktex.org/download
2. Install with default settings
3. MiKTeX will auto-install missing packages

**Option 2: TeX Live**
1. Download from: https://www.tug.org/texlive/
2. Full installation: ~4GB
3. Includes all packages

#### Linux (Ubuntu/Debian)
```bash
# Full installation (recommended)
sudo apt-get update
sudo apt-get install texlive-full

# Minimal installation
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-publishers
```

#### macOS
```bash
# Using Homebrew
brew install --cask mactex

# Or download from: https://www.tug.org/mactex/
```

### Required LaTeX Packages
The report uses these packages (auto-installed by MiKTeX):
- `IEEEtran` - IEEE conference paper format
- `cite` - Citation management
- `amsmath`, `amssymb`, `amsfonts` - Mathematical symbols
- `graphicx` - Figure inclusion
- `hyperref` - Clickable links
- `booktabs` - Professional tables
- `listings` - Code formatting

---

## 🔨 Compilation Methods

### Method 1: Command Line (Universal)

#### Step 1: Navigate to Report Directory
```bash
cd /d/CogniThreat/project_report
```

#### Step 2: Compile with pdfLaTeX
```bash
# First pass - generates auxiliary files
pdflatex CogniThreat_IEEE_Report.tex

# Process bibliography (if needed)
bibtex CogniThreat_IEEE_Report

# Second pass - resolves references
pdflatex CogniThreat_IEEE_Report.tex

# Third pass - finalizes cross-references
pdflatex CogniThreat_IEEE_Report.tex
```

#### Expected Output
```
Output written on CogniThreat_IEEE_Report.pdf (8 pages, 245678 bytes).
Transcript written on CogniThreat_IEEE_Report.log.
```

#### One-Line Command
```bash
pdflatex CogniThreat_IEEE_Report.tex && pdflatex CogniThreat_IEEE_Report.tex
```

---

### Method 2: Overleaf (Online - No Installation)

#### Step 1: Create Account
1. Go to https://www.overleaf.com
2. Sign up (free account sufficient)

#### Step 2: Upload Project
1. Click "New Project" → "Upload Project"
2. Create a ZIP file:
   ```bash
   cd /d/CogniThreat/project_report
   zip CogniThreat_Report.zip CogniThreat_IEEE_Report.tex
   ```
3. Upload the ZIP file

#### Step 3: Configure Compiler
1. Click "Menu" (top-left)
2. Set "Compiler" to: **pdfLaTeX**
3. Set "TeX Live version" to: Latest (2023 or 2024)

#### Step 4: Compile
1. Click "Recompile" button
2. PDF appears in right panel
3. Download using "Download PDF" button

**Advantages**: No local installation, cloud-based, collaborative editing

---

### Method 3: VS Code with LaTeX Workshop

#### Step 1: Install VS Code
Download from: https://code.visualstudio.com/

#### Step 2: Install Extensions
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search and install: **LaTeX Workshop**

#### Step 3: Open Report
```bash
cd /d/CogniThreat/project_report
code CogniThreat_IEEE_Report.tex
```

#### Step 4: Configure (if needed)
Create `.vscode/settings.json` in project_report folder:
```json
{
    "latex-workshop.latex.recipes": [
        {
            "name": "pdflatex x2",
            "tools": ["pdflatex", "pdflatex"]
        }
    ],
    "latex-workshop.latex.tools": [
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ]
        }
    ]
}
```

#### Step 5: Compile
- **Method A**: Save file (Ctrl+S) - auto-compiles
- **Method B**: Right-click → "Build LaTeX project"
- **Method C**: Press Ctrl+Alt+B

#### Step 6: View PDF
- PDF appears in side panel automatically
- Or click "View LaTeX PDF" icon

**Advantages**: Integrated editing, auto-compile, syntax highlighting, error navigation

---

### Method 4: TeXstudio (Dedicated LaTeX Editor)

#### Step 1: Install TeXstudio
Download from: https://www.texstudio.org/

#### Step 2: Open Report
1. Launch TeXstudio
2. File → Open → Select `CogniThreat_IEEE_Report.tex`

#### Step 3: Configure Compiler
1. Options → Configure TeXstudio
2. Build → Default Compiler: **pdfLaTeX**

#### Step 4: Compile
- **Method A**: Press F5 (Build & View)
- **Method B**: Tools → Build → Build (F1)
- **Method C**: Click green arrow icon

#### Step 5: View PDF
- Built-in PDF viewer opens automatically
- Or Tools → View → View PDF (F7)

**Advantages**: Beginner-friendly GUI, built-in PDF viewer, spell-check

---

## 🔍 Verification Steps

### 1. Check PDF Generated
```bash
ls -lh CogniThreat_IEEE_Report.pdf
# Expected: ~200-300 KB file
```

### 2. Verify Page Count
```bash
# Linux/macOS
pdfinfo CogniThreat_IEEE_Report.pdf | grep Pages
# Expected: Pages: 8

# Windows (using PowerShell)
(Get-Content CogniThreat_IEEE_Report.pdf | Select-String "/Count").Count
```

### 3. Open PDF
```bash
# Windows
start CogniThreat_IEEE_Report.pdf

# Linux
xdg-open CogniThreat_IEEE_Report.pdf

# macOS
open CogniThreat_IEEE_Report.pdf
```

### 4. Visual Inspection Checklist
- ✅ Title displays correctly
- ✅ Author information present
- ✅ Two-column format
- ✅ All sections numbered
- ✅ Tables render properly
- ✅ Equations formatted correctly
- ✅ References section complete
- ✅ No missing images placeholders

---

## 🐛 Troubleshooting

### Issue 1: "IEEEtran.cls not found"

**Solution A - MiKTeX (Auto-install)**
```bash
# MiKTeX will prompt to install
# Click "Yes" to install missing package
```

**Solution B - Manual Install**
```bash
# Download IEEEtran
wget http://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip
unzip IEEEtran.zip
# Copy .cls file to same directory as .tex file
cp IEEEtran/IEEEtran.cls /d/CogniThreat/project_report/
```

**Solution C - TeX Live Package Manager**
```bash
tlmgr install IEEEtran
```

### Issue 2: "File not found" Errors

**Check File Location**
```bash
pwd  # Should be in project_report directory
ls CogniThreat_IEEE_Report.tex  # File should exist
```

**Fix Path Issues**
```bash
cd /d/CogniThreat/project_report
pdflatex CogniThreat_IEEE_Report.tex
```

### Issue 3: Bibliography Not Appearing

**Full Compilation Sequence**
```bash
pdflatex CogniThreat_IEEE_Report.tex
bibtex CogniThreat_IEEE_Report
pdflatex CogniThreat_IEEE_Report.tex
pdflatex CogniThreat_IEEE_Report.tex
```

**Check .bbl File Generated**
```bash
ls CogniThreat_IEEE_Report.bbl
# If missing, bibtex step failed
```

### Issue 4: Overfull/Underfull Box Warnings

**Common Causes**
- Long URLs without line breaks
- Wide tables exceeding column width
- Long words without hyphenation

**Solutions**
```latex
% For URLs - add this to preamble
\usepackage{url}
\def\UrlBreaks{\do\/\do-}

% For paragraphs - use before problematic text
\sloppy
Your text here...
\fussy

% For tables - reduce font size
\begin{table}[htbp]
\small  % or \footnotesize
...
\end{table}
```

### Issue 5: Compilation Hangs

**Interactive Mode Fix**
```bash
# Use non-interactive mode
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex
```

**Check for Infinite Loops**
- Look for circular references
- Check for recursive includes
- Review recent edits

### Issue 6: Permission Denied

**Windows**
```bash
# Close all PDF viewers
taskkill /F /IM AcroRd32.exe  # Adobe Reader
taskkill /F /IM SumatraPDF.exe  # SumatraPDF
# Then recompile
```

**Linux/macOS**
```bash
# Check file permissions
ls -l CogniThreat_IEEE_Report.pdf
chmod 644 CogniThreat_IEEE_Report.pdf
```

---

## 📊 Compilation Statistics

### Expected Timings
- **First compilation**: 10-30 seconds
- **Subsequent compilations**: 5-15 seconds
- **With bibliography**: +5-10 seconds

### File Sizes
- **Source (.tex)**: ~25 KB
- **PDF output**: 200-300 KB (no figures)
- **Auxiliary files**: ~50-100 KB total

### Generated Files
```
CogniThreat_IEEE_Report.tex     # Source file (keep)
CogniThreat_IEEE_Report.pdf     # Output PDF (keep)
CogniThreat_IEEE_Report.aux     # Auxiliary (can delete)
CogniThreat_IEEE_Report.log     # Compilation log (can delete)
CogniThreat_IEEE_Report.bbl     # Bibliography (can delete)
CogniThreat_IEEE_Report.blg     # Bib log (can delete)
CogniThreat_IEEE_Report.out     # Hyperref (can delete)
```

### Clean Auxiliary Files
```bash
# Keep only .tex and .pdf
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
```

---

## 🚀 Quick Reference

### Compilation Commands Cheat Sheet

```bash
# Basic compilation
pdflatex CogniThreat_IEEE_Report.tex

# With bibliography
pdflatex CogniThreat_IEEE_Report.tex && \
bibtex CogniThreat_IEEE_Report && \
pdflatex CogniThreat_IEEE_Report.tex && \
pdflatex CogniThreat_IEEE_Report.tex

# Non-interactive mode
pdflatex -interaction=nonstopmode CogniThreat_IEEE_Report.tex

# Draft mode (faster)
pdflatex -draftmode CogniThreat_IEEE_Report.tex

# Clean and compile
rm -f *.aux *.log *.out && pdflatex CogniThreat_IEEE_Report.tex
```

### Editor Shortcuts

**VS Code + LaTeX Workshop**
- Build: `Ctrl+Alt+B`
- View PDF: `Ctrl+Alt+V`
- Clean: `Ctrl+Alt+C`

**TeXstudio**
- Build & View: `F5`
- Build: `F1`
- View PDF: `F7`

**Overleaf**
- Recompile: `Ctrl+Enter`
- Download PDF: `Ctrl+Shift+D`

---

## 📝 Customization After Compilation

### Adding Your Own Figures

1. **Generate figures** from your experiments
2. **Save as PNG or PDF** (PDF preferred for vector graphics)
3. **Place in project_report folder**
4. **Add to LaTeX**:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{confusion_matrix.pdf}
\caption{Confusion matrix for QCNN-QLSTM model on CIC-IDS-2017 test set.}
\label{fig:confusion}
\end{figure}
```

5. **Reference in text**: `Figure \ref{fig:confusion} shows...`
6. **Recompile** (twice to update references)

### Updating Results Tables

Simply replace the numbers in the table environments:

```latex
\begin{table}[htbp]
\caption{Your Updated Results}
\begin{center}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Your Model} & \textbf{Baseline} \\
\midrule
Accuracy & 97.2\% & 94.8\% \\  % Update these
Precision & 96.1\% & 93.4\% \\  % Update these
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

---

## ✅ Final Checklist

Before submitting your compiled PDF:

- [ ] PDF compiles without errors
- [ ] All pages render correctly (8 pages expected)
- [ ] Author information updated
- [ ] All tables show your real data
- [ ] References formatted properly
- [ ] No "??" in citations or references
- [ ] Figures display correctly (if added)
- [ ] Two-column format maintained
- [ ] Professional appearance

---

## 🆘 Getting Help

### Error Messages
1. **Check the .log file**: Contains detailed error information
2. **Search the error**: Google "latex [error message]"
3. **TeX Stack Exchange**: https://tex.stackexchange.com/

### Resources
- **Overleaf Documentation**: https://www.overleaf.com/learn
- **LaTeX Wikibook**: https://en.wikibooks.org/wiki/LaTeX
- **IEEE Template Guide**: https://www.ieee.org/conferences/publishing/templates.html

---

**You're now ready to compile your IEEE-standard project report!**

**Quick Start**: `pdflatex CogniThreat_IEEE_Report.tex && pdflatex CogniThreat_IEEE_Report.tex`
