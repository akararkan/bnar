FourierLab
==========

Educational Fourier series & FFT visualization (PySide6 + Matplotlib + NumPy).

Quick run (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Windows standalone executable (CI):

This repo contains a GitHub Actions workflow (.github/workflows/windows-build.yml) that will build a Windows "one-dir" bundle using PyInstaller and upload a ZIP artifact named `FourierLab-windows.zip`.

Notes on Windows Defender / AV false positives:
- Building with `--onedir` and `--noupx` reduces the chance of false positives. Avoid `--onefile` and UPX until you have signed the binary.
- To further reduce warnings: embed version info, avoid UPX, and sign the binary with an Authenticode certificate.
- If an AV product flags the binary, submit it to the vendor (e.g., Microsoft) as a false positive.

If you want me to create a signed installer or produce a code-signed .exe, provide a Windows signing certificate (or let me create a CI job that will sign using your secure secrets).

