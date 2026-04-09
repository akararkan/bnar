FourierLab
==========

Educational Fourier series & FFT visualization (PySide6 + Matplotlib + NumPy).

Windows standalone executable (local build and CI)

This repository focuses on Windows distribution. There are two ways to obtain a Windows runnable bundle:

- CI build (recommended): a GitHub Actions workflow is included at `.github/workflows/windows-build.yml`. It runs on a Windows runner, builds a PyInstaller "one-dir" bundle (no UPX) and uploads a ZIP artifact named `FourierLab-windows.zip`.

  How to run the workflow:
  1. Push changes to a branch that matches `fix/**` or to `main`. (The workflow triggers on pushes to `main` and `fix/**`.)
  2. Open the Actions tab in GitHub, choose the "Build Windows executable" workflow and wait for the run to complete.
  3. Download the `FourierLab-windows.zip` artifact from the run; it contains `dist\FourierLab\FourierLab.exe` and supporting files. Double-click `FourierLab.exe` on a Windows 10/11 machine to run.

- Local Windows build (if you prefer to build locally):

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt pyinstaller
  pyinstaller --noconfirm --clean --noupx --windowed --onedir --name FourierLab main.py
  # The built app will be under dist\FourierLab\FourierLab.exe
  ```

Notes on Windows Defender / AV false positives:
- Use `--onedir` and `--noupx` (the workflow already does this). These options reduce the chance of AV heuristics triggering compared with `--onefile` and UPX packing.
- Embed version info and file metadata and, if possible, code-sign the executable and any installer with an Authenticode certificate — code signing is the most effective way to avoid warnings.
- If an AV product flags the binary, submit it to the vendor (for example, Microsoft Defender portal) as a false-positive.

If you want me to add code-signing steps to the CI (using your certificate stored in GitHub Secrets) or to create a single-file installer (Inno Setup / NSIS), tell me and I'll add a workflow for that.

