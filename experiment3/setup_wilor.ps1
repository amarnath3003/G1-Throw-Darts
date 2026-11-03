$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv-wilor/Scripts/python.exe')) {
    python -m venv --system-site-packages .venv-wilor
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
if (-not (Test-Path 'vendor/WiLoR-mini')) {
    git clone https://github.com/warmshao/WiLoR-mini.git vendor/WiLoR-mini
    if ($LASTEXITCODE -ne 0) { throw 'clone failed' }
    git -C vendor/WiLoR-mini checkout ebec42f94c389070cdd7dda6fd1bf0b4a659c960
    if ($LASTEXITCODE -ne 0) { throw 'checkout failed' }
}
# Reuse the installed CUDA torch read-only; all added packages go into this venv.
& .venv-wilor/Scripts/python.exe -m pip install --no-deps smplx==0.1.28 einops==0.8.2 scikit-image==0.25.2 roma==1.8.0 chumpy==0.70 lazy-loader==0.6 tifffile==2025.5.10
if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }
& .venv-wilor/Scripts/python.exe -m pip install --no-deps -e vendor/WiLoR-mini
if ($LASTEXITCODE -ne 0) { throw 'WiLoR install failed' }
& .venv-wilor/Scripts/python.exe download_wilor.py
if ($LASTEXITCODE -ne 0) { throw 'asset download failed' }
