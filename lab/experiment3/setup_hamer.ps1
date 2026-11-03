$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv-hamer/Scripts/python.exe')) {
    python -m venv --system-site-packages .venv-hamer
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
if (-not (Test-Path 'vendor/hamer')) {
    git clone https://github.com/geopavlakos/hamer.git vendor/hamer
    if ($LASTEXITCODE -ne 0) { throw 'clone failed' }
    git -C vendor/hamer checkout 3a01849f4148352e9260b69bf28b65d1671a4905
    if ($LASTEXITCODE -ne 0) { throw 'checkout failed' }
}
# Crop boxes come from the saved detector, so Detectron2/ViTPose are unnecessary.
& .venv-hamer/Scripts/python.exe -m pip install --no-deps pytorch-lightning==2.2.4 lightning-utilities==0.15.3 torchmetrics==1.4.0 yacs==0.1.8 pyrender==0.1.45 omegaconf==2.4.0 hydra-core==1.3.8 webdataset==1.0.2 smplx==0.1.28 einops==0.8.2 scikit-image==0.25.2 roma==1.8.0 chumpy==0.70 lazy-loader==0.6 tifffile==2025.5.10 freetype-py==2.5.1 pyglet==1.5.31 PyOpenGL==3.1.0 antlr4-python3-runtime==4.9.3 pretty-errors==1.2.25 braceexpand==0.1.7 termcolor==3.3.0
if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }
& .venv-hamer/Scripts/python.exe -m pip install --no-deps -e vendor/hamer
if ($LASTEXITCODE -ne 0) { throw 'HaMeR install failed' }
& .venv-hamer/Scripts/python.exe download_hamer.py
if ($LASTEXITCODE -ne 0) { throw 'asset download failed' }
