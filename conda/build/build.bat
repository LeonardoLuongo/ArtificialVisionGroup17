@echo off
echo Building Conda environment 'soccer_tracking'...
call conda env create -f conda\build\requirements.yaml
echo Build complete.
pause