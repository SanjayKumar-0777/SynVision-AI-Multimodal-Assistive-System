@echo off
echo Starting Sign Language Recognition System...

set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

call .venv\Scripts\activate.bat

python -u app.py

echo.
echo Application stopped.
pause