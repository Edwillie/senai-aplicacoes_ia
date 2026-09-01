@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Ambiente virtual .venv nao encontrado.
    echo Execute primeiro:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Abrindo o painel Streamlit...
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py

pause
