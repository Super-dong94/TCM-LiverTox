@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_CMD=.venv\Scripts\python.exe"
if not exist "%PYTHON_CMD%" set "PYTHON_CMD=python"

set "SPEC_FILE=%CD%\build\toxherb_folder.spec"
set "BUILD_LOG=%CD%\build\toxherb_folder_build.log"
set "DIST_DIR=%CD%\dist"
set "WORK_DIR=%CD%\build\pyinstaller_folder"
set "APP_DIR=%DIST_DIR%\ToxHERB"
set "APP_EXE=%APP_DIR%\ToxHERB.exe"

if /i "%~1"=="--write-spec-only" (
    if not exist "build" mkdir "build"
    call :write_spec
    if errorlevel 1 exit /b 1
    echo [ToxHERB] Wrote folder spec:
    echo "%SPEC_FILE%"
    exit /b 0
)

echo.
echo [ToxHERB] Building folder Windows executable...
echo [ToxHERB] Project: %CD%
echo [ToxHERB] Python:
"%PYTHON_CMD%" --version
if errorlevel 1 goto :failed

echo.
echo [ToxHERB] Checking required project files...
for %%P in ("main.py" "backend" "frontend" "models" "data" "prediction_scripts") do (
    if not exist "%%~P" (
        echo [ToxHERB] Missing required path: %%~P
        goto :failed
    )
)
for %%P in ("frontend\Hepatotoxicity_Platform.html" "models\01best_model_tuned.joblib" "models\02best_model_tuned.joblib" "models\03best_model_tuned.joblib" "data\00_ctd_lookup.sqlite" "data\12_intoblood_ref.csv") do (
    if not exist "%%~P" (
        echo [ToxHERB] Missing required file: %%~P
        goto :failed
    )
)

echo.
echo [ToxHERB] Ensuring PyInstaller is installed...
"%PYTHON_CMD%" -m pip show pyinstaller >nul 2>nul
if errorlevel 1 (
    "%PYTHON_CMD%" -m pip install -U pyinstaller
    if errorlevel 1 goto :failed
)

if not exist "build" mkdir "build"
if not exist "dist" mkdir "dist"

if exist "%APP_DIR%" (
    echo.
    echo [ToxHERB] Removing previous output folder:
    echo [ToxHERB] %APP_DIR%
    rmdir /s /q "%APP_DIR%"
    if exist "%APP_DIR%" (
        echo [ToxHERB] Could not remove old output folder. Close any running ToxHERB.exe and try again.
        goto :failed
    )
)

echo.
echo [ToxHERB] Writing PyInstaller folder spec:
echo [ToxHERB] %SPEC_FILE%
call :write_spec
if errorlevel 1 goto :failed

echo.
echo [ToxHERB] Starting PyInstaller folder build. Large data files may take a long time.
echo [ToxHERB] Top 20 project files before packaging:
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath '%CD%' -Recurse -File | Sort-Object Length -Descending | Select-Object -First 20 @{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,2)}}, FullName | Format-Table -AutoSize"
echo [ToxHERB] Log: %BUILD_LOG%
"%PYTHON_CMD%" -m PyInstaller --clean --noconfirm --distpath "%DIST_DIR%" --workpath "%WORK_DIR%" "%SPEC_FILE%" > "%BUILD_LOG%" 2>&1
if errorlevel 1 (
    echo.
    echo [ToxHERB] Build failed. Last log lines:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%BUILD_LOG%' -Tail 100"
    goto :failed
)

if not exist "%APP_EXE%" (
    echo [ToxHERB] Build finished but ToxHERB.exe was not found.
    goto :failed
)

echo.
echo [ToxHERB] Build finished.
echo [ToxHERB] Executable:
echo "%APP_EXE%"
echo.
echo [ToxHERB] Copy the whole folder below when distributing the software:
echo "%APP_DIR%"
echo.
if defined TOXHERB_NO_PAUSE exit /b 0
pause
exit /b 0

:failed
echo.
echo [ToxHERB] Build failed. Please check:
echo "%BUILD_LOG%"
echo.
if defined TOXHERB_NO_PAUSE exit /b 1
pause
exit /b 1

:write_spec
> "%SPEC_FILE%" echo(# -*- mode: python ; coding: utf-8 -*-
>> "%SPEC_FILE%" echo(import importlib.util
>> "%SPEC_FILE%" echo(from pathlib import Path
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(root = Path(SPECPATH).resolve().parent
>> "%SPEC_FILE%" echo(datas = []
>> "%SPEC_FILE%" echo(hiddenimports = []
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(SKIP_PARTS = {".venv", "build", "dist", "__pycache__", ".pytest_cache", "tests", "toxicity_prediction_outputs", "prediction_outputs", "run_outputs", "job_logs", "api_validation_outputs_20260615_food_homology", "api_validation_outputs_20260615_food_homology_final", "batch_input_api_outputs"}
>> "%SPEC_FILE%" echo(SKIP_NAMES = {"00_ctd_lookup.tmp.sqlite", "jobs.sqlite"}
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(def add_tree(src, dest):
>> "%SPEC_FILE%" echo(    src_path = root / src
>> "%SPEC_FILE%" echo(    if not src_path.exists():
>> "%SPEC_FILE%" echo(        raise FileNotFoundError(src_path)
>> "%SPEC_FILE%" echo(    for path in src_path.rglob("*"):
>> "%SPEC_FILE%" echo(        if not path.is_file():
>> "%SPEC_FILE%" echo(            continue
>> "%SPEC_FILE%" echo(        if any(part in SKIP_PARTS for part in path.parts):
>> "%SPEC_FILE%" echo(            continue
>> "%SPEC_FILE%" echo(        if path.name in SKIP_NAMES or path.suffix in {".pyc", ".pid", ".log"}:
>> "%SPEC_FILE%" echo(            continue
>> "%SPEC_FILE%" echo(        rel = path.relative_to(src_path)
>> "%SPEC_FILE%" echo(        datas.append((str(path), str(Path(dest) / rel.parent)))
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(for folder in ("backend", "frontend", "models", "data", "prediction_scripts"):
>> "%SPEC_FILE%" echo(    add_tree(folder, folder)
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(BLOCKED_SUBMODULE_PARTS = (
>> "%SPEC_FILE%" echo(    ".tests",
>> "%SPEC_FILE%" echo(    ".testing",
>> "%SPEC_FILE%" echo(    ".conftest",
>> "%SPEC_FILE%" echo(    ".examples",
>> "%SPEC_FILE%" echo(    ".example",
>> "%SPEC_FILE%" echo(    ".benchmarks",
>> "%SPEC_FILE%" echo(    ".docs",
>> "%SPEC_FILE%" echo(    ".web",
>> "%SPEC_FILE%" echo(    ".graphgym",
>> "%SPEC_FILE%" echo(    ".dask",
>> "%SPEC_FILE%" echo(    ".sping.WX",
>> "%SPEC_FILE%" echo()
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(def keep_module(name):
>> "%SPEC_FILE%" echo(    return not any(part in name for part in BLOCKED_SUBMODULE_PARTS)
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(def add_package_sources(package):
>> "%SPEC_FILE%" echo(    package_spec = importlib.util.find_spec(package)
>> "%SPEC_FILE%" echo(    if package_spec is None or package_spec.submodule_search_locations is None:
>> "%SPEC_FILE%" echo(        return
>> "%SPEC_FILE%" echo(    dest_root = Path(package.replace(".", "/"))
>> "%SPEC_FILE%" echo(    skip_dirs = {"__pycache__", "test", "tests", "testing", "examples", "graphgym"}
>> "%SPEC_FILE%" echo(    for package_root in package_spec.submodule_search_locations:
>> "%SPEC_FILE%" echo(        package_path = Path(package_root)
>> "%SPEC_FILE%" echo(        for path in package_path.rglob("*.py"):
>> "%SPEC_FILE%" echo(            if any(part in skip_dirs for part in path.parts):
>> "%SPEC_FILE%" echo(                continue
>> "%SPEC_FILE%" echo(            rel = path.relative_to(package_path)
>> "%SPEC_FILE%" echo(            datas.append((str(path), str(dest_root / rel.parent)))
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(for package in (
>> "%SPEC_FILE%" echo(    "uvicorn",
>> "%SPEC_FILE%" echo(    "fastapi",
>> "%SPEC_FILE%" echo(    "pydantic",
>> "%SPEC_FILE%" echo(    "sklearn",
>> "%SPEC_FILE%" echo(    "lightgbm",
>> "%SPEC_FILE%" echo(    "xgboost",
>> "%SPEC_FILE%" echo(    "catboost",
>> "%SPEC_FILE%" echo(    "imblearn",
>> "%SPEC_FILE%" echo(    "rdkit",
>> "%SPEC_FILE%" echo(    "torch_geometric",
>> "%SPEC_FILE%" echo():
>> "%SPEC_FILE%" echo(    try:
>> "%SPEC_FILE%" echo(        hiddenimports += collect_submodules(package, filter=keep_module)
>> "%SPEC_FILE%" echo(    except TypeError:
>> "%SPEC_FILE%" echo(        hiddenimports += [name for name in collect_submodules(package) if keep_module(name)]
>> "%SPEC_FILE%" echo(    except Exception:
>> "%SPEC_FILE%" echo(        pass
>> "%SPEC_FILE%" echo(    try:
>> "%SPEC_FILE%" echo(        datas += collect_data_files(package)
>> "%SPEC_FILE%" echo(    except Exception:
>> "%SPEC_FILE%" echo(        pass
>> "%SPEC_FILE%" echo(    try:
>> "%SPEC_FILE%" echo(        datas += copy_metadata(package)
>> "%SPEC_FILE%" echo(    except Exception:
>> "%SPEC_FILE%" echo(        pass
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(# TorchScript needs the original torch_geometric .py files for some runtime-compiled classes.
>> "%SPEC_FILE%" echo(add_package_sources("torch_geometric")
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(hiddenimports += [
>> "%SPEC_FILE%" echo(    "uvicorn.lifespan.on",
>> "%SPEC_FILE%" echo(    "uvicorn.loops.auto",
>> "%SPEC_FILE%" echo(    "uvicorn.protocols.http.auto",
>> "%SPEC_FILE%" echo(    "uvicorn.protocols.websockets.auto",
>> "%SPEC_FILE%" echo(]
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(a = Analysis(
>> "%SPEC_FILE%" echo(    [str(root / "main.py")],
>> "%SPEC_FILE%" echo(    pathex=[str(root)],
>> "%SPEC_FILE%" echo(    binaries=[],
>> "%SPEC_FILE%" echo(    datas=datas,
>> "%SPEC_FILE%" echo(    hiddenimports=sorted(set(hiddenimports)),
>> "%SPEC_FILE%" echo(    hookspath=[],
>> "%SPEC_FILE%" echo(    hooksconfig={},
>> "%SPEC_FILE%" echo(    runtime_hooks=[],
>> "%SPEC_FILE%" echo(    excludes=[
>> "%SPEC_FILE%" echo(        "dask",
>> "%SPEC_FILE%" echo(        "flask",
>> "%SPEC_FILE%" echo(        "yacs",
>> "%SPEC_FILE%" echo(        "pidWX",
>> "%SPEC_FILE%" echo(        "pytest",
>> "%SPEC_FILE%" echo(        "IPython",
>> "%SPEC_FILE%" echo(    ],
>> "%SPEC_FILE%" echo(    noarchive=False,
>> "%SPEC_FILE%" echo()
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(pyz = PYZ(a.pure, a.zipped_data, cipher=None)
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(exe = EXE(
>> "%SPEC_FILE%" echo(    pyz,
>> "%SPEC_FILE%" echo(    a.scripts,
>> "%SPEC_FILE%" echo(    [],
>> "%SPEC_FILE%" echo(    exclude_binaries=True,
>> "%SPEC_FILE%" echo(    name="ToxHERB",
>> "%SPEC_FILE%" echo(    debug=False,
>> "%SPEC_FILE%" echo(    bootloader_ignore_signals=False,
>> "%SPEC_FILE%" echo(    strip=False,
>> "%SPEC_FILE%" echo(    upx=False,
>> "%SPEC_FILE%" echo(    upx_exclude=[],
>> "%SPEC_FILE%" echo(    runtime_tmpdir=None,
>> "%SPEC_FILE%" echo(    console=True,
>> "%SPEC_FILE%" echo(    disable_windowed_traceback=False,
>> "%SPEC_FILE%" echo(    argv_emulation=False,
>> "%SPEC_FILE%" echo(    target_arch=None,
>> "%SPEC_FILE%" echo(    codesign_identity=None,
>> "%SPEC_FILE%" echo(    entitlements_file=None,
>> "%SPEC_FILE%" echo()
>> "%SPEC_FILE%" echo(
>> "%SPEC_FILE%" echo(coll = COLLECT(
>> "%SPEC_FILE%" echo(    exe,
>> "%SPEC_FILE%" echo(    a.binaries,
>> "%SPEC_FILE%" echo(    a.zipfiles,
>> "%SPEC_FILE%" echo(    a.datas,
>> "%SPEC_FILE%" echo(    strip=False,
>> "%SPEC_FILE%" echo(    upx=False,
>> "%SPEC_FILE%" echo(    upx_exclude=[],
>> "%SPEC_FILE%" echo(    name="ToxHERB",
>> "%SPEC_FILE%" echo()
exit /b %errorlevel%
