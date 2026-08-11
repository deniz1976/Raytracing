@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%PROJECT_ROOT%\WalnutApp.sln" (
    echo WalnutApp.sln was not found. Run scripts\Setup.bat first.
    exit /b 1
)

if not defined VULKAN_SDK (
    echo VULKAN_SDK is not set. Install the Vulkan SDK and restart this terminal.
    exit /b 1
)

if not exist "%VULKAN_SDK%\Bin\glslc.exe" (
    echo glslc.exe was not found under VULKAN_SDK\Bin.
    exit /b 1
)

if not exist "%VSWHERE%" (
    echo Visual Studio Installer could not be found.
    exit /b 1
)

set "MSBUILD_EXE="
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do (
    if not defined MSBUILD_EXE set "MSBUILD_EXE=%%I"
)

if not defined MSBUILD_EXE (
    echo MSBuild could not be found. Install Desktop development with C++.
    exit /b 1
)

"%MSBUILD_EXE%" "%PROJECT_ROOT%\WalnutApp.sln" /m /p:Configuration=Debug /p:Platform=x64
if errorlevel 1 exit /b %errorlevel%

pushd "%PROJECT_ROOT%\WalnutApp"
"%PROJECT_ROOT%\bin\Debug-windows-x86_64\WalnutApp\WalnutApp.exe"
set "APP_EXIT_CODE=%errorlevel%"
popd

exit /b %APP_EXIT_CODE%
