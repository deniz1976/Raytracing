@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%"
vendor\bin\premake5.exe vs2022
set "SETUP_EXIT_CODE=%errorlevel%"
popd
exit /b %SETUP_EXIT_CODE%
