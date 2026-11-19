# PowerShell wrapper: python -m SCOPE.exp_app <args> using the repository virtualenv.
#
#   .\scripts\scope_exp.ps1 status --campaign rehearsal
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
Push-Location $root
try {
    & $python -m SCOPE.exp_app @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}