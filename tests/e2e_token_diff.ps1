# E1 — End-to-end token-stream diff between SIMD-on and SIMD-off paths.
#
# Builds (or assumes existing) MLz.exe, runs the same prompt twice with
# greedy sampling (temperature 0, fixed seed), and asserts the two token
# streams are byte-identical.  This is the canonical "did our SIMD path
# silently diverge from ggml" gate.
#
# Usage:
#   powershell -File tests/e2e_token_diff.ps1 [-Model <path>] [-MaxTokens <n>]

[CmdletBinding()]
param(
    [string]$Model = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [int]$MaxTokens = 64,
    [string]$Prompt = "Once upon a time in a small village there lived",
    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "..\zig-out\bin\MLz.exe"
if (-not (Test-Path $exe)) {
    Write-Error "MLz.exe not found at $exe — run 'zig build -Dsimd-backend=true -Doptimize=ReleaseFast' first."
}
if (-not (Test-Path $Model)) {
    Write-Error "Model not found at $Model"
}

function Run-MLz([string]$tag, [string[]]$extra) {
    $argList = @(
        $Model,
        "--prompt", $Prompt,
        "--temp", "0",
        "--seed", "$Seed",
        "--ctx", "2048",
        "--stream", "false"
    ) + $extra
    Write-Host "[$tag] $exe $($argList -join ' ')"
    $stdoutFile = [IO.Path]::GetTempFileName()
    $stderrFile = [IO.Path]::GetTempFileName()
    $proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    $stdout = Get-Content $stdoutFile -Raw
    Remove-Item $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Write-Error "[$tag] MLz exited with $($proc.ExitCode)"
    }
    return ($stdout).Trim()
}

$simd_off = Run-MLz "simd-off" @("--no-simd")
$simd_on  = Run-MLz "simd-on"  @()

# Strip the engine banner (anything before the actual generation).  The runtime
# prints model metadata before the completion; we compare suffixes deterministic
# across both runs by hashing the full output.
$h_off = (Get-FileHash -InputStream ([System.IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($simd_off))) -Algorithm SHA256).Hash
$h_on  = (Get-FileHash -InputStream ([System.IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($simd_on)))  -Algorithm SHA256).Hash

Write-Host "simd-off SHA256 = $h_off"
Write-Host "simd-on  SHA256 = $h_on"

if ($h_off -ne $h_on) {
    Write-Host "---- DIFF ----"
    Compare-Object ($simd_off -split "`n") ($simd_on -split "`n") | Format-Table -AutoSize
    Write-Error "FAIL: token streams diverge between simd-off and simd-on builds"
}

Write-Host "PASS: simd-on and simd-off produced identical output (~$MaxTokens tokens)"

