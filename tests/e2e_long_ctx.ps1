# E2 — Long-context regression test.
#
# Per repo memory (mlz-simd-architecture.md): when llama.cpp upgraded to b8308
# the FLASH_ATTN_EXT work-buffer calculation changed (tiled FA, GGML_FA_TILE_*),
# and undersized buffers caused heap overflow → segfault deep into generation.
#
# This test generates 2048 tokens with --ctx 4096 and asserts the process
# exits cleanly (exit 0).  Run with both SIMD on and off.
#
# Usage:
#   powershell -File tests/e2e_long_ctx.ps1 [-Model <path>]

[CmdletBinding()]
param(
    [string]$Model = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [int]$Ctx = 4096,
    [int]$MaxTokens = 2048,
    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "..\zig-out\bin\MLz.exe"
if (-not (Test-Path $exe)) {
    Write-Error "MLz.exe not found at $exe"
}
if (-not (Test-Path $Model)) {
    Write-Error "Model not found at $Model"
}

# A long-form prompt that encourages the model to keep generating.
$Prompt = "Write a detailed multi-chapter fictional story about a robot exploring a vast alien library.  Take your time, develop the characters, describe the surroundings, and include dialogue."

function Run-Variant([string]$tag, [string[]]$extra) {
    $argList = @(
        $Model,
        "--prompt", $Prompt,
        "--temp", "0.8",
        "--seed", "$Seed",
        "--ctx", "$Ctx",
        "--n-predict", "$MaxTokens",
        "--stream", "false"
    ) + $extra
    Write-Host "[$tag] generating up to $MaxTokens tokens at ctx=$Ctx ..."
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $stdoutFile = [IO.Path]::GetTempFileName()
    $stderrFile = [IO.Path]::GetTempFileName()
    $proc = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    $sw.Stop()
    $stderr = Get-Content $stderrFile -Raw
    Remove-Item $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Write-Error "[$tag] FAIL: exit=$($proc.ExitCode) after $($sw.Elapsed.TotalSeconds.ToString('0.0'))s`nstderr: $stderr"
    }
    Write-Host "[$tag] PASS ($($sw.Elapsed.TotalSeconds.ToString('0.0'))s)"
}

Run-Variant "simd-off" @("--no-simd")
Run-Variant "simd-on"  @()
Run-Variant "simd-on+flash-attn" @("--simd-flash-attn")

Write-Host "PASS: long-context generation completed cleanly under all SIMD configurations"
