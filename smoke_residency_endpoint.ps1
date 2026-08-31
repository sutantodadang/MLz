# Smoke test for POST /v1/residency/completions
# Starts MLz with --residency-budget-mib, sends a request, verifies response,
# then stops the server.
param(
    [string]$ExePath = "zig-out/bin/mlz.exe",
    [string]$ModelPath = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$Url = "http://127.0.0.1:18080/v1/residency/completions",
    [int]$BudgetMib = 8,
    [int]$MaxTokens = 12
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) { throw "server exe not found: $ExePath" }
if (-not (Test-Path $ModelPath)) { throw "model not found: $ModelPath" }

$proc = Start-Process -FilePath $ExePath `
    -ArgumentList @(
        "--server", "--host", "127.0.0.1", "--port", "18080",
        "--model", $ModelPath,
        "--residency-budget-mib", "$BudgetMib"
    ) `
    -PassThru -NoNewWindow -RedirectStandardOutput "residency_srv_out.log" -RedirectStandardError "residency_srv_err.log"

try {
    # Wait for the port to accept connections.
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 500
        if ($proc.HasExited) { throw "server exited early with code $($proc.ExitCode)" }
        try {
            $tcp = New-Object Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", 18080)
            $tcp.Close()
            $ready = $true
            break
        } catch { }
    }
    if (-not $ready) { throw "server did not become ready" }

    $body = @{
        prompt     = "The capital of France is"
        max_tokens = $MaxTokens
        stream     = $false
    } | ConvertTo-Json

    $resp = Invoke-RestMethod -Uri $Url -Method Post -Body $body `
        -ContentType "application/json" -TimeoutSec 600

    Write-Host "=== non-stream response ==="
    Write-Host ($resp | ConvertTo-Json -Depth 5)

    if (-not $resp.choices -or $resp.choices.Count -lt 1) { throw "missing choices" }
    if (-not $resp.usage) { throw "missing usage" }
    if ($resp.residency.weight_budget_bytes -ne ($BudgetMib * 1048576)) { throw "wrong budget echo" }
    if ($resp.residency.peak_mapped_weight_bytes -gt $resp.residency.weight_budget_bytes) { throw "budget exceeded" }
    if ($resp.choices[0].text.Length -lt 1) { throw "empty completion" }
    Write-Host "PASS non-streaming"

    # Streaming smoke.
    $streamBody = @{
        prompt     = "Count: 1 2"
        max_tokens = 6
        stream     = $true
    } | ConvertTo-Json

    $req = [Net.HttpWebRequest]::Create($Url)
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Timeout = 600000
    $bytes = [Text.Encoding]::UTF8.GetBytes($streamBody)
    $rs = $req.GetRequestStream()
    $rs.Write($bytes, 0, $bytes.Length)
    $rs.Close()
    $httpResp = $req.GetResponse()
    $reader = New-Object IO.StreamReader($httpResp.GetResponseStream())
    $chunks = 0
    $sawDone = $false
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if ($line -eq "data: [DONE]") { $sawDone = $true; break }
        if ($line.StartsWith("data: ")) { $chunks++ }
    }
    $reader.Close()
    $httpResp.Close()
    Write-Host "=== stream chunks: $chunks, sawDone: $sawDone ==="
    if ($chunks -lt 1) { throw "no SSE chunks" }
    if (-not $sawDone) { throw "missing data: [DONE]" }
    Write-Host "PASS streaming"
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
        $proc.WaitForExit()
    }
}
