# Smoke test for POST /v1/residency/completions
# Starts MLz with --residency-budget-mib, sends a request, verifies response,
# then stops the server.
param(
    [string]$ExePath = "zig-out/bin/mlz.exe",
    [string]$ModelPath = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$Url = "http://127.0.0.1:18080/v1/residency/completions",
    [int]$BudgetMib = 8,
    [int]$MaxTokens = 12,
    [int]$Slots = 1
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) { throw "server exe not found: $ExePath" }
if (-not (Test-Path $ModelPath)) { throw "model not found: $ModelPath" }

$proc = Start-Process -FilePath $ExePath `
    -ArgumentList @(
        "--server", "--host", "127.0.0.1", "--port", "18080",
        "--model", $ModelPath,
        "--residency-budget-mib", "$BudgetMib",
        "--residency-slots", "$Slots"
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

    # Chat messages smoke (jinja chat template through the residency service).
    $chatBody = @{
        messages   = @(
            @{ role = "user"; content = "The capital of France is" }
        )
        max_tokens = $MaxTokens
        stream     = $false
    } | ConvertTo-Json -Depth 4

    $chatResp = Invoke-RestMethod -Uri $Url -Method Post -Body $chatBody `
        -ContentType "application/json" -TimeoutSec 600

    Write-Host "=== chat response ==="
    Write-Host ($chatResp | ConvertTo-Json -Depth 5)

    if (-not $chatResp.choices -or $chatResp.choices.Count -lt 1) { throw "chat: missing choices" }
    if ($chatResp.choices[0].text.Length -lt 1) { throw "chat: empty completion" }
    if ($chatResp.residency.peak_mapped_weight_bytes -gt $chatResp.residency.weight_budget_bytes) { throw "chat: budget exceeded" }
    Write-Host "PASS chat messages"

    # Per-request budget override: smaller budget must still succeed and be
    # echoed back exactly.
    $overrideBody = @{
        prompt               = "The capital of France is"
        max_tokens           = 6
        residency_budget_mib = 4
    } | ConvertTo-Json

    $overrideResp = Invoke-RestMethod -Uri $Url -Method Post -Body $overrideBody `
        -ContentType "application/json" -TimeoutSec 600

    if ($overrideResp.residency.weight_budget_bytes -ne (4 * 1048576)) { throw "override: wrong budget echo" }
    if ($overrideResp.residency.peak_mapped_weight_bytes -gt $overrideResp.residency.weight_budget_bytes) { throw "override: budget exceeded" }
    if ($overrideResp.choices[0].text.Length -lt 1) { throw "override: empty completion" }
    Write-Host "PASS per-request budget override"

    # Invalid override must be rejected with 400.
    try {
        $badBody = @{ prompt = "x"; residency_budget_mib = 0 } | ConvertTo-Json
        Invoke-RestMethod -Uri $Url -Method Post -Body $badBody `
            -ContentType "application/json" -TimeoutSec 600 | Out-Null
        throw "invalid override was not rejected"
    } catch {
        $status = $null
        try { $status = [int]$_.Exception.Response.StatusCode } catch { }
        if ($status -ne 400) { throw "invalid override: expected 400, got $status" }
    }
    Write-Host "PASS invalid budget override rejected"

    # Concurrency: with slots=2, two requests must both succeed while the
    # server handles them in parallel (each in its own service slot).
    if ($Slots -ge 2) {
        $jobs = 1..2 | ForEach-Object {
            $b = @{ prompt = "The capital of France is"; max_tokens = 6 } | ConvertTo-Json
            Start-Job -ScriptBlock {
                param($u, $body)
                return Invoke-RestMethod -Uri $u -Method Post -Body $body `
                    -ContentType "application/json" -TimeoutSec 600
            } -ArgumentList $Url, $b
        }
        $results = $jobs | Receive-Job -Wait -AutoRemoveJob
        if ($results.Count -ne 2) { throw "concurrency: expected 2 responses, got $($results.Count)" }
        foreach ($r in $results) {
            if (-not $r.choices -or $r.choices[0].text.Length -lt 1) { throw "concurrency: empty completion" }
            if ($r.residency.peak_mapped_weight_bytes -gt $r.residency.weight_budget_bytes) { throw "concurrency: budget exceeded" }
        }
        Write-Host "PASS concurrent requests ($Slots slots)"
    }
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
        $proc.WaitForExit()
    }
}
