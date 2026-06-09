# Run all remaining Qwen2-VL experiments overnight with auto-retry.
# Usage: Open PowerShell, cd to project root, run:
#   .\scripts\run_overnight.ps1

$maxRetries = 3
$experiments = @(
    @{ subset = "digits";       expected = 300  },
    @{ subset = "hate_slangs";  expected = 690  },
    @{ subset = "hate_symbols"; expected = 1170 }
)

Set-Location "C:\Users\manig\Downloads\content-moderation"

function Test-ResultComplete($subset, $expectedSamples) {
    $file = "results\qwen2vl_$subset.json"
    if (-not (Test-Path $file)) { return $false }
    try {
        $data = Get-Content $file -Raw | ConvertFrom-Json
        # Must have 13 filter entries with correct sample count
        if ($data.Count -ne 13) { return $false }
        if ($data[0].sample_predictions.Count -ne $expectedSamples) { return $false }
        return $true
    } catch {
        return $false
    }
}

Write-Host "=== Overnight Qwen2-VL Benchmark Runner ===" -ForegroundColor Cyan
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Experiments: $($experiments.Count) subsets, max $maxRetries retries each"
Write-Host ""

foreach ($exp in $experiments) {
    $subset = $exp.subset
    $expected = $exp.expected

    # Skip if already complete
    if (Test-ResultComplete $subset $expected) {
        Write-Host "[SKIP] qwen2vl_$subset already complete" -ForegroundColor Green
        continue
    }

    $success = $false
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        Write-Host ""
        Write-Host "[RUN] qwen2vl $subset (attempt $attempt/$maxRetries) - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow

        uv run python scripts/benchmark_qwen2vl.py --subset $subset --device cuda

        if (Test-ResultComplete $subset $expected) {
            Write-Host "[DONE] qwen2vl_$subset complete" -ForegroundColor Green
            $success = $true
            break
        } else {
            Write-Host "[FAIL] qwen2vl_$subset crashed or incomplete, retrying in 10s..." -ForegroundColor Red
            Start-Sleep -Seconds 10
        }
    }

    if (-not $success) {
        Write-Host "[ERROR] qwen2vl_$subset failed after $maxRetries attempts" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== All done: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
Write-Host "Check results\qwen2vl_*.json for outputs."
