# Pull every raw SCIAMA output that belongs in git into the local clone (Windows PowerShell).
# Copies only; never deletes anything on either side. Run from the repository root.
# macOS/Linux equivalent: sciama/sync_from_sciama.sh
#
# Nothing is lost on overwrite: if a pulled file would replace a local file with different
# content, the local version is first moved to results/sciama/archive/sync_<UTC time>/<same path>
# (tracked in git). Large artefacts (models, embeddings) are only added, never replaced.
$ErrorActionPreference = "Stop"
$Host_ = "sciama"
$RemoteRoot = $env:SCIAMA_REMOTE_ROOT
if (-not $RemoteRoot) { throw "set SCIAMA_REMOTE_ROOT to the repo checkout on SCIAMA" }
$Remote = "${Host_}:${RemoteRoot}"
$Stamp = "sync_" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$Archive = Join-Path (Get-Location) "results/sciama/archive/$Stamp"
$Staging = Join-Path ([System.IO.Path]::GetTempPath()) "sciama_sync_$Stamp"
$Dirs = @("results/test", "results/validation", "results/rationales", "results/revision", "results/sciama", "runs/slurm_logs")
$Archived = 0
foreach ($d in $Dirs) {
    $stageParent = Join-Path $Staging (Split-Path $d -Parent)
    New-Item -ItemType Directory -Force -Path $stageParent | Out-Null
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    scp -r "${Remote}/${d}" "$stageParent/"
    if ($LASTEXITCODE -ne 0) { Write-Warning "scp failed or nothing to copy for $d (exit $LASTEXITCODE)"; continue }
    $stageDir = Join-Path $Staging $d
    foreach ($f in Get-ChildItem -Path $stageDir -Recurse -File) {
        $rel = $f.FullName.Substring($stageDir.Length).TrimStart('\', '/')
        if ($rel -like "archive*") { continue }
        $dest = Join-Path $d $rel
        if (Test-Path $dest) {
            if ((Get-FileHash $dest).Hash -eq (Get-FileHash $f.FullName).Hash) { continue }
            $backup = Join-Path (Join-Path $Archive $d) $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
            Move-Item -Path $dest -Destination $backup
            $Archived++
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
        Copy-Item -Path $f.FullName -Destination $dest
    }
}
Remove-Item -Recurse -Force $Staging
if ($Archived -gt 0) { Write-Output "Archived $Archived replaced local files under $Archive (commit them with the sync)." }
# Large artefacts are NOT copied into git; back them up to $env:SCIAMA_BACKUP_DIR if set, else locally.
$BackupDir = @($env:SCIAMA_BACKUP_DIR) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
$BackupRoot = if ($BackupDir) { $BackupDir } else { ".sciama_backup/results" }
foreach ($d in @("models", "embeddings")) {
    $files = ssh $Host_ "cd $RemoteRoot/results/$d 2>/dev/null && find . -type f"
    foreach ($rel in $files) {
        $relClean = $rel.TrimStart('.', '/')
        $dest = Join-Path (Join-Path $BackupRoot $d) $relClean
        if (Test-Path $dest) { continue }  # never replace an existing backup copy
        New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
        scp "${Remote}/results/$d/$relClean" "$dest"
    }
}
Write-Output "Sync done. Review with: git status; git diff --stat"
