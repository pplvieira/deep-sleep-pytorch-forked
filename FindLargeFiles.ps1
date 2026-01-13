# Check if we are in a git repo
if (-not (Test-Path ".git")) {
    Write-Error "Current directory is not a git repository."
    exit
}

Write-Host "Scanning pack files for largest objects..." -ForegroundColor Cyan

# 1. Get all pack index files
$packFiles = Get-ChildItem ".git/objects/pack/pack-*.idx"

# 2. Run git verify-pack, capture output
# We execute the command and filter explicitly for lines starting with a SHA hash
$rawOutput = git verify-pack -v $packFiles.FullName | Where-Object { $_ -match "^[a-f0-9]{40}" }

# 3. Parse the output into objects, Sort by Size, Take top 10
$topObjects = $rawOutput | ForEach-Object {
    $parts = $_.Trim() -split "\s+"
    [PSCustomObject]@{
        SHA         = $parts[0]
        Type        = $parts[1]
        Size        = [int64]$parts[2]
        PackedSize  = [int64]$parts[3]
    }
} | Sort-Object Size -Descending | Select-Object -First 10

Write-Host "All sizes are in kB. Resolving file paths..." -ForegroundColor Cyan

$results = @()

# 4. Loop through the top 10 and find their file paths
foreach ($obj in $topObjects) {
    # Convert bytes to KB
    $sizeKB = [math]::Round($obj.Size / 1024, 0)
    $packedKB = [math]::Round($obj.PackedSize / 1024, 0)
    
    # Find the object name/location
    # This matches: other=`git rev-list --all --objects | grep $sha`
    $revList = git rev-list --all --objects | Select-String $obj.SHA -SimpleMatch | Select-Object -First 1
    
    $path = "Unknown/Tree/Blob"
    if ($revList) {
        # The output is "SHA <space> Path", so we skip the first 41 chars
        $line = $revList.ToString().Trim()
        if ($line.Length -gt 41) {
            $path = $line.Substring(41)
        }
    }

    $results += [PSCustomObject]@{
        "Size (KB)"   = $sizeKB
        "Packed (KB)" = $packedKB
        SHA           = $obj.SHA
        Location      = $path
    }
}

# 5. Output as a nice table
$results | Format-Table -AutoSize