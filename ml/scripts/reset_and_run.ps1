$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Die($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

Info "STOPPING STACK"
docker compose down --remove-orphans --rmi local -v | Out-Host
if ($LASTEXITCODE -ne 0) { Die "docker compose down failed" }

Info "REMOVING LOCAL GENERATED FILES (HOST)"
if (Test-Path ".\ml\artifacts\model_registry\v0004") { Remove-Item -Recurse -Force ".\ml\artifacts\model_registry\v0004" }
if (Test-Path ".\ml\mlruns") { Remove-Item -Recurse -Force ".\ml\mlruns" }
if (Test-Path ".\ml\reports") { Remove-Item -Recurse -Force ".\ml\reports" }
if (Test-Path ".\ml\.pytest_cache") { Remove-Item -Recurse -Force ".\ml\.pytest_cache" }

if (Test-Path ".\tools\dataset\plantvillage_tomato\aug") { Remove-Item -Recurse -Force ".\tools\dataset\plantvillage_tomato\aug" }
New-Item -ItemType Directory -Force ".\tools\dataset\plantvillage_tomato\aug" | Out-Null

Info "BUILDING IMAGES (NO CACHE)"
docker compose build --no-cache --progress=plain | Out-Host
if ($LASTEXITCODE -ne 0) { Die "docker compose build failed" }

Info "SANITY CHECK: DATASET RAW EXISTS INSIDE CONTAINER"
$rawCount = docker compose run --rm -T garden-ml-dev bash -lc `
  "find dataset/plantvillage_tomato/raw -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l"
if ($LASTEXITCODE -ne 0) { Die "raw dataset count failed" }
$rawCount = [int]$rawCount.Trim()
Write-Host "raw images found: $rawCount"
if ($rawCount -le 0) { Die "No images found in dataset/plantvillage_tomato/raw (check volume mount ./tools/dataset -> /app/ml/dataset)" }

Info "RUNNING PYTEST"
docker compose run --rm -T garden-ml-dev bash -lc "pytest -vv -s tests --maxfail=1 --durations=20" | Out-Host
if ($LASTEXITCODE -ne 0) { Die "pytest failed" }

Info "AUGMENT DATASET"
docker compose run --rm -T garden-ml-dev bash -lc @"
garden-ml-augment \
  --input_dir dataset/plantvillage_tomato/raw \
  --output_dir dataset/plantvillage_tomato/aug \
  --img_size 128 \
  --aug_per_image 5 \
  --seed 42 \
  --segment_before_aug
"@ | Out-Host
if ($LASTEXITCODE -ne 0) { Die "augmentation failed" }

Info "CHECK AUG OUTPUT"
$manifestExists = docker compose run --rm -T garden-ml-dev bash -lc `
  "test -f dataset/plantvillage_tomato/aug/augmentation_manifest.csv && echo OK || echo NO"
if ($LASTEXITCODE -ne 0) { Die "manifest existence check failed" }
if ($manifestExists.Trim() -ne "OK") { Die "augmentation_manifest.csv not found in dataset/plantvillage_tomato/aug" }

$augFiles = docker compose run --rm -T garden-ml-dev bash -lc `
  "find dataset/plantvillage_tomato/aug -type f -iname '*__aug*.jpg' | wc -l"
if ($LASTEXITCODE -ne 0) { Die "aug files count failed" }
$augFiles = [int]$augFiles.Trim()
Write-Host "aug __aug files found: $augFiles"
if ($augFiles -le 0) { Die "No __aug files found in dataset/plantvillage_tomato/aug (augmentation did not generate aug samples)" }

$totalCount = docker compose run --rm -T garden-ml-dev bash -lc `
  "find dataset/plantvillage_tomato/aug -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l"
if ($LASTEXITCODE -ne 0) { Die "total files count failed" }
$totalCount = [int]$totalCount.Trim()
Write-Host "total images found in aug dir: $totalCount"
if ($totalCount -le 0) { Die "No images found in dataset/plantvillage_tomato/aug" }

Info "TRAIN MODEL"
docker compose run --rm -T garden-ml-dev bash -lc @"
garden-ml-train \
  --dataset_dir dataset/plantvillage_tomato/aug \
  --output_dir artifacts/model_registry/v0004 \
  --img_size 128 \
  --test_size 0.30 \
  --seed 42 \
  --cv_folds 5 \
  --manifest augmentation_manifest.csv
"@ | Out-Host
if ($LASTEXITCODE -ne 0) { Die "training failed" }

Info "CHECK TRAIN OUTPUT (MODEL FILE)"
$modelExists = docker compose run --rm -T garden-ml-dev bash -lc `
  "test -f artifacts/model_registry/v0004/modelo_tomate.pkl && echo OK || echo NO"
if ($LASTEXITCODE -ne 0) { Die "model existence check failed" }
if ($modelExists.Trim() -ne "OK") { Die "modelo_tomate.pkl not found in artifacts/model_registry/v0004 (train did not produce artifacts)" }

Info "EVALUATE MODEL"
docker compose run --rm -T garden-ml-dev bash -lc @"
garden-ml-eval \
  --dataset_dir dataset/plantvillage_tomato/aug \
  --artifacts_dir artifacts/model_registry/v0004 \
  --img_size 128 \
  --manifest augmentation_manifest.csv
"@ | Out-Host
if ($LASTEXITCODE -ne 0) { Die "evaluation failed" }

Info "STARTING API"
docker compose up -d garden-ml-api | Out-Host
if ($LASTEXITCODE -ne 0) { Die "docker compose up failed" }

Info "HEALTHCHECK"
Start-Sleep -Seconds 2
try {
  Invoke-RestMethod "http://localhost:5000/health" | ConvertTo-Json -Depth 5 | Write-Host
}
catch {
  Write-Host "API NOT READY YET (wait a bit and try: curl http://localhost:5000/health)" -ForegroundColor Yellow
}
