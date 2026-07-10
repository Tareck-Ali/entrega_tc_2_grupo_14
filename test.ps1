# Docker must be running for it to work

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

try {
    Write-Host "Starting Docker services..."
    docker compose up
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Installing dependencies..."
    poetry install
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Running Ruff..."
    poetry run ruff
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Running DVC pipeline..."
    dvc repro
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "All tests completed successfully."
}
finally {
    Write-Host "Stopping Docker services..."
    docker compose logs -f
    docker compose down
}