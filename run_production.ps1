# PowerShell Production Script for Synthetic Data Generator

param(
    [Parameter(Mandatory=$true)]
    [string]$Command,
    
    [string]$InputFile = "",
    [int]$NumRows = 1000,
    [string]$ConfigFile = "config/config_template.yaml",
    [string]$OutputDir = "output",
    [string]$Model = "granite4:latest"
)

function Show-Help {
    Write-Host "Synthetic Data Generator - PowerShell Production Script" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run_production.ps1 -Command <command> [options]"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Green
    Write-Host "  install     - Install dependencies"
    Write-Host "  setup       - Setup Ollama and models"
    Write-Host "  generate    - Generate synthetic data"
    Write-Host "  batch       - Batch process multiple files"
    Write-Host "  validate    - Validate data quality"
    Write-Host "  clean       - Clean output files"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run_production.ps1 -Command generate -InputFile 'data.csv' -NumRows 5000"
    Write-Host "  .\run_production.ps1 -Command batch -InputFile 'data_folder\' -NumRows 1000"
    Write-Host "  .\run_production.ps1 -Command validate -InputFile 'original.csv'"
    Write-Host "  .\run_production.ps1 -Command setup"
}

function Install-Dependencies {
    Write-Host "📦 Installing production dependencies..." -ForegroundColor Blue
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

function Setup-Ollama {
    Write-Host "🤖 Setting up Ollama and models..." -ForegroundColor Blue
    python setup_ollama.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ollama setup completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Ollama setup failed" -ForegroundColor Red
        exit 1
    }
}

function Generate-SyntheticData {
    if (-not $InputFile) {
        Write-Host "❌ Input file required for generate command" -ForegroundColor Red
        Write-Host "Usage: .\run_production.ps1 -Command generate -InputFile 'your_data.csv'" -ForegroundColor Yellow
        exit 1
    }
    
    if (-not (Test-Path $InputFile)) {
        Write-Host "❌ Input file not found: $InputFile" -ForegroundColor Red
        exit 1
    }
    
    # Create output directory if it doesn't exist
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }
    
    $OutputFile = Join-Path $OutputDir "synthetic_$(Split-Path $InputFile -Leaf)"
    
    Write-Host "🎯 Generating synthetic data..." -ForegroundColor Blue
    Write-Host "  Input: $InputFile" -ForegroundColor Cyan
    Write-Host "  Output: $OutputFile" -ForegroundColor Cyan
    Write-Host "  Rows: $NumRows" -ForegroundColor Cyan
    
    $arguments = @(
        "synthetic_data_generator.py"
        "generate"
        $InputFile
        "--num-rows"
        $NumRows
        "--output-path"
        $OutputFile
        "--model"
        $Model
    )
    
    if (Test-Path $ConfigFile) {
        $arguments += "--config-file"
        $arguments += $ConfigFile
    }
    
    & python @arguments
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Synthetic data generated successfully: $OutputFile" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to generate synthetic data" -ForegroundColor Red
        exit 1
    }
}

function Validate-Data {
    if (-not $InputFile) {
        Write-Host "❌ Input file required for validate command" -ForegroundColor Red
        exit 1
    }
    
    $SyntheticFile = Join-Path $OutputDir "synthetic_$(Split-Path $InputFile -Leaf)"
    
    if (-not (Test-Path $SyntheticFile)) {
        Write-Host "❌ Synthetic data file not found: $SyntheticFile" -ForegroundColor Red
        Write-Host "Run generate command first" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "🔍 Validating data quality..." -ForegroundColor Blue
    python data_quality_validator.py validate $InputFile $SyntheticFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Validation completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Validation failed" -ForegroundColor Red
        exit 1
    }
}

function Clean-Output {
    Write-Host "🧹 Cleaning output files..." -ForegroundColor Blue
    
    if (Test-Path $OutputDir) {
        Remove-Item "$OutputDir\*" -Force -Recurse
        Write-Host "✅ Cleaned output directory: $OutputDir" -ForegroundColor Green
    }
    
    # Clean Python cache
    if (Test-Path "__pycache__") {
        Remove-Item "__pycache__" -Force -Recurse
        Write-Host "✅ Cleaned Python cache" -ForegroundColor Green
    }
    
    Get-ChildItem -Name "*.pyc" -Recurse | Remove-Item -Force
    Write-Host "✅ Cleaned compiled Python files" -ForegroundColor Green
}

function Test-ProductionReadiness {
    Write-Host "🔍 Checking production readiness..." -ForegroundColor Blue
    
    $checks = @()
    
    # Check Python dependencies
    try {
        python -c "import pandas, numpy, ollama, typer, rich; print('Core dependencies: OK')" 2>$null
        $checks += "✅ Core dependencies available"
    } catch {
        $checks += "❌ Missing core dependencies"
    }
    
    # Check Ollama
    try {
        ollama --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $checks += "✅ Ollama available"
        } else {
            $checks += "❌ Ollama not available"
        }
    } catch {
        $checks += "❌ Ollama not found"
    }
    
    # Check config file
    if (Test-Path $ConfigFile) {
        $checks += "✅ Configuration file available"
    } else {
        $checks += "❌ Configuration file missing"
    }
    
    # Check main script
    if (Test-Path "synthetic_data_generator.py") {
        $checks += "✅ Main application available"
    } else {
        $checks += "❌ Main application missing"
    }
    
    foreach ($check in $checks) {
        if ($check.StartsWith("✅")) {
            Write-Host $check -ForegroundColor Green
        } else {
            Write-Host $check -ForegroundColor Red
        }
    }
    
    $failedChecks = ($checks | Where-Object { $_.StartsWith("❌") }).Count
    if ($failedChecks -eq 0) {
        Write-Host "`n🎉 Production environment ready!" -ForegroundColor Green
    } else {
        Write-Host "`n⚠️ $failedChecks issues found. Please resolve before production deployment." -ForegroundColor Yellow
    }
}

function Batch-ProcessFiles {
    if (-not $InputFile) {
        Write-Host "❌ Input directory required for batch command" -ForegroundColor Red
        Write-Host "Usage: .\run_production.ps1 -Command batch -InputFile 'data_folder\'" -ForegroundColor Yellow
        exit 1
    }
    
    if (-not (Test-Path $InputFile)) {
        Write-Host "❌ Input directory not found: $InputFile" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "🔄 Batch processing CSV files..." -ForegroundColor Blue
    Write-Host "  Input directory: $InputFile" -ForegroundColor Cyan
    Write-Host "  Output directory: $OutputDir" -ForegroundColor Cyan
    Write-Host "  Rows per file: $NumRows" -ForegroundColor Cyan
    
    $arguments = @(
        "batch_processor.py"
        $InputFile
        $OutputDir
        "--num-rows"
        $NumRows
        "--model"
        $Model
    )
    
    if (Test-Path $ConfigFile) {
        $arguments += "--config-file"
        $arguments += $ConfigFile
    }
    
    & python @arguments
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Batch processing completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Batch processing failed" -ForegroundColor Red
        exit 1
    }
}

# Main script execution
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "setup" { Setup-Ollama }
    "generate" { Generate-SyntheticData }
    "batch" { Batch-ProcessFiles }
    "validate" { Validate-Data }
    "clean" { Clean-Output }
    "check" { Test-ProductionReadiness }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}