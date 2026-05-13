$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$RUN_ROOT = Join-Path $env:LOCALAPPDATA "proyecto_modelos_4090"
New-Item -ItemType Directory -Force -Path $RUN_ROOT | Out-Null

$logFile = Join-Path $RUN_ROOT "training_4090_long_$timestamp.log"

Start-Transcript -Path $logFile

try {
    Write-Host "Carpeta actual:"
    Get-Location

    Write-Host "Carpeta local de salida:"
    Write-Host $RUN_ROOT

    if (-not (Test-Path ".\train_competitive_4090_long_FIXED.py")) {
        throw "No encuentro train_competitive_4090_long_FIXED.py en esta carpeta."
    }

    if (Test-Path ".\train.csv") {
        $trainPath = "train.csv"
    } elseif (Test-Path ".\train(2).csv") {
        $trainPath = "train(2).csv"
    } else {
        throw "No encuentro train.csv ni train(2).csv."
    }

    if (Test-Path ".\eval.csv") {
        $evalPath = "eval.csv"
    } elseif (Test-Path ".\eval(4).csv") {
        $evalPath = "eval(4).csv"
    } else {
        throw "No encuentro eval.csv ni eval(4).csv."
    }

    Write-Host "Train path detectado: $trainPath"
    Write-Host "Eval path detectado: $evalPath"

    Write-Host "Python:"
    python --version

    Write-Host "Verificando GPU..."
    nvidia-smi
    python -c "import torch; print('Torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:TOKENIZERS_PARALLELISM = "false"

    $outputDir = Join-Path $RUN_ROOT "competition_runs_4090_long_$timestamp"
    $submissionPath = Join-Path $RUN_ROOT "submission_competitive_4090_long_$timestamp.csv"
    $oofPath = Join-Path $RUN_ROOT "oof_predictions_4090_long_$timestamp.csv"
    $testProbsPath = Join-Path $RUN_ROOT "test_probabilities_4090_long_$timestamp.npy"

    Write-Host "Output dir:"
    Write-Host $outputDir

    Write-Host "Verificando archivo Python que se va a ejecutar:"
    Get-Item ".\train_competitive_4090_long_FIXED.py"
    Get-Content ".\train_competitive_4090_long_FIXED.py" -Tail 5

    Write-Host "Lanzando entrenamiento largo para RTX 4090..."

    python -u ".\train_competitive_4090_long_FIXED.py" `
      --train-path "$trainPath" `
      --eval-path "$evalPath" `
      --model-name "BSC-LT/MrBERT-es" `
      --output-dir "$outputDir" `
      --submission-path "$submissionPath" `
      --oof-path "$oofPath" `
      --test-probs-path "$testProbsPath" `
      --max-length 512 `
      --stride 128 `
      --n-splits 3 `
      --epochs 4 `
      --train-batch-size 24 `
      --eval-batch-size 64 `
      --grad-accum-steps 1 `
      --dataloader-num-workers 8 `
      --save-steps 500 `
      --eval-steps 500 `
      --logging-steps 25 `
      --save-total-limit 2 `
      --tf32

    if ($LASTEXITCODE -ne 0) {
        throw "El entrenamiento falló con exit code $LASTEXITCODE"
    }

    Write-Host "Entrenamiento terminado correctamente."
    Write-Host "Resultados guardados en:"
    Write-Host $RUN_ROOT
}
finally {
    Stop-Transcript
}