$ErrorActionPreference = "Continue"

# Carpeta base local para evitar problemas de cuota en \\CODD o Desktop institucional
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RUN_ROOT = Join-Path $env:LOCALAPPDATA "proyecto_modelos_4090"
$baseDir = Join-Path $RUN_ROOT "model_queue_$timestamp"

New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

# Verificar que el .py correcto exista
$scriptPath = ".\train_competitive_4090_long_FIXED.py"

if (-not (Test-Path $scriptPath)) {
    throw "No encuentro train_competitive_4090_long_FIXED.py en esta carpeta. Copialo a la misma carpeta del .ps1."
}

# Detectar archivos de entrenamiento y evaluación
if (Test-Path ".\train(2).csv") {
    $trainPath = "train(2).csv"
} elseif (Test-Path ".\train.csv") {
    $trainPath = "train.csv"
} else {
    throw "No encuentro train(2).csv ni train.csv."
}

if (Test-Path ".\eval(4).csv") {
    $evalPath = "eval(4).csv"
} elseif (Test-Path ".\eval.csv") {
    $evalPath = "eval.csv"
} else {
    throw "No encuentro eval(4).csv ni eval.csv."
}

Write-Host "Train detectado: $trainPath"
Write-Host "Eval detectado: $evalPath"

# Validar que los CSV sí se puedan leer antes de dejar corriendo la cola
Write-Host "Validando CSV..."
python -c "import pandas as pd; print('train:', pd.read_csv('$trainPath').shape); print('eval:', pd.read_csv('$evalPath').shape)"

if ($LASTEXITCODE -ne 0) {
    throw "El train/eval no se pudo leer. Revisa si eval.csv está corrupto o incompleto."
}

# Verificar GPU
Write-Host "Verificando GPU..."
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

if ($LASTEXITCODE -ne 0) {
    throw "Falló la verificación de PyTorch/CUDA."
}

# Variables útiles para entrenamientos largos
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"

# Modelos a probar en cola
$models = @(
    "BSC-LT/MrBERT-es",
    "EuroBERT/EuroBERT-210m",
    "microsoft/mdeberta-v3-base",
    "PlanTL-GOB-ES/roberta-base-bne",
    "bertin-project/bertin-roberta-base-spanish",
    "dccuchile/bert-base-spanish-wwm-cased",
    "FacebookAI/xlm-roberta-large"
)

$resultsFile = Join-Path $baseDir "queue_results.csv"
"model,status,start_time,end_time,exit_code,log_file,output_dir,submission" | Out-File $resultsFile -Encoding utf8

foreach ($model in $models) {
    $safeName = $model -replace "/", "_" -replace ":", "_" -replace "\\", "_"
    $runDir = Join-Path $baseDir $safeName
    $logFile = Join-Path $runDir "train.log"
    $submissionPath = Join-Path $runDir "submission_$safeName.csv"
    $oofPath = Join-Path $runDir "oof_$safeName.csv"
    $probsPath = Join-Path $runDir "test_probs_$safeName.npy"

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host ""
    Write-Host "======================================================"
    Write-Host "Iniciando modelo: $model"
    Write-Host "Output: $runDir"
    Write-Host "Log: $logFile"
    Write-Host "======================================================"

    python -u $scriptPath `
        --train-path "$trainPath" `
        --eval-path "$evalPath" `
        --model-name "$model" `
        --output-dir "$runDir" `
        --submission-path "$submissionPath" `
        --oof-path "$oofPath" `
        --test-probs-path "$probsPath" `
        --max-length 384 `
        --stride 96 `
        --n-splits 3 `
        --epochs 1 `
        --train-batch-size 16 `
        --eval-batch-size 32 `
        --grad-accum-steps 1 `
        --learning-rate 2e-5 `
        --weight-decay 0.01 `
        --warmup-ratio 0.1 `
        --dataloader-num-workers 8 `
        --aggregation mean `
        --save-steps 500 `
        --eval-steps 500 `
        --logging-steps 25 `
        --save-total-limit 1 `
        --tf32 `
        2>&1 | Tee-Object -FilePath $logFile

    $exitCode = $LASTEXITCODE
    $endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    if ($exitCode -eq 0) {
        $status = "OK"
        Write-Host "Modelo terminado correctamente: $model"
    } else {
        $status = "FAILED"
        Write-Host "Modelo falló: $model con exit code $exitCode"
    }

    "`"$model`",$status,`"$startTime`",`"$endTime`",$exitCode,`"$logFile`",`"$runDir`",`"$submissionPath`"" | Out-File $resultsFile -Append -Encoding utf8

    Write-Host "Limpiando memoria antes del siguiente modelo..."
    Start-Sleep -Seconds 30
}

Write-Host ""
Write-Host "Cola terminada."
Write-Host "Resultados en: $resultsFile"