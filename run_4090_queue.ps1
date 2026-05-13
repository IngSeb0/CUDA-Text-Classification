$ErrorActionPreference = "Continue"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$baseDir = "model_queue_$timestamp"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

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

    python train_competitive_4090_long.py `
        --train-path "train(2).csv" `
        --eval-path "eval(4).csv" `
        --model-name $model `
        --output-dir $runDir `
        --submission-path $submissionPath `
        --oof-path $oofPath `
        --test-probs-path $probsPath `
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