$ErrorActionPreference = "Stop"

# Cambia a $true si en el otro PC aún no instalaste dependencias.
$INSTALL_DEPS = $false

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$RUN_ROOT = Join-Path $env:LOCALAPPDATA "proyecto_modelos_4090"
$baseDir = Join-Path $RUN_ROOT "model_queue_pc2_$timestamp"

New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

$scriptPath = ".\train_competitive_4090_long_FIXED.py"

if (-not (Test-Path $scriptPath)) {
    throw "No encuentro train_competitive_4090_long_FIXED.py en esta carpeta."
}

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

if ($INSTALL_DEPS) {
    Write-Host "Instalando dependencias..."
    python -m pip install -U pip
    python -m pip install -U transformers datasets accelerate sentencepiece scikit-learn pandas numpy
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
}

Write-Host "Validando CSV..."
python -c "import pandas as pd; print('train:', pd.read_csv('$trainPath').shape); print('eval:', pd.read_csv('$evalPath').shape)"

if ($LASTEXITCODE -ne 0) {
    throw "El train/eval no se pudo leer. Revisa si eval.csv está corrupto o incompleto."
}

Write-Host "Verificando GPU..."
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

if ($LASTEXITCODE -ne 0) {
    throw "Falló la verificación de PyTorch/CUDA."
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"

# PC2: pruebas distintas y más enfocadas. No repite exactamente toda la cola.
$configs = @(
    @{
        tag = "roberta_bne_e1"
        model = "PlanTL-GOB-ES/roberta-base-bne"
        maxLength = 384
        stride = 96
        epochs = 1
        trainBatch = 24
        evalBatch = 64
        lr = "2e-5"
    },
    @{
        tag = "roberta_bne_e2"
        model = "PlanTL-GOB-ES/roberta-base-bne"
        maxLength = 384
        stride = 96
        epochs = 2
        trainBatch = 24
        evalBatch = 64
        lr = "2e-5"
    },
    @{
        tag = "mdeberta_e1"
        model = "microsoft/mdeberta-v3-base"
        maxLength = 384
        stride = 96
        epochs = 1
        trainBatch = 16
        evalBatch = 32
        lr = "2e-5"
    },
    @{
        tag = "mdeberta_e2"
        model = "microsoft/mdeberta-v3-base"
        maxLength = 384
        stride = 96
        epochs = 2
        trainBatch = 16
        evalBatch = 32
        lr = "1.5e-5"
    },
    @{
        tag = "bertin_e1"
        model = "bertin-project/bertin-roberta-base-spanish"
        maxLength = 384
        stride = 96
        epochs = 1
        trainBatch = 24
        evalBatch = 64
        lr = "2e-5"
    }
)

$resultsFile = Join-Path $baseDir "queue_results.csv"
$bestFile = Join-Path $baseDir "best_model_so_far.txt"

foreach ($cfg in $configs) {
    $model = $cfg.model
    $tag = $cfg.tag

    $safeName = $tag -replace "/", "_" -replace ":", "_" -replace "\\", "_"
    $runDir = Join-Path $baseDir $safeName
    $logFile = Join-Path $runDir "train.log"
    $submissionPath = Join-Path $runDir "submission_$safeName.csv"
    $oofPath = Join-Path $runDir "oof_$safeName.csv"
    $probsPath = Join-Path $runDir "test_probs_$safeName.npy"

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host ""
    Write-Host "======================================================"
    Write-Host "Iniciando configuración: $tag"
    Write-Host "Modelo: $model"
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
        --max-length $cfg.maxLength `
        --stride $cfg.stride `
        --n-splits 3 `
        --epochs $cfg.epochs `
        --train-batch-size $cfg.trainBatch `
        --eval-batch-size $cfg.evalBatch `
        --grad-accum-steps 1 `
        --learning-rate $cfg.lr `
        --weight-decay 0.01 `
        --warmup-ratio 0.1 `
        --dataloader-num-workers 2 `
        --aggregation mean `
        --eval-strategy epoch `
        --save-strategy epoch `
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

    $oofAccuracy = "NA"

    if ((Test-Path $oofPath) -and ($status -eq "OK")) {
        $scoreScript = Join-Path $runDir "score_oof.py"

@"
import pandas as pd

df = pd.read_csv(r"$oofPath")

if "decade" not in df.columns or "pred_decade" not in df.columns:
    print("NA")
else:
    acc = (df["decade"].astype(str) == df["pred_decade"].astype(str)).mean()
    print(f"{acc:.6f}")
"@ | Out-File $scoreScript -Encoding utf8

        $oofAccuracy = python $scoreScript
        $oofAccuracy = $oofAccuracy.Trim()
    }

    [PSCustomObject]@{
        tag = $tag
        model = $model
        status = $status
        oof_accuracy = $oofAccuracy
        start_time = $startTime
        end_time = $endTime
        exit_code = $exitCode
        log_file = $logFile
        output_dir = $runDir
        submission = $submissionPath
    } | Export-Csv -Path $resultsFile -NoTypeInformation -Append -Encoding utf8

    Write-Host "OOF accuracy de esta corrida: $oofAccuracy"

    $completed = Import-Csv $resultsFile | Where-Object {
        $_.status -eq "OK" -and $_.oof_accuracy -ne "NA"
    }

    if ($completed.Count -gt 0) {
        $best = $completed | Sort-Object { [double]$_.oof_accuracy } -Descending | Select-Object -First 1

        "BEST SO FAR" | Out-File $bestFile -Encoding utf8
        "tag: $($best.tag)" | Out-File $bestFile -Append -Encoding utf8
        "model: $($best.model)" | Out-File $bestFile -Append -Encoding utf8
        "oof_accuracy: $($best.oof_accuracy)" | Out-File $bestFile -Append -Encoding utf8
        "submission: $($best.submission)" | Out-File $bestFile -Append -Encoding utf8

        Write-Host ""
        Write-Host "Mejor hasta ahora:"
        Write-Host "Tag: $($best.tag)"
        Write-Host "Modelo: $($best.model)"
        Write-Host "OOF accuracy: $($best.oof_accuracy)"
        Write-Host "Submission: $($best.submission)"
    }

    Write-Host "Limpiando antes del siguiente modelo..."
    Start-Sleep -Seconds 30
}

Write-Host ""
Write-Host "Cola terminada."
Write-Host "Resultados en: $resultsFile"
Write-Host "Mejor modelo en: $bestFile"

Write-Host ""
Write-Host "Resumen ordenado:"
Import-Csv $resultsFile |
    Sort-Object {
        if ($_.oof_accuracy -eq "NA") { -1 } else { [double]$_.oof_accuracy }
    } -Descending |
    Format-Table tag, model, status, oof_accuracy, submission -Auto