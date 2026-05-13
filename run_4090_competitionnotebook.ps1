$ErrorActionPreference = "Stop"

Write-Host "Python:"
python --version

Write-Host "Instalando dependencias base..."
python -m pip install -U pip
python -m pip install -U transformers datasets accelerate sentencepiece scikit-learn ipywidgets

Write-Host "Instalando PyTorch con CUDA..."
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Write-Host "Verificando GPU..."
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host "Lanzando entrenamiento competitivo para RTX 4090..."
python train_competitive.py `
  --model-name BSC-LT/MrBERT-es `
  --max-length 384 `
  --stride 96 `
  --n-splits 3 `
  --epochs 3 `
  --train-batch-size 16 `
  --eval-batch-size 32 `
  --grad-accum-steps 1 `
  --dataloader-num-workers 10 `
  --tf32