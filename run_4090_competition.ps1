$ErrorActionPreference = "Stop"

Write-Host "Python:"
python --version

Write-Host "Instalando dependencias base..."
python -m pip install -U pip
python -m pip install -U notebook jupyter nbconvert ipykernel
python -m pip install -U transformers datasets accelerate sentencepiece scikit-learn ipywidgets pandas numpy

Write-Host "Verificando GPU..."
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host "Ejecutando notebook Entrega1_Proyecto_(1).ipynb..."
jupyter nbconvert `
  --to notebook `
  --execute "Entrega1_Proyecto_(1).ipynb" `
  --output "Entrega1_Proyecto_ejecutado.ipynb" `
  --ExecutePreprocessor.timeout=-1