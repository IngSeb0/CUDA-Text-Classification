import argparse
import gc
import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Optional, cast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long competitive text classification training optimized for RTX 4090"
    )

    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--eval-path", default="eval.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--target-col", default="decade")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--model-name", default="BSC-LT/MrBERT-es")

    parser.add_argument("--output-dir", default="competition_runs_4090_long")
    parser.add_argument("--submission-path", default="submission_competitive_4090_long.csv")
    parser.add_argument("--oof-path", default="oof_predictions_4090_long.csv")
    parser.add_argument("--test-probs-path", default="test_probabilities_4090_long.npy")

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed-folds", action="store_true")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch"], default="steps")
    parser.add_argument("--save-strategy", choices=["steps", "epoch"], default="steps")
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=None)

    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-auto-batch-size", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--optim", default="adamw_torch_fused", choices=["adamw_torch_fused", "adamw_torch"])

    return parser.parse_args()


def clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_chunk_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    text_col: str,
    max_length: int,
    stride: int,
    with_labels: bool = True,
) -> Dataset:
    texts = df[text_col].tolist()
    doc_ids = df["doc_id"].tolist()

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    chunk_doc_ids = [doc_ids[idx] for idx in sample_mapping]

    features = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "doc_id": chunk_doc_ids,
    }

    if "token_type_ids" in tokenized:
        features["token_type_ids"] = tokenized["token_type_ids"]

    if with_labels:
        labels = df["label_id"].tolist()
        features["labels"] = [labels[idx] for idx in sample_mapping]

    return Dataset.from_dict(features)


def aggregate_predictions(doc_ids: np.ndarray, probs: np.ndarray, method: str = "mean"):
    prob_cols = [f"p_{i}" for i in range(probs.shape[1])]
    tmp = pd.DataFrame({"doc_id": doc_ids})
    prob_df = pd.DataFrame(probs, columns=prob_cols)
    tmp = pd.concat([tmp, prob_df], axis=1)

    if method == "max":
        agg = tmp.groupby("doc_id", sort=False)[prob_cols].max().reset_index()
    else:
        agg = tmp.groupby("doc_id", sort=False)[prob_cols].mean().reset_index()

    pred_ids = agg[prob_cols].to_numpy().argmax(axis=1)

    return agg["doc_id"].to_numpy(), pred_ids, agg[prob_cols].to_numpy(dtype=np.float32)


def compute_chunk_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def predict_dataset(trainer: Trainer, dataset: Dataset):
    return trainer.predict(cast(Any, dataset))


def setup_torch(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    cuda_available = torch.cuda.is_available()

    if not cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "CUDA no está disponible para PyTorch. "
            "Tu GPU puede salir en nvidia-smi, pero PyTorch no la está usando. "
            "Instala torch CUDA con: "
            "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
        )

    torch.set_float32_matmul_precision("high")

    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = True

    bf16 = cuda_available and (not args.no_bf16) and torch.cuda.is_bf16_supported()
    fp16 = cuda_available and (not bf16) and args.fp16
    tf32 = cuda_available and bool(args.tf32)

    return bf16, fp16, tf32


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    bf16, fp16, tf32 = setup_torch(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Torch:", torch.__version__, flush=True)
    print("CUDA disponible:", torch.cuda.is_available(), flush=True)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("CUDA capability:", torch.cuda.get_device_capability(0), flush=True)
        print("VRAM total GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), flush=True)

    print("Modelo:", args.model_name, flush=True)
    print("bf16:", bf16, "fp16:", fp16, "tf32:", tf32, flush=True)
    print("Train path:", args.train_path, flush=True)
    print("Eval path:", args.eval_path, flush=True)
    print("Output dir:", args.output_dir, flush=True)

    if not Path(args.train_path).exists():
        raise FileNotFoundError(f"No existe el archivo de train: {args.train_path}")

    if not Path(args.eval_path).exists():
        raise FileNotFoundError(f"No existe el archivo de eval: {args.eval_path}")

    train_df = pd.read_csv(args.train_path)
    eval_df = pd.read_csv(args.eval_path)

    train_df[args.text_col] = train_df[args.text_col].map(clean_text)
    eval_df[args.text_col] = eval_df[args.text_col].map(clean_text)

    train_df["doc_id"] = np.arange(len(train_df))
    eval_df["doc_id"] = eval_df[args.id_col].astype(int)

    label_encoder = LabelEncoder()
    train_df["label_id"] = label_encoder.fit_transform(train_df[args.target_col])
    num_labels = len(label_encoder.classes_)

    print("train shape:", train_df.shape, flush=True)
    print("eval shape:", eval_df.shape, flush=True)
    print("num labels:", num_labels, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if (bf16 or fp16) else None,
    )

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    folds = list(skf.split(train_df, train_df["label_id"]))
    end_fold = args.end_fold if args.end_fold is not None else args.n_splits

    oof_probs = np.zeros((len(train_df), num_labels), dtype=np.float32)
    test_probs_folds: list[np.ndarray] = []
    fold_scores: list[float] = []

    print("Construyendo chunks de evaluación una sola vez...", flush=True)

    eval_chunk_ds = build_chunk_dataset(
        eval_df,
        tokenizer,
        args.text_col,
        args.max_length,
        args.stride,
        with_labels=False,
    )

    eval_chunk_doc_ids = np.array(eval_chunk_ds["doc_id"])
    eval_ds_for_trainer = eval_chunk_ds.remove_columns(["doc_id"])

    save_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "bf16": bf16,
            "fp16": fp16,
            "tf32_effective": tf32,
        },
    )

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        if fold < args.start_fold or fold > end_fold:
            continue

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        done_path = fold_dir / "fold_done.json"
        val_doc_ids_path = fold_dir / "val_doc_ids.npy"
        val_probs_path = fold_dir / "val_probs_docs.npy"
        test_probs_path = fold_dir / "test_probs_docs.npy"

        if (
            args.skip_completed_folds
            and done_path.exists()
            and val_doc_ids_path.exists()
            and val_probs_path.exists()
            and test_probs_path.exists()
        ):
            print(f"\n========== Fold {fold}/{args.n_splits} ya completado. Reusando outputs. ==========", flush=True)

            val_doc_ids_out = np.load(val_doc_ids_path)
            val_probs_docs = np.load(val_probs_path)
            test_probs_docs = np.load(test_probs_path)

            oof_probs[val_doc_ids_out] = val_probs_docs

            with done_path.open("r", encoding="utf-8") as f:
                fold_info = json.load(f)

            fold_scores.append(float(fold_info["fold_acc"]))
            test_probs_folds.append(test_probs_docs)
            continue

        print(f"\n========== Fold {fold}/{args.n_splits} ==========", flush=True)

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        print("Construyendo chunks train/val...", flush=True)

        train_ds = build_chunk_dataset(
            fold_train,
            tokenizer,
            args.text_col,
            args.max_length,
            args.stride,
            with_labels=True,
        )

        val_ds = build_chunk_dataset(
            fold_val,
            tokenizer,
            args.text_col,
            args.max_length,
            args.stride,
            with_labels=True,
        )

        train_ds_for_trainer = train_ds.remove_columns(["doc_id"])
        val_ds_for_trainer = val_ds.remove_columns(["doc_id"])

        print("Chunks train:", len(train_ds_for_trainer), flush=True)
        print("Chunks val:", len(val_ds_for_trainer), flush=True)

        print("Cargando modelo...", flush=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
        )

        model.config.use_cache = False

        effective_optim = args.optim

        if effective_optim == "adamw_torch_fused" and not torch.cuda.is_available():
            effective_optim = "adamw_torch"

        training_args = TrainingArguments(
            output_dir=str(fold_dir),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            optim=effective_optim,
            eval_strategy=args.eval_strategy,
            save_strategy=args.save_strategy,
            eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
            save_steps=args.save_steps if args.save_strategy == "steps" else None,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            bf16=bf16,
            fp16=fp16,
            tf32=tf32,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=torch.cuda.is_available(),
            auto_find_batch_size=not args.no_auto_batch_size,
            report_to="none",
            seed=args.seed + fold,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_for_trainer,
            eval_dataset=val_ds_for_trainer,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_chunk_metrics,
        )

        resume_checkpoint: Optional[str] = None

        if args.resume:
            resume_checkpoint = get_last_checkpoint(str(fold_dir))

            if resume_checkpoint is not None:
                print("Reanudando desde checkpoint:", resume_checkpoint, flush=True)

        print("Iniciando entrenamiento...", flush=True)

        trainer.train(resume_from_checkpoint=resume_checkpoint)

        print("Prediciendo validación...", flush=True)

        val_pred = predict_dataset(trainer, val_ds_for_trainer)
        val_probs_chunks = torch.softmax(torch.tensor(val_pred.predictions), dim=-1).numpy()

        val_doc_ids = np.array(val_ds["doc_id"])

        val_doc_ids_out, val_pred_ids, val_probs_docs = aggregate_predictions(
            val_doc_ids,
            val_probs_chunks,
            method=args.aggregation,
        )

        val_truth = (
            fold_val[["doc_id", "label_id"]]
            .drop_duplicates("doc_id")
            .set_index("doc_id")
            .loc[val_doc_ids_out]["label_id"]
            .to_numpy()
        )

        fold_acc = accuracy_score(val_truth, val_pred_ids)

        fold_scores.append(float(fold_acc))
        oof_probs[val_doc_ids_out] = val_probs_docs

        print(f"Fold {fold} doc-level accuracy: {fold_acc:.5f}", flush=True)

        print("Prediciendo eval/test...", flush=True)

        test_pred = predict_dataset(trainer, eval_ds_for_trainer)
        test_probs_chunks = torch.softmax(torch.tensor(test_pred.predictions), dim=-1).numpy()

        _, _, test_probs_docs = aggregate_predictions(
            eval_chunk_doc_ids,
            test_probs_chunks,
            method=args.aggregation,
        )

        test_probs_folds.append(test_probs_docs)

        np.save(val_doc_ids_path, val_doc_ids_out)
        np.save(val_probs_path, val_probs_docs)
        np.save(test_probs_path, test_probs_docs)

        save_json(
            done_path,
            {
                "fold": fold,
                "fold_acc": float(fold_acc),
            },
        )

        del model
        del trainer
        del train_ds
        del val_ds
        del train_ds_for_trainer
        del val_ds_for_trainer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not test_probs_folds:
        raise RuntimeError(
            "No se entrenó ni se reutilizó ningún fold. "
            "Revisa --start-fold/--end-fold o --skip-completed-folds."
        )

    print("\nFold scores:", [round(x, 5) for x in fold_scores], flush=True)
    print("CV mean accuracy:", round(float(np.mean(fold_scores)), 5), flush=True)

    covered = oof_probs.sum(axis=1) > 0

    if covered.all():
        oof_pred_ids = oof_probs.argmax(axis=1)
        oof_acc = accuracy_score(train_df["label_id"], oof_pred_ids)
        print("OOF accuracy global:", round(float(oof_acc), 5), flush=True)
    else:
        oof_pred_ids = oof_probs.argmax(axis=1)
        print(f"OOF parcial: {covered.sum()}/{len(covered)} documentos cubiertos.", flush=True)

    oof_df = train_df[["doc_id", args.target_col, "label_id"]].copy()
    oof_df["pred_label_id"] = oof_pred_ids
    oof_df["pred_decade"] = label_encoder.inverse_transform(oof_pred_ids)
    oof_df.to_csv(args.oof_path, index=False)

    test_probs = np.mean(test_probs_folds, axis=0)
    test_pred_ids = test_probs.argmax(axis=1)
    test_pred_labels = label_encoder.inverse_transform(test_pred_ids)

    submission = eval_df[[args.id_col]].copy()
    submission[args.target_col] = test_pred_labels
    submission.to_csv(args.submission_path, index=False)

    np.save(args.test_probs_path, test_probs)

    print("OOF guardado en:", args.oof_path, flush=True)
    print("Submission guardada en:", args.submission_path, flush=True)
    print("Probabilidades guardadas en:", args.test_probs_path, flush=True)


if __name__ == "__main__":
    main()