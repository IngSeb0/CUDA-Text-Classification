import argparse
import gc
import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, cast

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

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Competitive text classification training")
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--eval-path", default="eval.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--target-col", default="decade")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--model-name", default="BSC-LT/MrBERT-es")
    parser.add_argument("--output-dir", default="competition_runs")
    parser.add_argument("--submission-path", default="submission_competitive.csv")
    parser.add_argument("--oof-path", default="oof_predictions.csv")
    parser.add_argument("--test-probs-path", default="test_probabilities.npy")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dataloader-num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
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
    return agg["doc_id"].to_numpy(), pred_ids, agg[prob_cols].to_numpy()


def compute_chunk_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def predict_dataset(trainer: Trainer, dataset: Dataset):
    return trainer.predict(cast(Any, dataset))


def main() -> None:
    args = parse_args()

    bf16 = torch.cuda.is_available() and not args.no_bf16
    fp16 = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Modelo:", args.model_name)
    print("bf16:", bf16)

    train_df = pd.read_csv(args.train_path)
    eval_df = pd.read_csv(args.eval_path)

    train_df[args.text_col] = train_df[args.text_col].map(clean_text)
    eval_df[args.text_col] = eval_df[args.text_col].map(clean_text)

    train_df["doc_id"] = np.arange(len(train_df))
    eval_df["doc_id"] = eval_df[args.id_col].astype(int)

    label_encoder = LabelEncoder()
    train_df["label_id"] = label_encoder.fit_transform(train_df[args.target_col])
    num_labels = len(label_encoder.classes_)

    print("train shape:", train_df.shape)
    print("eval shape:", eval_df.shape)
    print("num labels:", num_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if (bf16 or fp16) else None,
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(train_df), num_labels), dtype=np.float32)
    test_probs_folds = []
    fold_scores = []

    eval_chunk_ds = build_chunk_dataset(
        eval_df, tokenizer, args.text_col, args.max_length, args.stride, with_labels=False
    )
    eval_chunk_doc_ids = np.array(eval_chunk_ds["doc_id"])
    eval_ds_for_trainer = eval_chunk_ds.remove_columns(["doc_id"])

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label_id"]), start=1):
        print(f"\n========== Fold {fold}/{args.n_splits} ==========")

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        train_ds = build_chunk_dataset(
            fold_train, tokenizer, args.text_col, args.max_length, args.stride, with_labels=True
        )
        val_ds = build_chunk_dataset(
            fold_val, tokenizer, args.text_col, args.max_length, args.stride, with_labels=True
        )

        train_ds_for_trainer = train_ds.remove_columns(["doc_id"])
        val_ds_for_trainer = val_ds.remove_columns(["doc_id"])

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
        )
        model.config.use_cache = False

        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            optim="adamw_torch_fused",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            bf16=bf16,
            fp16=fp16,
            tf32=args.tf32,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
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

        trainer.train()

        val_pred = predict_dataset(trainer, val_ds_for_trainer)
        val_probs_chunks = torch.softmax(torch.tensor(val_pred.predictions), dim=-1).numpy()
        val_doc_ids = np.array(val_ds["doc_id"])
        val_doc_ids_out, val_pred_ids, val_probs_docs = aggregate_predictions(
            val_doc_ids, val_probs_chunks, method=args.aggregation
        )

        val_truth = (
            fold_val[["doc_id", "label_id"]]
            .drop_duplicates("doc_id")
            .set_index("doc_id")
            .loc[val_doc_ids_out]["label_id"]
            .to_numpy()
        )
        fold_acc = accuracy_score(val_truth, val_pred_ids)
        fold_scores.append(fold_acc)
        oof_probs[val_doc_ids_out] = val_probs_docs
        print(f"Fold {fold} doc-level accuracy: {fold_acc:.5f}")

        test_pred = predict_dataset(trainer, eval_ds_for_trainer)
        test_probs_chunks = torch.softmax(torch.tensor(test_pred.predictions), dim=-1).numpy()
        _, _, test_probs_docs = aggregate_predictions(
            eval_chunk_doc_ids, test_probs_chunks, method=args.aggregation
        )
        test_probs_folds.append(test_probs_docs)

        del model, trainer, train_ds, val_ds, train_ds_for_trainer, val_ds_for_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nFold scores:", [round(x, 5) for x in fold_scores])
    print("CV mean accuracy:", round(float(np.mean(fold_scores)), 5))

    oof_pred_ids = oof_probs.argmax(axis=1)
    oof_acc = accuracy_score(train_df["label_id"], oof_pred_ids)
    print("OOF accuracy global:", round(float(oof_acc), 5))

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

    config_to_save = vars(args).copy()
    config_to_save["bf16"] = bf16
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    print("OOF guardado en:", args.oof_path)
    print("Submission guardada en:", args.submission_path)
    print("Probabilidades guardadas en:", args.test_probs_path)


if __name__ == "__main__":
    main()
