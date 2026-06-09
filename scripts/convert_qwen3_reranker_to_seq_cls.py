from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def from_2_way_softmax(
    causal_lm,
    seq_cls_model,
    tokenizer,
    tokens: list[str],
    device: str,
) -> None:
    """
    Qwen3-Reranker 官方原始形式本质上依赖 "no"/"yes" 两个 token 的 logits。
    这里把它转换成单分数 seq-cls 头：
        score_weight = W_yes - W_no
    """
    if len(tokens) != 2:
        raise ValueError("from_2_way_softmax requires exactly 2 tokens, e.g. ['no', 'yes'].")

    false_token, true_token = tokens
    false_id = tokenizer.convert_tokens_to_ids(false_token)
    true_id = tokenizer.convert_tokens_to_ids(true_token)

    if false_id is None or true_id is None:
        raise ValueError(f"Cannot find token ids for tokens={tokens}")

    lm_head_weights = causal_lm.lm_head.weight
    score_weight = (
        lm_head_weights[true_id].to(device=device, dtype=torch.float32)
        - lm_head_weights[false_id].to(device=device, dtype=torch.float32)
    )

    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight.unsqueeze(0))
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()


def pick_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-Reranker causal-lm checkpoint to seq-cls checkpoint."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Original model path, e.g. /root/autodl-tmp/RAG/Qwen3-Reranker-0.6B",
    )
    parser.add_argument(
        "--classifier_from_tokens",
        type=str,
        default='["no", "yes"]',
        help='JSON list of tokens, default: ["no", "yes"]',
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Output path of converted seq-cls model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Conversion device, e.g. "cuda", "cuda:0", or "cpu"',
    )
    parser.add_argument(
        "--use-pad-token",
        action="store_true",
        help="Whether to mark config.use_pad_token = True",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = json.loads(args.classifier_from_tokens)
    if not isinstance(tokens, list) or not all(isinstance(x, str) for x in tokens):
        raise ValueError("--classifier_from_tokens must be a JSON string list")

    dtype = pick_dtype(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    causal_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device).eval()

    seq_cls_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    ).to(args.device).eval()

    from_2_way_softmax(
        causal_lm=causal_lm,
        seq_cls_model=seq_cls_model,
        tokenizer=tokenizer,
        tokens=tokens,
        device=args.device,
    )

    seq_cls_model.config.pad_token_id = tokenizer.pad_token_id
    seq_cls_model.config.use_pad_token = args.use_pad_token

    seq_cls_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Converted seq-cls model saved to: {output_dir}")


if __name__ == "__main__":
    main()
