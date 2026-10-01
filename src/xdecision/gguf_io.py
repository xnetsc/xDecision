"""Export and verify complete xDecision checkpoints in GGUF.

The file keeps the original tensor names and includes the ModernBERT encoder,
decision transformer, marker scorer, act/escalate head, temperatures, and
tokenizer metadata. The custom ``xdecision`` architecture name prevents a
generic BERT loader from silently dropping the decision-specific tensors.

Examples:

  python -m xdecision.gguf_io export --checkpoint work/model \
      --output models/xDecision-F16.gguf --quantization F16
  python -m xdecision.gguf_io verify --checkpoint work/model \
      --gguf models/xDecision-F16.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


FORMAT_VERSION = 2
ARCHITECTURE = "xdecision"
QUANTIZATIONS = {
    "F16": None,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
}


def _sha256(path: os.PathLike[str] | str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: os.PathLike[str] | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _tokenizer_metadata(writer: gguf.GGUFWriter, tokenizer_path: Path) -> None:
    tokenizer = _json(tokenizer_path)
    model = tokenizer["model"]
    vocab = model["vocab"]
    tokens = [""] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    if any(token == "" for token in tokens):
        raise ValueError("tokenizer vocabulary has missing ids")

    added = {item["id"]: item for item in tokenizer.get("added_tokens", [])}
    token_types = []
    for token_id in range(len(tokens)):
        item = added.get(token_id)
        if item and item.get("special"):
            token_types.append(gguf.TokenType.CONTROL)
        elif item:
            token_types.append(gguf.TokenType.USER_DEFINED)
        else:
            token_types.append(gguf.TokenType.NORMAL)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("xdecision-mmbert-bpe")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(model.get("merges", []))
    writer.add_pad_token_id(0)
    writer.add_eos_token_id(1)
    writer.add_sep_token_id(1)
    writer.add_bos_token_id(2)
    writer.add_unk_token_id(3)
    writer.add_mask_token_id(4)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(True)


def export_gguf(checkpoint: os.PathLike[str] | str, output: os.PathLike[str] | str,
                quantization: str = "F16") -> dict:
    checkpoint = Path(checkpoint)
    output = Path(output)
    quantization = quantization.upper()
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {sorted(QUANTIZATIONS)}")

    source = checkpoint / "model.safetensors"
    encoder_config_path = checkpoint / "encoder" / "config.json"
    agent_config_path = checkpoint / "rl_agent_config.json"
    tokenizer_path = checkpoint / "tokenizer" / "tokenizer.json"
    for required in (source, encoder_config_path, agent_config_path, tokenizer_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    encoder_config = _json(encoder_config_path)
    agent_config = _json(agent_config_path)
    agent_config = {k: v for k, v in agent_config.items() if k not in ("training", "posttrain", "calibration")}
    agent_config["model_name"] = "xDecision"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    writer = gguf.GGUFWriter(temp_output, ARCHITECTURE, use_temp_file=True)
    writer.add_name("xDecision")
    writer.add_basename("xDecision")
    writer.add_finetune("xDecision")
    writer.add_author("xnetsc")
    writer.add_type("model")
    writer.add_description("A multilingual structured-decision model based on Laya and mmBERT, with stored temperature scaling.")
    writer.add_license("Apache-2.0")
    writer.add_repo_url("https://github.com/xnetsc/xDecision")
    writer.add_base_model_count(2)
    writer.add_base_model_name(0, "Laya Multilingual")
    writer.add_base_model_repo_url(0, "https://huggingface.co/convaiinnovations/laya-multilingual")
    writer.add_base_model_name(1, "mmBERT-base")
    writer.add_base_model_repo_url(1, "https://huggingface.co/jhu-clsp/mmBERT-base")
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16 if quantization == "F16"
                             else gguf.LlamaFileType.MOSTLY_Q8_0))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_context_length(int(agent_config.get("max_len", 1024)))
    writer.add_embedding_length(int(encoder_config["hidden_size"]))
    writer.add_block_count(int(encoder_config["num_hidden_layers"]))
    writer.add_head_count(int(encoder_config["num_attention_heads"]))
    writer.add_feed_forward_length(int(encoder_config["intermediate_size"]))
    writer.add_vocab_size(int(encoder_config["vocab_size"]))
    writer.add_string("xdecision.format", "complete-checkpoint")
    writer.add_string("xdecision.name", "xDecision")
    writer.add_uint32("xdecision.format_version", FORMAT_VERSION)
    writer.add_string("xdecision.quantization", quantization)
    writer.add_string("xdecision.source_sha256", _sha256(source))
    writer.add_string("xdecision.encoder_config", json.dumps(encoder_config, sort_keys=True, separators=(",", ":")))
    writer.add_string("xdecision.agent_config", json.dumps(agent_config, sort_keys=True, separators=(",", ":")))
    writer.add_string("xdecision.tokenizer_json", tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_config = _json(checkpoint / "tokenizer" / "tokenizer_config.json")
    tokenizer_config.pop("name_or_path", None)
    tokenizer_config.pop("_name_or_path", None)
    writer.add_string("xdecision.tokenizer_config", json.dumps(tokenizer_config, ensure_ascii=False))
    writer.add_uint32("xdecision.head_layers", int(agent_config.get("head_layers", 2)))
    writer.add_bool("xdecision.has_decision_head", True)
    writer.add_bool("xdecision.has_act_head", True)
    _tokenizer_metadata(writer, tokenizer_path)

    qtype = QUANTIZATIONS[quantization]
    quantized_count = 0
    with safe_open(source, framework="np") as tensors:
        names = list(tensors.keys())
        writer.add_array("xdecision.tensor_names", names)
        writer.add_uint32("xdecision.tensor_count", len(names))
        for name in names:
            tensor = tensors.get_tensor(name)
            if qtype is not None and tensor.ndim >= 2 and tensor.shape[-1] % 32 == 0:
                encoded = gguf.quantize(tensor.astype(np.float32, copy=False), qtype)
                writer.add_tensor(name, encoded, raw_dtype=qtype)
                quantized_count += 1
            else:
                writer.add_tensor(name, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    os.replace(temp_output, output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "quantization": quantization,
        "tensor_count": len(names),
        "quantized_tensor_count": quantized_count,
        "source_sha256": _sha256(source),
    }


def _decoded_tensor(tensor) -> np.ndarray:
    if tensor.tensor_type in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F64,
        gguf.GGMLQuantizationType.I8,
        gguf.GGMLQuantizationType.I16,
        gguf.GGMLQuantizationType.I32,
        gguf.GGMLQuantizationType.I64,
    ):
        return np.asarray(tensor.data)
    return gguf.dequantize(tensor.data, tensor.tensor_type)


def verify_gguf(checkpoint: os.PathLike[str] | str, gguf_path: os.PathLike[str] | str) -> dict:
    checkpoint = Path(checkpoint)
    source_path = checkpoint / "model.safetensors"
    reader = gguf.GGUFReader(gguf_path)
    encoded = {tensor.name: tensor for tensor in reader.tensors}
    rows = []
    with safe_open(source_path, framework="np") as source:
        source_names = list(source.keys())
        if set(source_names) != set(encoded):
            missing = sorted(set(source_names) - set(encoded))
            extra = sorted(set(encoded) - set(source_names))
            raise ValueError(f"tensor names differ: missing={missing}, extra={extra}")
        squared_error = 0.0
        element_count = 0
        max_abs = 0.0
        for name in source_names:
            expected = source.get_tensor(name).astype(np.float32, copy=False)
            actual = _decoded_tensor(encoded[name]).astype(np.float32, copy=False)
            if expected.shape != actual.shape:
                raise ValueError(f"{name}: shape {actual.shape} != {expected.shape}")
            error = actual - expected
            tensor_max = float(np.max(np.abs(error))) if error.size else 0.0
            squared_error += float(np.sum(error * error, dtype=np.float64))
            element_count += error.size
            max_abs = max(max_abs, tensor_max)
            rows.append({"name": name, "shape": list(expected.shape),
                         "type": encoded[name].tensor_type.name, "max_abs": tensor_max})
    return {
        "gguf": str(gguf_path),
        "source": str(source_path),
        "tensor_count": len(rows),
        "rmse": (squared_error / element_count) ** 0.5,
        "max_abs": max_abs,
        "gguf_sha256": _sha256(gguf_path),
        "source_sha256": _sha256(source_path),
        "tensors": rows,
    }


def restore_gguf(gguf_path: os.PathLike[str] | str, metadata_dir: os.PathLike[str] | str,
                 output: os.PathLike[str] | str) -> dict:
    """Restore a GGUF release to the directory layout consumed by the Laya runtime."""
    gguf_path = Path(gguf_path)
    metadata_dir = Path(metadata_dir) if metadata_dir else None
    output = Path(output)
    required = ([metadata_dir / "encoder" / "config.json", metadata_dir / "tokenizer" / "tokenizer.json",
                metadata_dir / "rl_agent_config.json"] if metadata_dir else [])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    reader = gguf.GGUFReader(gguf_path)
    embedded = {}
    if metadata_dir is None:
        for key, destination in {
            "xdecision.encoder_config": "encoder/config.json",
            "xdecision.agent_config": "rl_agent_config.json",
            "xdecision.tokenizer_json": "tokenizer/tokenizer.json",
            "xdecision.tokenizer_config": "tokenizer/tokenizer_config.json",
        }.items():
            field = reader.get_field(key)
            if field is None:
                raise ValueError(f"Missing {key}; older GGUF requires --metadata-dir")
            embedded[destination] = json.loads(field.contents())
    tensors = {}
    for tensor in reader.tensors:
        decoded = _decoded_tensor(tensor)
        if tensor.name == "temperature":
            decoded = decoded.astype(np.float32, copy=False)
        else:
            decoded = decoded.astype(np.float16, copy=False)
        tensors[tensor.name] = np.ascontiguousarray(decoded)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "model.safetensors.tmp"
    save_file(tensors, temporary)
    os.replace(temporary, output / "model.safetensors")
    if metadata_dir:
        for directory in ("encoder", "tokenizer"):
            shutil.copytree(metadata_dir / directory, output / directory)
        shutil.copy2(metadata_dir / "rl_agent_config.json", output / "rl_agent_config.json")
    else:
        for filename, value in embedded.items():
            destination = output / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "gguf": str(gguf_path),
        "output": str(output),
        "tensor_count": len(tensors),
        "model_sha256": _sha256(output / "model.safetensors"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--quantization", choices=sorted(QUANTIZATIONS), default="F16")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--checkpoint", required=True)
    verify_parser.add_argument("--gguf", required=True)
    verify_parser.add_argument("--output")
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--gguf", required=True)
    restore_parser.add_argument("--metadata-dir", help="Only required for older format-version 1 exports")
    restore_parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "export":
        result = export_gguf(args.checkpoint, args.output, args.quantization)
    elif args.command == "verify":
        result = verify_gguf(args.checkpoint, args.gguf)
    else:
        result = restore_gguf(args.gguf, args.metadata_dir, args.output)
    printed = result if args.command != "verify" else {k: v for k, v in result.items() if k != "tensors"}
    print(json.dumps(printed, ensure_ascii=False, indent=2))
    if getattr(args, "output", None) and args.command == "verify":
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
