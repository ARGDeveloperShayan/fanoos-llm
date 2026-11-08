from __future__ import annotations
import argparse
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser(description="Convert HF export to GGUF")
    ap.add_argument("--hf-dir", required=True, help="HF export directory (with model.safetensors, config.json)")
    ap.add_argument("--out-dir", required=True, help="Output directory for GGUF files")
    ap.add_argument("--llama-cpp-dir", default=None, help="Path to llama.cpp repo (for convert_hf_to_gguf.py)")
    ap.add_argument("--outtype", choices=["f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"], default="f16")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.llama_cpp_dir:
        convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    else:
        for p in [
            os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
            "/usr/local/llama.cpp/convert_hf_to_gguf.py",
            "llama.cpp/convert_hf_to_gguf.py",
        ]:
            if os.path.isfile(p):
                convert_script = p
                break
        else:
            print("[export_gguf] ERROR: Could not find llama.cpp/convert_hf_to_gguf.py")
            print("         Clone llama.cpp and pass --llama-cpp-dir, or install llama-cpp-python")
            sys.exit(1)

    if not os.path.isfile(convert_script):
        print(f"[export_gguf] ERROR: {convert_script} not found")
        sys.exit(1)

    f16_path = os.path.join(args.out_dir, "ggml-model-f16.gguf")
    print(f"[export_gguf] Converting to GGUF f16...")
    cmd = [
        sys.executable, convert_script,
        args.hf_dir,
        "--outfile", f16_path,
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[export_gguf] Conversion failed:\n{result.stderr}")
        sys.exit(1)
    print(f"[export_gguf] f16 GGUF saved to {f16_path}")

    if args.outtype != "f16":
        quant_path = os.path.join(args.out_dir, f"ggml-model-{args.outtype}.gguf")
        print(f"[export_gguf] Quantizing to {args.outtype}...")

        llama_cpp_dir = os.path.dirname(convert_script)
        quantize_bin = os.path.join(llama_cpp_dir, "quantize")
        if not os.path.isfile(quantize_bin):
            quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "quantize")
        if not os.path.isfile(quantize_bin):
            print(f"[export_gguf] WARNING: quantize binary not found in {llama_cpp_dir}")
            print("         You can quantize manually: llama.cpp/quantize <f16> <out> <type>")
        else:
            cmd = [quantize_bin, f16_path, quant_path, args.outtype]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[export_gguf] Quantization failed:\n{result.stderr}")
            else:
                print(f"[export_gguf] Quantized GGUF saved to {quant_path}")

    print(f"[export_gguf] Done! Output in {args.out_dir}")

if __name__ == "__main__":
    main()