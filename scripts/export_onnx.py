#!/usr/bin/env python3
"""
ONNX Export Script for KLA Semiconductor Image Restoration
Exports trained PyTorch model to ONNX for fast inference on H100.
"""
import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model, create_model_from_config


def export_onnx(
    weights_path: str,
    output_path: str,
    model_type: str = 'nafnet',
    scale: int = 2,
    input_shape: tuple = (1, 1, 256, 256),
    opset_version: int = 17,
    fp16: bool = False,
    dynamic_axes: bool = True
):
    """
    Export PyTorch model to ONNX format.

    Args:
        weights_path: Path to .pt checkpoint
        output_path: Path to save .onnx model
        model_type: 'nafnet' or 'nafnet_local'
        scale: Upscale factor (2 or 4)
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version
        fp16: Export as FP16
        dynamic_axes: Enable dynamic batch/height/width
    """
    device = torch.device('cpu')  # Export on CPU for compatibility

    print(f"Loading model from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        model = create_model_from_config(checkpoint['config'], model_type_default=model_type)
    else:
        model = create_model(model_type=model_type, scale=scale)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    if fp16:
        model.half()
        print("Converting to FP16")

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    if fp16:
        dummy_input = dummy_input.half()

    # Dynamic axes for variable input sizes
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'degraded': {0: 'batch', 2: 'height', 3: 'width'},
            'restored': {0: 'batch', 2: 'height', 3: 'width'}
        }

    print(f"Exporting to ONNX: {output_path}")
    print(f"Input shape: {input_shape}")
    print(f"Opset version: {opset_version}")
    print(f"Dynamic axes: {dynamic_axes}")

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['degraded'],
        output_names=['restored'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )

    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")

    # Print model info
    print(f"\nModel info:")
    print(f"  Input: {onnx_model.graph.input[0].name}")
    print(f"  Output: {onnx_model.graph.output[0].name}")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        print(f"  Input shape: {shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
        print(f"  Output shape: {shape}")

    print(f"\nONNX model saved to: {output_path}")


def benchmark_onnx(onnx_path: str, input_shape: tuple = (1, 1, 256, 256), num_runs: int = 100):
    """Benchmark ONNX model inference speed"""
    import onnxruntime as ort
    import numpy as np
    import time

    print(f"\nBenchmarking ONNX model on {ort.get_device()}...")

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Warmup
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Remove first few outliers
    times = times[5:]
    avg_time = sum(times) / len(times) * 1000  # ms
    fps = 1000 / avg_time

    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='Export NAFNet to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt weights file')
    parser.add_argument('--output', type=str, required=True, help='Output .onnx path')
    parser.add_argument('--model_type', type=str, default='nafnet', choices=['nafnet', 'nafnet_local'])
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help='Upscale factor')
    parser.add_argument('--input_h', type=int, default=256, help='Input height')
    parser.add_argument('--input_w', type=int, default=256, help='Input width')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true', help='Export as FP16')
    parser.add_argument('--static', action='store_true', help='Disable dynamic axes (fixed input size)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after export')
    parser.add_argument('--num_runs', type=int, default=100, help='Benchmark runs')

    args = parser.parse_args()

    input_shape = (1, 1, args.input_h, args.input_w)

    export_onnx(
        weights_path=args.weights,
        output_path=args.output,
        model_type=args.model_type,
        scale=args.scale,
        input_shape=input_shape,
        opset_version=args.opset,
        fp16=args.fp16,
        dynamic_axes=not args.static
    )

    if args.benchmark:
        benchmark_onnx(args.output, input_shape, args.num_runs)


if __name__ == '__main__':
    main()