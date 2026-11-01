"""Small offline MPS timing/implementation check; random directions, no feature fit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 4, 8])
    ap.add_argument('--lengths', type=int, nargs='+', default=[5, 32])
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/timing')
    args = ap.parse_args()
    if min(args.batches + args.lengths + [args.repeats]) < 1:
        ap.error('Batch sizes, lengths, and repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ROOT / 'configs/local-assets.json').read_text())
    pin = inventory['model']
    assert torch.backends.mps.is_available(), 'This benchmark targets Apple MPS'
    device = torch.device('mps')
    torch.set_num_threads(6)
    torch.manual_seed(42)

    def sync():
        torch.mps.synchronize()

    def elapsed(fn):
        sync()
        start = time.perf_counter()
        value = fn()
        sync()
        return time.perf_counter() - start, value

    started = time.perf_counter()
    config = AutoConfig.from_pretrained(pin['id'], revision=pin['revision'],
                                        local_files_only=True).get_text_config()
    model = AutoModelForCausalLM.from_pretrained(
        pin['id'], revision=pin['revision'], config=config, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(pin['id'], revision=pin['revision'],
                                              local_files_only=True)
    sync()
    load_seconds = time.perf_counter() - started
    print(f'Model loaded in {load_seconds:.2f}s', flush=True)
    assert len(model.model.layers) == 32 and config.hidden_size == 2560
    layers = [8, 16, 26, 30]
    token_pool = tokenizer(
        'The notebook was lying open on the table beside a cup of tea. ' * 20,
        add_special_tokens=False)['input_ids']
    assert len(token_pool) >= max(args.lengths)
    rng = torch.Generator().manual_seed(42)
    random_directions = torch.randn(2, 2560, generator=rng).to(device)

    def forward(ids, angle_deg=None, capture=False):
        saved = {}
        checks = {}
        handles = []

        def intervention(module, inputs, output):
            assert torch.is_tensor(output), 'Expected a tensor decoder-block output'
            a = output[:, -1, :].float()
            saved['input'] = a.detach().clone()
            if angle_deg is None or angle_deg == 0:
                return output
            unit_a = a / a.norm(dim=-1, keepdim=True)
            u1 = random_directions[0] - (unit_a * random_directions[0]).sum(-1, keepdim=True) * unit_a
            u1 = u1 / u1.norm(dim=-1, keepdim=True)
            u2 = random_directions[1] - (unit_a * random_directions[1]).sum(-1, keepdim=True) * unit_a
            u2 = u2 - (u1 * u2).sum(-1, keepdim=True) * u1
            u2 = u2 / u2.norm(dim=-1, keepdim=True)
            alpha = math.radians(angle_deg)
            tangent = (u1 + u2) / math.sqrt(2)
            changed = math.cos(alpha) * a + math.sin(alpha) * a.norm(dim=-1, keepdim=True) * tangent
            result = output.clone()
            result[:, -1, :] = changed.to(output.dtype)
            saved['injected'] = result[:, -1, :].detach().float().clone()
            if capture:
                checks['norm_relative_error_fp32'] = ((changed.norm(dim=-1) / a.norm(dim=-1)) - 1).abs()
                checks['norm_relative_error_after_bf16'] = ((result[:, -1, :].float().norm(dim=-1) / a.norm(dim=-1)) - 1).abs()
            return result

        def recorder(layer):
            def record(module, inputs, output):
                if capture or layer == 30:
                    saved[layer] = output[:, -1, :].detach().float().clone()
            return record

        handles.append(model.model.layers[2].register_forward_hook(intervention))
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(recorder(layer)))
        try:
            # Full decoder pass preserves hybrid DeltaNet/attention execution.
            # Skip the large vocabulary projection; it is absent from the primary metric.
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return saved, checks

    results = []
    examples = {}
    with torch.inference_mode():
        for length in args.lengths:
            single_ids = torch.tensor([token_pool[:length]], device=device)
            # Warm up this sequence shape, then verify baseline and zero-angle hook.
            forward(single_ids)
            _, (clean, _) = elapsed(lambda: forward(single_ids, capture=True))
            _, (zero, _) = elapsed(lambda: forward(single_ids, 0, capture=True))
            zero_l2 = float((zero[30] - clean[30]).norm())
            assert zero_l2 == 0, f'Zero-angle hook changed activations: {zero_l2}'
            _, (perturbed, checks) = elapsed(lambda: forward(single_ids, 15, capture=True))
            ideal_error = float(checks['norm_relative_error_fp32'].max())
            bf16_error = float(checks['norm_relative_error_after_bf16'].max())
            assert ideal_error < 1e-5 and bf16_error < 0.005
            response = float((perturbed[30] - clean[30]).norm())
            assert math.isfinite(response) and response > 0
            examples[length] = {'ids': token_pool[:length], 'zero_angle_l2': zero_l2,
                                'random_15deg_l2': response,
                                'norm_relative_error_fp32': ideal_error,
                                'norm_relative_error_after_bf16': bf16_error}
            for batch in args.batches:
                ids = single_ids.expand(batch, -1).contiguous()
                forward(ids, 15, capture=True)
                samples = []
                for _ in range(args.repeats):
                    seconds, (saved, _) = elapsed(lambda: forward(ids, 15, capture=True))
                    samples.append(seconds)
                median = statistics.median(samples)
                entry = {'sequence_length': length, 'batch_size': batch,
                         'seconds_per_batch_samples': samples, 'seconds_per_batch_median': median,
                         'examples_per_second': batch / median,
                         'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                         'mps_driver_bytes': torch.mps.driver_allocated_memory()}
                results.append(entry)
                print(json.dumps(entry), flush=True)

        # Time selected-layer R/J++ readout separately from primary forwards.
        head = model.lm_head.weight.detach().float()
        gain = 1 + model.model.norm.weight.detach().float()
        eps = model.model.norm.eps
        excluded_ids = json.loads((Path(inventory['r_jpp_release']['directory']) / 'excluded_token_ids.json').read_text())
        excluded = torch.tensor(excluded_ids, device=device)
        readout_results = []
        for kind in ['r', 'jpp']:
            payload = torch.load(inventory['assets'][kind]['path'], map_location='cpu',
                                 weights_only=True, mmap=True)
            for layer in layers:
                matrix = payload['parameters']['jacobians'][layer].to(device)
                h = saved[layer]

                def readout():
                    z = h @ matrix.T
                    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
                    logits = (z * gain) @ head.T
                    logits[:, excluded] = -torch.inf
                    return logits.topk(5, dim=-1).indices

                readout()
                samples = [elapsed(readout)[0] for _ in range(args.repeats)]
                seconds = statistics.median(samples)
                ids = readout()[0].cpu().tolist()
                readout_results.append({'kind': kind, 'layer': layer, 'batch_size': h.shape[0],
                                        'seconds_median': seconds, 'samples': samples,
                                        'example_top5_token_ids': ids,
                                        'example_top5_tokens': [tokenizer.decode([i]) for i in ids]})
                del matrix
            del payload

    record = {'purpose': 'Timing and intervention sanity check only; random directions, no feature fit',
              'model': {'id': pin['id'], 'revision': pin['revision'], 'dtype': 'bfloat16'},
              'device': str(device), 'model_load_seconds': load_seconds,
              'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'numpy']},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'perturbation': {'block': 2, 'position': 'last token', 'mixing_angle_deg': 45,
                               'angle_deg': 15, 'directions': 'seed42 isotropic Gaussian projected and orthonormalized'},
              'measurement': {'block': 30, 'layer_convention': 'zero-based block output before final norm',
                              'metric': 'unsquared FP32 L2 of BF16 forward activations'},
              'prompt_format': 'plain text, add_special_tokens=False', 'examples': examples,
              'forward_timings': results, 'readout_timings': readout_results,
              'readout_policy': 'FP32 transport, offset RMSNorm, native tied head, release vocabulary filter',
              'limitations': ['Repeated copies of one anchor; workload may differ for independent anchors.',
                             'Two timing repeats by default; indicative throughput, not a stable benchmark.',
                             'Full forwards include the final decoder block and norm, but no LM head.',
                             'No O-lens, semantic directions, superellipse fit, or sweep in this run.']}
    (args.out / 'timing.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Saved {args.out / "timing.json"}', flush=True)


if __name__ == '__main__':
    main()
