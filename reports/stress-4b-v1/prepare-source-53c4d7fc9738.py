"""Prospective fresh-model FSEC stress test. Preparation never selects on fitted p."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


class CaptureComplete(Exception):
    pass


class Engine:
    def __init__(self, pin, layer=2, measure=30):
        self.pin, self.layer, self.measure = pin, layer, measure
        torch.set_num_threads(6)
        torch.manual_seed(42)
        assert torch.backends.mps.is_available()
        config = AutoConfig.from_pretrained(pin['id'], revision=pin['revision'], local_files_only=True).get_text_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            pin['id'], revision=pin['revision'], config=config, dtype=torch.bfloat16,
            attn_implementation='eager', local_files_only=True).to('mps').eval()
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(pin['id'], revision=pin['revision'], local_files_only=True)
        self.baselines = {}
        self.dtype_name = 'bfloat16'

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)['input_ids']

    @torch.inference_mode()
    def capture(self, ids, stop_at, full=False):
        saved = {}
        def hook(module, inputs, output):
            assert torch.is_tensor(output)
            saved['value'] = output[:, -1].detach().float().clone()
            if not full:
                raise CaptureComplete
        handle = self.model.model.layers[stop_at].register_forward_hook(hook)
        try:
            try:
                self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            except CaptureComplete:
                pass
        finally:
            handle.remove()
        assert 'value' in saved
        return saved['value']

    def validate_capture(self, token_ids):
        ids = torch.tensor([token_ids], device='mps')
        result = {}
        for layer in [self.layer, self.measure]:
            stopped = self.capture(ids, layer)
            full = self.capture(ids, layer, full=True)
            error = float((stopped - full).abs().max())
            assert error == 0, (layer, error)
            result[str(layer)] = error
        return result

    def set_dtype(self, name):
        if name != self.dtype_name:
            self.model.to(torch.float32 if name == 'float32' else torch.bfloat16)
            self.dtype_name = name
            self.baselines.clear()
            torch.mps.empty_cache()

    @torch.inference_mode()
    def sweep_batch(self, token_ids, direction1, direction2, coordinates):
        batch = len(coordinates)
        ids = torch.tensor([token_ids], device='mps').expand(batch, -1).contiguous()
        key = (tuple(token_ids), batch, self.dtype_name)
        if key not in self.baselines:
            self.baselines[key] = self.capture(ids, self.measure)
        clean = self.baselines[key]
        d1, d2 = direction1.to('mps'), direction2.to('mps')
        alpha = torch.deg2rad(torch.tensor(coordinates, device='mps', dtype=torch.float32))
        radius = alpha.square().sum(-1).sqrt()
        norm_errors = []
        saved = {}
        def inject(module, inputs, output):
            a = output[:, -1].float()
            unit = a / a.norm(dim=-1, keepdim=True)
            u = d1 - (unit * d1).sum(-1, keepdim=True) * unit
            u = u / u.norm(dim=-1, keepdim=True)
            v = d2 - (unit * d2).sum(-1, keepdim=True) * unit
            v = v - (u * v).sum(-1, keepdim=True) * u
            assert float(v.norm(dim=-1).min()) > 1e-6
            v = v / v.norm(dim=-1, keepdim=True)
            safe = torch.where(radius > 1e-12, radius, torch.ones_like(radius))
            tangent = (alpha[:, :1] * u + alpha[:, 1:] * v) / safe[:, None]
            changed = torch.cos(radius)[:, None] * a + torch.sin(radius)[:, None] * a.norm(dim=-1, keepdim=True) * tangent
            changed = torch.where((radius == 0)[:, None], a, changed)
            result = output.clone()
            result[:, -1] = changed.to(output.dtype)
            norm_errors.append(((result[:, -1].float().norm(dim=-1) / a.norm(dim=-1)) - 1).abs())
            return result
        def measure(module, inputs, output):
            saved['response'] = (output[:, -1].float() - clean).norm(dim=-1)
            raise CaptureComplete
        handles = [self.model.model.layers[self.layer].register_forward_hook(inject),
                   self.model.model.layers[self.measure].register_forward_hook(measure)]
        try:
            try:
                self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            except CaptureComplete:
                pass
        finally:
            for handle in handles:
                handle.remove()
        response = saved['response'].cpu().numpy()
        error = float(torch.cat(norm_errors).max())
        assert np.isfinite(response).all() and error < .005
        zero = np.linalg.norm(np.asarray(coordinates), axis=1) == 0
        zero_error = float(response[zero].max()) if zero.any() else 0.
        assert zero_error == 0, ('zero-angle batch parity failed', batch, zero_error)
        return response, error


def prepare(engine, protocol, seal, out):
    feature_data = json.loads((ROOT / protocol['feature_data']).read_text())
    features = feature_data['directions']
    corpus = json.loads(Path(protocol['corpus_path']).read_text())
    assert sha(protocol['corpus_path']) == protocol['corpus_sha256']
    assert sha(protocol['corpus_provenance_path']) == protocol['corpus_provenance_sha256']
    rng = np.random.default_rng(protocol['anchor_seed'])
    order = rng.permutation(len(corpus))
    anchor_indices = list(map(int, order[:protocol['anchor_count']]))
    primary = [{'id': f'dclm_{i:03d}', 'corpus_row': i, 'token_ids': engine.encode(corpus[i])[:protocol['anchor_tokens']]} for i in anchor_indices]
    control_indices = list(map(int, order[protocol['anchor_count']:]))
    pool_ids = [engine.encode(corpus[i])[:protocol['anchor_tokens']] for i in control_indices]
    assert all(len(ids) == 5 for ids in pool_ids + [x['token_ids'] for x in primary])
    challenges = [{'id': f'challenge_{i}', 'text': text, 'token_ids': engine.encode(text)[:5]} for i, text in enumerate(protocol['challenge_anchors'])]
    for row in primary + challenges:
        row['decoded_window'] = engine.tokenizer.decode(row['token_ids'])
    # Group by exact token length so no padding enters extraction.
    texts = sorted({text for f in features for side in ['a', 'b'] for text in f[side]})
    encoded = {text: engine.encode(text) for text in texts}
    groups = defaultdict(list)
    for text, ids in encoded.items():
        groups[len(ids)].append(text)
    states = {}
    for length, strings in sorted(groups.items()):
        for start in range(0, len(strings), protocol['batch_size']):
            chunk = strings[start:start + protocol['batch_size']]
            ids = torch.tensor([encoded[s] for s in chunk], device='mps')
            values = engine.capture(ids, protocol['direction_layer']).cpu()
            for text, value in zip(chunk, values):
                states[text] = value
        print(f'Extracted prompt length {length}: {len(states)}/{len(texts)} unique prompts', flush=True)
    directions = {}
    feature_meta = {}
    raw_sides = {}
    for feature in features:
        name = feature['name']
        a, b = [torch.stack([states[s] for s in feature[side]]) for side in ['a', 'b']]
        vector = (b - a).mean(0)
        assert torch.isfinite(vector).all() and vector.norm() > 1e-8
        directions[name] = vector / vector.norm()
        feature_meta[name] = {key: feature[key] for key in ['family', 'source', 'sign', 'pairs']}
        feature_meta[name]['raw_norm'] = float(vector.norm())
        raw_sides[name] = (a, b)
    pool = []
    for start in range(0, len(pool_ids), protocol['batch_size']):
        ids = torch.tensor(pool_ids[start:start + protocol['batch_size']], device='mps')
        pool.append(engine.capture(ids, protocol['direction_layer']).cpu())
    pool = torch.cat(pool)
    control_pairs, controls_meta = [], {}
    for seed_index, seed in enumerate(protocol['control_seeds']):
        for family in ['isotropic_random', 'random_activation_difference_mean30', 'same_prompt_random_relabelling']:
            names = []
            for axis in range(2):
                name = f'{family}_{seed}_{axis}'
                generator = torch.Generator().manual_seed(seed * 100 + axis)
                meta = {'family': family, 'seed': seed, 'axis': axis}
                if family == 'isotropic_random':
                    vector = torch.randn(2560, generator=generator)
                elif family == 'random_activation_difference_mean30':
                    indices = torch.randperm(len(pool), generator=generator)[:60]
                    vector = (pool[indices[30:]] - pool[indices[:30]]).mean(0)
                    meta['pool_row_indices'] = indices.tolist()
                else:
                    source = protocol['label_shuffle_sources'][seed_index][axis]
                    a, b = raw_sides[source]
                    h = torch.cat([a, b]); n = len(a)
                    permutation = torch.randperm(len(h), generator=generator)
                    vector = h[permutation[n:]].mean(0) - h[permutation[:n]].mean(0)
                    meta.update(source_feature=source, permutation=permutation.tolist(), group_size=n)
                assert vector.norm() > 1e-8
                directions[name] = vector / vector.norm()
                controls_meta[name] = meta
                names.append(name)
            control_pairs.append({'a': names[0], 'b': names[1], 'kind': 'control', 'stratum': family, 'reasons': ['fixed_seed_control']})
    names = sorted(feature_meta)
    eligible, rejected = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cos = float(directions[a] @ directions[b])
            families = sorted([feature_meta[a]['family'], feature_meta[b]['family']])
            row = {'a': a, 'b': b, 'cosine': cos, 'stratum': '×'.join(families),
                   'kind': 'novel' if 'novel' in families else 'paper'}
            (eligible if abs(cos) < protocol['max_abs_cosine'] else rejected).append(row)
    eligible_map = {frozenset([r['a'], r['b']]): r for r in eligible}
    chosen = {}
    fixed_excluded = []
    def add(row, reason):
        key = frozenset([row['a'], row['b']])
        if key not in chosen:
            chosen[key] = {**row, 'reasons': []}
        chosen[key]['reasons'].append(reason)
    for pair in protocol['fixed_pairs']:
        key = frozenset(pair)
        if key in eligible_map:
            add(eligible_map[key], 'fixed_challenge')
        else:
            fixed_excluded.append({'pair': pair, 'cosine': float(directions[pair[0]] @ directions[pair[1]])})
    groups = defaultdict(list)
    for row in eligible:
        groups[row['stratum']].append(row)
    rng = np.random.default_rng(protocol['selection_seed'])
    for label, rows in sorted(groups.items()):
        quota = protocol['novel_pairs_per_stratum'] if 'novel' in label else protocol['paper_pairs_per_stratum']
        for index in rng.permutation(len(rows))[:quota]:
            add(rows[int(index)], 'stratified_sample')
    possible = {n for row in eligible for n in [row['a'], row['b']]}
    covered = {n for row in chosen.values() for n in [row['a'], row['b']]}
    for index in rng.permutation(len(eligible)):
        if len(chosen) >= protocol['max_feature_pairs'] or covered >= possible:
            break
        row = eligible[int(index)]
        if not {row['a'], row['b']} <= covered:
            add(row, 'feature_coverage')
            covered.update([row['a'], row['b']])
    pairs = [{**row, 'id': f'feature_{i:03d}'} for i, row in enumerate(chosen.values())]
    control_excluded = []
    for row in control_pairs:
        row['cosine'] = float(directions[row['a']] @ directions[row['b']])
        if abs(row['cosine']) >= protocol['max_abs_cosine']:
            control_excluded.append(row)
        else:
            pairs.append({**row, 'id': f'control_{len(pairs):03d}'})
    save_file(directions, str(out / 'directions.safetensors'))
    save_file({'activations': pool}, str(out / 'control-pool.safetensors'))
    validations = engine.validate_capture(primary[0]['token_ids'])
    challenge_keys = {frozenset(pair) for pair in protocol['challenge_pairs']}
    tasks = []
    def task(pair, tag, angles, anchors, signs=(1, 1), dtype='bfloat16'):
        tasks.append({'id': pair['id'] + '_' + tag, 'pair_id': pair['id'], 'tag': tag,
                      'a': pair['a'], 'b': pair['b'], 'kind': pair['kind'], 'stratum': pair['stratum'],
                      'angles_deg': angles, 'anchor_bank': anchors, 'signs': list(signs), 'dtype': dtype})
    feature_pairs = [p for p in pairs if p['kind'] != 'control']
    null_pairs = [p for p in pairs if p['kind'] == 'control']
    # Interleave initial controls with features to reduce order/thermal confounding.
    for i in range(max(len(feature_pairs), len(null_pairs))):
        for collection in [feature_pairs, null_pairs]:
            if i < len(collection):
                task(collection[i], 'primary', protocol['angles_deg'], list(range(len(primary))))
    for pair in feature_pairs:
        if frozenset([pair['a'], pair['b']]) in challenge_keys:
            task(pair, 'challenge_anchors', protocol['angles_deg'], list(range(len(primary), len(primary) + len(challenges))))
            for signs in protocol['sign_check_quadrants']:
                task(pair, f'signs_{signs[0]}_{signs[1]}', protocol['angles_deg'], protocol['sign_check_primary_anchor_indices'], signs)
    fine_feature = next((p for p in feature_pairs if 'fixed_challenge' in p['reasons']), feature_pairs[0])
    fine_control = next(p for p in null_pairs if p['stratum'] == 'isotropic_random')
    for pair in [fine_feature, fine_control]:
        task(pair, 'fine_grid', protocol['fine_angles_deg'], list(range(len(primary))))
        task(pair, 'fp32', protocol['angles_deg'], list(range(len(primary))), dtype='float32')
    selected = {'seal': seal, 'feature_metadata': feature_meta, 'control_metadata': controls_meta,
                'pairs': pairs, 'eligible_feature_pair_count': len(eligible), 'rejected_feature_pairs': rejected,
                'fixed_pairs_excluded': fixed_excluded, 'controls_excluded': control_excluded,
                'feature_names_covered': sorted(covered), 'feature_names_uncovered': sorted(set(names) - covered),
                'primary_anchors': primary, 'challenge_anchors': challenges, 'control_pool_corpus_rows': control_indices,
                'capture_validation_max_abs_error': validations, 'unique_extraction_prompts': len(states),
                'tasks': tasks, 'planned_examples': sum(len(t['anchor_bank']) * len(t['angles_deg']) ** 2 for t in tasks)}
    write(out / 'plan.json', selected)
    print('PREPARED', json.dumps({'feature_directions': len(feature_meta), 'feature_pairs': len(feature_pairs),
          'control_pairs': len(null_pairs), 'covered': len(covered), 'excluded_fixed': fixed_excluded,
          'tasks': len(tasks), 'planned_examples': selected['planned_examples']}), flush=True)


def sweep(engine, protocol, seal, out, source_hash):
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['seal'] == seal
    vectors = load_file(str(out / 'directions.safetensors'))
    anchors = plan['primary_anchors'] + plan['challenge_anchors']
    results = out / 'surfaces'; results.mkdir(exist_ok=True)
    start = time.perf_counter(); completed_examples = 0
    for task_index, task in enumerate(plan['tasks']):
        target = results / (task['id'] + '.npz')
        metadata_path = target.with_suffix('.json')
        if target.exists() and metadata_path.exists():
            prior = json.loads(metadata_path.read_text())
            assert prior['seal'] == seal and prior['task'] == task
            print('Already complete:', task['id'], flush=True)
            continue
        engine.set_dtype(task['dtype'])
        axis = task['angles_deg']
        coordinates = [(x, y) for x in axis for y in axis]
        response_grid = np.zeros((len(task['anchor_bank']), len(axis), len(axis)), dtype=np.float32)
        d1 = vectors[task['a']] * task['signs'][0]
        d2 = vectors[task['b']] * task['signs'][1]
        begin = time.perf_counter(); max_norm_error = 0.
        for anchor_i, index in enumerate(task['anchor_bank']):
            values = []
            for chunk_start in range(0, len(coordinates), protocol['batch_size']):
                chunk = coordinates[chunk_start:chunk_start + protocol['batch_size']]
                responses, error = engine.sweep_batch(anchors[index]['token_ids'], d1, d2, chunk)
                values.extend(responses.tolist()); max_norm_error = max(max_norm_error, error)
            response_grid[anchor_i] = np.array(values).reshape(len(axis), len(axis))
        with target.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, angles_deg=np.array(axis), l2=response_grid)
        target.with_suffix('.tmp').replace(target)
        duration = time.perf_counter() - begin
        completed_examples += response_grid.size
        write(metadata_path, {'seal': seal, 'task': task, 'source_sha256': source_hash,
                              'seconds': duration, 'max_relative_norm_error': max_norm_error,
                              'zero_angle_l2_max': float(response_grid[:, 0, 0].max())})
        progress = {'completed_tasks': task_index + 1, 'total_tasks': len(plan['tasks']),
                    'last_task': task['id'], 'elapsed_seconds_this_run': time.perf_counter() - start,
                    'examples_per_second_this_run': completed_examples / (time.perf_counter() - start)}
        write(out / 'progress.json', progress)
        print('COMPLETE', json.dumps(progress), flush=True)
    write(out / 'COMPLETE.json', {'seal': seal, 'tasks': len(plan['tasks']), 'source_sha256': source_hash})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'sweep'], required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/stress-4b-v1.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/stress-4b-v1')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    inventory = json.loads((ROOT / 'configs/local-assets.json').read_text())
    seal = {'protocol_sha256': sha(args.protocol), 'feature_data_sha256': sha(ROOT / protocol['feature_data']),
            'model_id': inventory['model']['id'], 'model_revision': inventory['model']['revision']}
    args.out.mkdir(parents=True, exist_ok=True)
    sealed = args.out / 'input-seal.json'
    if sealed.exists():
        assert json.loads(sealed.read_text()) == seal
    else:
        write(sealed, seal)
    source_bytes = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    (args.out / f'{args.stage}-source-{source_hash[:12]}.py').write_bytes(source_bytes)
    engine = Engine(inventory['model'], protocol['direction_layer'], protocol['measurement_layer'])
    if args.stage == 'prepare':
        prepare(engine, protocol, seal, args.out)
    else:
        sweep(engine, protocol, seal, args.out, source_hash)
    write(args.out / f'{args.stage}-receipt.json', {'seal': seal, 'source_sha256': source_hash,
          'packages': {name: importlib.metadata.version(name) for name in ['torch','transformers','numpy','safetensors']}})


if __name__ == '__main__':
    main()
