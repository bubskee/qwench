"""Build matched contrastive directions and validate held-out separation and steering."""
from __future__ import annotations

import argparse
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


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/direction-validation-v1.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/directions/v1')
    parser.add_argument('--stage', choices=['all', 'extract', 'causal', 'generate'], default='all')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    data_path = ROOT / protocol['data']
    data = json.loads(data_path.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    source_bytes = Path(__file__).read_bytes()
    (out / f'{args.stage}-source.py').write_bytes(source_bytes)
    seal = {'protocol_sha256': checksum(args.protocol), 'data_sha256': checksum(data_path)}
    if (out / 'input-seal.json').exists():
        assert json.loads((out / 'input-seal.json').read_text()) == seal, 'Inputs changed; use a new output directory/version'
    else:
        write_json(out / 'input-seal.json', seal)
    inventory = json.loads((ROOT / 'configs/local-assets.json').read_text())
    pin = inventory['model']
    torch.set_num_threads(6)
    torch.manual_seed(protocol['seed'])
    assert torch.backends.mps.is_available()
    device = 'mps'
    config = AutoConfig.from_pretrained(pin['id'], revision=pin['revision'], local_files_only=True).get_text_config()
    model = AutoModelForCausalLM.from_pretrained(
        pin['id'], revision=pin['revision'], config=config, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(pin['id'], revision=pin['revision'], local_files_only=True)
    block = protocol['block']

    def encode(text):
        return tokenizer(text, add_special_tokens=False)['input_ids']

    @torch.inference_mode()
    def forward(token_ids, direction=None, angle=0, position=-1, score_positions=None, orthogonal_to=None):
        ids = torch.tensor([token_ids], device=device)
        pos = position if position >= 0 else len(token_ids) + position
        capture = {}

        def hook(module, inputs, output):
            a = output[0, pos].float()
            capture['activation'] = a.detach().clone()
            if direction is None or angle == 0:
                return output
            unit = a / a.norm()
            tangent = direction.to(device) - (direction.to(device) @ unit) * unit
            if orthogonal_to is not None:
                first = orthogonal_to.to(device) - (orthogonal_to.to(device) @ unit) * unit
                first = first / first.norm()
                tangent = tangent - (tangent @ first) * first
            assert float(tangent.norm()) > 1e-6, 'Degenerate tangent direction'
            tangent = tangent / tangent.norm()
            alpha = math.radians(angle)
            changed = math.cos(alpha) * a + math.sin(alpha) * a.norm() * tangent
            edited = output.clone()
            edited[0, pos] = changed.to(output.dtype)
            capture['norm_error'] = ((edited[0, pos].float().norm() / a.norm()) - 1).abs()
            return edited

        handle = model.model.layers[block].register_forward_hook(hook)
        try:
            states = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        finally:
            handle.remove()
        logits = None
        if score_positions is not None:
            logits = model.lm_head(states[0, score_positions]).float()
        return capture, logits

    if args.stage in ['all', 'extract']:
        rows = []
        states = []
        for split in ['train', 'heldout']:
            for item in data[split]:
                for i, (language, sentiment) in enumerate([('en', 1), ('en', -1), ('es', 1), ('es', -1)]):
                    text = data['template'].format(sentence=item[i + 1])
                    ids = encode(text)
                    capture, _ = forward(ids)
                    states.append(capture['activation'].cpu())
                    rows.append({'split': split, 'item': item[0], 'language': language,
                                 'sentiment': sentiment, 'prompt': text, 'token_ids': ids, 'tokens': len(ids)})
                print(f'Extracted {split}/{item[0]} ({len(rows)} prompts)', flush=True)
        assert len({row['token_ids'][-1] for row in rows}) == 1, 'Final marker token differs'
        h = torch.stack(states)
        train = h[:len(data['train']) * 4].reshape(-1, 4, config.hidden_size)
        sentiment = ((train[:, 0] - train[:, 1]) + (train[:, 2] - train[:, 3])).mean(0) / 2
        language = ((train[:, 2] - train[:, 0]) + (train[:, 3] - train[:, 1])).mean(0) / 2
        directions = {'sentiment': sentiment / sentiment.norm(), 'language': language / language.norm()}
        save_file({**directions, 'training_centroid': train.mean((0, 1))}, str(out / 'directions.safetensors'))
        save_file({'activations': h}, str(out / 'activations.safetensors'))
        metrics = {}
        projection_rows = []
        for name, direction in directions.items():
            scores = (h - train.mean((0, 1))) @ direction
            labels = torch.tensor([row['sentiment'] if name == 'sentiment' else (1 if row['language'] == 'es' else -1) for row in rows])
            mask = torch.tensor([row['split'] == 'heldout' for row in rows])
            held = scores[mask].reshape(-1, 4)
            if name == 'sentiment':
                gaps = torch.stack([held[:, 0] - held[:, 1], held[:, 2] - held[:, 3]], dim=-1)
                subgroups = {'en': [0, 1], 'es': [2, 3]}
            else:
                gaps = torch.stack([held[:, 2] - held[:, 0], held[:, 3] - held[:, 1]], dim=-1)
                subgroups = {'positive': [0, 2], 'negative': [1, 3]}
            accuracy = float(((scores[mask] > 0) == (labels[mask] > 0)).float().mean())
            metrics[name] = {'heldout_accuracy': accuracy, 'heldout_pair_ordering': float((gaps > 0).float().mean()),
                             'heldout_pair_margins': gaps.tolist(), 'raw_direction_norm': float(sentiment.norm() if name == 'sentiment' else language.norm()),
                             'subgroup_accuracies': {group: float(((held[:, cols] > 0) == (labels[mask].reshape(-1, 4)[:, cols] > 0)).float().mean()) for group, cols in subgroups.items()},
                             'passes': accuracy >= protocol['min_heldout_accuracy'] and float((gaps > 0).float().mean()) >= protocol['min_heldout_pair_ordering']}
            for row, score in zip(rows, scores.tolist()):
                projection_rows.append({**row, 'direction': name, 'centered_projection': score})
        cosine = float(directions['sentiment'] @ directions['language'])
        geometry = []
        anchor_states = []
        for item_id, _, prompt in data['causal_anchors']:
            capture, _ = forward(encode(prompt))
            a = capture['activation'].cpu()
            anchor_states.append(a)
            a_unit = a / a.norm()
            projected = {name: vector - (vector @ a_unit) * a_unit for name, vector in directions.items()}
            first = projected['sentiment'] / projected['sentiment'].norm()
            second_before = projected['language'] / projected['language'].norm()
            second = second_before - (second_before @ first) * first
            second = second / second.norm()
            geometry.append({'item': item_id, 'projected_cosine_before_gram_schmidt': float(first @ second_before),
                             'language_rotation_degrees_from_projected': math.degrees(math.acos(float((second @ second_before).clamp(-1, 1)))),
                             'sentiment_cosine_raw_to_tangent': float(directions['sentiment'] @ first),
                             'language_cosine_raw_to_orthonormal_tangent': float(directions['language'] @ second),
                             'final_tangent_cosine': float(first @ second)})
        save_file({'anchor_activations': torch.stack(anchor_states)}, str(out / 'anchor-activations.safetensors'))
        length_summary = {f'{split}_{lang}_{label}': float(np.mean([r['tokens'] for r in rows if r['split'] == split and r['language'] == lang and r['sentiment'] == label])) for split in ['train', 'heldout'] for lang in ['en', 'es'] for label in [-1, 1]}
        result = {'model_revision': pin['revision'], 'block': block, 'convention': 'zero-based block output, last common marker token',
                  **seal, 'raw_cosine': cosine, 'overlap_passes': abs(cosine) < protocol['max_abs_raw_cosine'],
                  'metrics': metrics, 'mean_token_lengths': length_summary, 'anchor_geometry': geometry,
                  'last_token_id': rows[0]['token_ids'][-1], 'last_token_text': tokenizer.decode([rows[0]['token_ids'][-1]])}
        write_json(out / 'extraction.json', result)
        write_json(out / 'projections.json', projection_rows)
        print('EXTRACTION', json.dumps(result), flush=True)

    if args.stage in ['all', 'causal', 'generate']:
        tensors = load_file(str(out / 'directions.safetensors'))
        directions = {key: tensors[key] for key in ['sentiment', 'language']}
        for seed in protocol['random_control_seeds']:
            vector = torch.randn(config.hidden_size, generator=torch.Generator().manual_seed(seed))
            directions[f'random_{seed}'] = vector / vector.norm()

    if args.stage in ['all', 'causal']:
        conditions = [('baseline', None, 0)]
        for name in directions:
            angles = [x for x in protocol['angles_deg'] if x != 0] if not name.startswith('random') else [-30, 30]
            conditions += [(name, directions[name], angle) for angle in angles]
        continuations = {key: encode(value) for key, value in data['continuations'].items()}
        records = []
        for item_id, prompt_language, prompt in data['causal_anchors']:
            prompt_ids = encode(prompt)
            for name, direction, angle in conditions:
                logps = {}
                norm_errors = []
                for key, completion_ids in continuations.items():
                    assert encode(prompt + data['continuations'][key]) == prompt_ids + completion_ids, 'Token boundary changed'
                    ids = prompt_ids + completion_ids[:-1]
                    positions = list(range(len(prompt_ids) - 1, len(ids)))
                    orthogonal_to = directions['sentiment'] if name == 'language' and protocol.get('orthonormalize_for_steering') else None
                    capture, logits = forward(ids, direction, angle, len(prompt_ids) - 1, positions, orthogonal_to)
                    labels = torch.tensor(completion_ids, device=device)
                    values = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
                    logps[key] = float(values.sum())
                    if 'norm_error' in capture:
                        norm_errors.append(float(capture['norm_error']))
                def combine(keys):
                    return float(torch.logsumexp(torch.tensor([logps[k] for k in keys]), 0))
                sentiment_score = combine(['en_positive', 'es_positive']) - combine(['en_negative', 'es_negative'])
                language_score = combine(['es_positive', 'es_negative']) - combine(['en_positive', 'en_negative'])
                records.append({'item': item_id, 'prompt_language': prompt_language, 'prompt': prompt, 'condition': name, 'angle_deg': angle,
                                'completion_logps': logps, 'sentiment_logodds': sentiment_score, 'language_logodds': language_score,
                                'max_bf16_norm_error': max(norm_errors, default=0)})
            write_json(out / 'causal-records.json', records)
            print(f'Causal scores completed: {item_id}', flush=True)
        contrasts = []
        for item_id, _, _ in data['causal_anchors']:
            for name in directions:
                for magnitude in ([15, 30] if not name.startswith('random') else [30]):
                    plus = next(r for r in records if r['item'] == item_id and r['condition'] == name and r['angle_deg'] == magnitude)
                    minus = next(r for r in records if r['item'] == item_id and r['condition'] == name and r['angle_deg'] == -magnitude)
                    contrasts.append({'item': item_id, 'direction': name, 'magnitude_deg': magnitude,
                                      **{target: plus[target + '_logodds'] - minus[target + '_logodds'] for target in ['sentiment', 'language']}})
        summary = {}
        for target in ['sentiment', 'language']:
            by_direction = {}
            for name in directions:
                values = [r[target] for r in contrasts if r['direction'] == name and r['magnitude_deg'] == 30]
                by_direction[name] = {'median_plus_minus_logodds': float(np.median(values)), 'positive_anchor_count': sum(x > 0 for x in values), 'values': values}
            candidate = by_direction[target]
            summary[target] = {'directions': by_direction, 'passes': candidate['positive_anchor_count'] >= protocol['min_causal_correct_anchors'] and candidate['median_plus_minus_logodds'] > max(by_direction[f'random_{seed}']['median_plus_minus_logodds'] for seed in protocol['random_control_seeds'])}
        write_json(out / 'causal-contrasts.json', contrasts)
        write_json(out / 'causal-summary.json', summary)
        print('CAUSAL', json.dumps(summary), flush=True)

    if args.stage in ['all', 'generate']:
        outputs = []
        for item_id, _, prompt in data['causal_anchors']:
            if item_id not in protocol['generation_anchors']:
                continue
            conditions = [('baseline', None, 0)] + [(name, directions[name], angle) for name in ['sentiment', 'language'] for angle in [-30, 30]]
            for name, direction, angle in conditions:
                prompt_ids = encode(prompt)
                ids = list(prompt_ids)
                generated = []
                for _ in range(protocol['generation_max_new_tokens']):
                    orthogonal_to = directions['sentiment'] if name == 'language' and protocol.get('orthonormalize_for_steering') else None
                    _, logits = forward(ids, direction, angle, len(prompt_ids) - 1, [-1], orthogonal_to)
                    token = int(logits[0].argmax())
                    if token == tokenizer.eos_token_id:
                        break
                    generated.append(token)
                    ids.append(token)
                row = {'item': item_id, 'prompt': prompt, 'direction': name, 'angle_deg': angle,
                       'generated_ids': generated, 'text': tokenizer.decode(generated, skip_special_tokens=True),
                       'max_new_tokens': protocol['generation_max_new_tokens']}
                outputs.append(row)
                write_json(out / 'generations.json', outputs)
                print('GENERATION', json.dumps(row, ensure_ascii=False), flush=True)

    write_json(out / f'{args.stage}-receipt.json', {'stage': args.stage, **seal, 'script_sha256': hashlib.sha256(source_bytes).hexdigest(),
               'model_revision': pin['revision'], 'packages': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy', 'safetensors']},
               'note': 'No FSEC sweep or exponent fit in this script. Continuation scores use native model outputs, not lens judgments.'})


if __name__ == '__main__':
    main()
