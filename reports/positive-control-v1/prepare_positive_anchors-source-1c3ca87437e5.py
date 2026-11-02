"""Regenerate the released FineWeb anchor recipe for both pinned tokenizers."""
import hashlib,importlib.metadata,json,faulthandler
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/positive-control-v1'
protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())
RUN.mkdir(exist_ok=True)
if (RUN/'anchors.json').exists():
    raise SystemExit('Anchor file already exists; do not regenerate sealed inputs.')
tokenizers={m['key']:AutoTokenizer.from_pretrained(m['id'],revision=m['revision']) for m in protocol['models']}
d=protocol['dataset']
faulthandler.dump_traceback_later(45,repeat=False)
print('STREAM_START',d,flush=True)
ds=load_dataset(d['id'],name=d['config'],revision=d['revision'],split=d['split'],streaming=True)
print('DATASET_READY',flush=True)
ds=ds.shuffle(seed=d['shuffle_seed'],buffer_size=d['shuffle_buffer_size'])
print('SHUFFLE_READY',flush=True)
texts=[];rows=[]
for sample in ds:
    texts.append(sample['text']);rows.append({k:sample.get(k) for k in ['id','url','dump','token_count']})
    if len(texts)%100==0:print('STREAM_READ',len(texts),flush=True)
    if len(texts)==d['tokenizer_batch_size']:break
faulthandler.cancel_dump_traceback_later()
print('STREAM_BATCH',len(texts),flush=True)
encoded={key:tok(texts,truncation=True,max_length=protocol['anchor_tokens'],add_special_tokens=True,padding=False)['input_ids'] for key,tok in tokenizers.items()}
acceptance={key:[i for i,ids in enumerate(value) if len(ids)>=5][:30] for key,value in encoded.items()}
assert all(len(v)==30 for v in acceptance.values())
assert len({tuple(v) for v in acceptance.values()})==1,acceptance
indices=next(iter(acceptance.values()))
anchors=[]
for index in indices:
    anchors.append({'stream_index':index,'source':rows[index],'text':texts[index],
                    'text_sha256':hashlib.sha256(texts[index].encode()).hexdigest(),
                    'models':{key:{'token_ids':encoded[key][index][:5],
                                   'decoded_window':tok.decode(encoded[key][index][:5])} for key,tok in tokenizers.items()}})
result={'protocol_sha256':hashlib.sha256((ROOT/'configs/positive-control-v1.json').read_bytes()).hexdigest(),
        'dataset':d,'sampling':'Original streaming shuffle and 1000-text tokenization batch recipe.',
        'historical_identity':'Original authors token cache unavailable; source-recipe regeneration, not verified historical token identity.',
        'first_accepted_indices_by_model':acceptance,'anchors':anchors,
        'packages':{p:importlib.metadata.version(p) for p in ['datasets','transformers','huggingface_hub','numpy']}}
(RUN/'anchors.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('ANCHORS_SAVED',len(anchors),flush=True)
for i,a in enumerate(anchors):print(i,{k:v['decoded_window'] for k,v in a['models'].items()},flush=True)
