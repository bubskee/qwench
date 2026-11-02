"""Verify fixed inputs, source receipts, numerical invariants and complete sweeps."""
import hashlib,json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/positive-control-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=json.loads((ROOT/'configs/positive-control-v1.json').read_text())
    seal=json.loads((RUN/'input-seal.json').read_text())
    assert seal['protocol_sha256']==sha(ROOT/'configs/positive-control-v1.json')
    assert seal['anchors_sha256']==sha(RUN/'anchors.json')
    assert seal['feature_bank_sha256']==sha(ROOT/'data/stress-features-v1.json')
    anchors=json.loads((RUN/'anchors.json').read_text())
    assert len(anchors['anchors'])==30
    for row in anchors['anchors']:
        assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
        assert all(len(row['models'][m['key']]['token_ids'])==5 for m in protocol['models'])
    models=[];measurements=0;all_complete=True
    for model in protocol['models']:
        folder=RUN/model['key'];prepared=json.loads((folder/'prepared.json').read_text())
        assert prepared['seal']==seal and prepared['model']==model
        assert sha(folder/'directions.safetensors')==prepared['directions_sha256']
        assert sha(folder/'fixed-prefixes.safetensors')==prepared['prefixes_sha256']
        vectors=load_file(str(folder/'directions.safetensors'))
        assert all(np.isfinite(v).all() and abs(np.linalg.norm(v)-1)<1e-5 for v in vectors.values())
        validation=prepared['validation']
        for key in ['stop_full_layer_2','stop_full_layer_'+str(model['measure_layer']),'fixed_prefix_replay_vs_native']:
            assert validation[key]==0
        if model['key']=='qwen17':
            assert all(r['max_absolute_error']==0 for r in validation['original_suffix_parity'])
            assert min(prepared['original_direction_cosines'].values())>.9999
        for direction in prepared['directions'].values():assert direction['released_prompt_hashes_match']
        rows=[]
        for pair in prepared['pairs']:
            assert abs(pair['raw_cosine'])<.1 and pair['eligible']
            path=folder/(pair['id']+'.npz');meta=path.with_suffix('.json')
            if not(path.exists() and meta.exists()):all_complete=False;continue
            data=np.load(path);receipt=json.loads(meta.read_text())
            if not receipt['complete']:all_complete=False;continue
            assert receipt['seal']==seal and receipt['pair']==pair
            assert receipt['source_sha256']==sha(RUN/('sweep-source-'+receipt['source_sha256'][:12]+'.py'))
            assert data['l2_paper'].shape==(30,31,31) and data['l2_matched'].shape==(30,31,31)
            assert np.array_equal(data['angles_deg'],np.arange(0,61,2))
            assert data['completed_anchors'].all()
            for key in ['l2_paper','l2_matched']:assert np.isfinite(data[key]).all() and (data[key]>=0).all()
            assert np.all(data['l2_matched'][:,0,0]==0) and receipt['matched_origin_max']==0
            assert data['norm_error'].max()<.005 and receipt['norm_error_max']<.005
            assert np.all(data['seconds']>0)
            assert np.isclose(data['l2_paper'][:,0,0].max(),receipt['paper_origin_max'])
            measurements+=data['l2_paper'].size
            rows.append({'pair':pair['id'],'cases':int(data['l2_paper'].size),
                         'matched_origin_max':receipt['matched_origin_max'],
                         'paper_origin_max':receipt['paper_origin_max'],
                         'norm_error_max':receipt['norm_error_max'],'seconds':receipt['seconds'],
                         'raw_sha256':sha(path)})
        done=(folder/'COMPLETE.json').exists();all_complete &= done
        models.append({'model':model['key'],'complete':done,'complete_surfaces':len(rows),'surfaces':rows,
                       'original_direction_cosines':prepared['original_direction_cosines']})
    if all_complete:
        assert measurements==protocol['nominal_measurements']
        assert sum(m['complete_surfaces'] for m in models)==6
    result={'complete':all_complete,'all_assertions_passed':True,'models':models,
            'fresh_perturbation_cases':measurements,'expected_cases':protocol['nominal_measurements'],
            'reference_modes':'Two L2 readings per native perturbed output; cases are not double counted.',
            'historical_anchor_identity_verified':False,'fresh_source_documents_shared':True,
            'scope':'Numerical and artifact checks; not validation of semantic purity or error correction.'}
    target=RUN/'analysis/quality-audit.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
