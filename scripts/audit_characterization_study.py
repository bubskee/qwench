"""Combined saved-artifact/derived-metric audit using only NumPy and stdlib."""
from pathlib import Path
import hashlib
import json
import statistics as S
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
OUT = REPORTS / 'toy-characterization-study-v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path): return json.loads(path.read_text())


def main():
    summaries=load(OUT/'networks.json');assert len(summaries)==810
    families={'orthogonal':'toy-characterization-v1','coupled':'toy-characterization-coupled-v1'}
    audits={};points=0;surfaces=0;flagged=0;invalid=0;max_summary_error=0.;max_secondary_error=0.
    for family,folder in families.items():
        source=REPORTS/folder;audit=load(source/'quality-audit.json');assert audit['all_assertions_passed']
        assert audit['source_sha256']==sha(ROOT/'scripts/audit_characterization.py')
        audits[family]=audit;points+=audit['response_points'];surfaces+=audit['aggregate_surfaces']
        flagged+=audit['primary_fits_flagged'];invalid+=audit['primary_fits_invalid']
        assets=dict(np.load(source/'assets.npz'));a=assets['encoder'];projection=a@np.linalg.inv(a.T@a)@a.T
        assert np.abs(projection@a-a).max()<1e-12
        for path in (source/'analysis').glob('*/*.json'):
            record=load(path);meta=record['network'];label=family if meta['track']=='new' else 'historical'
            assert record['dependencies']['source_sha256']==sha(ROOT/'scripts/analyze_characterization.py')
            assert record['dependencies']['fit_helper_sha256']==sha(ROOT/'scripts/analyze_toy_cis.py')
            summary=next(row for row in summaries if row['family']==label and row['key']==meta['key'])
            for which in ['natural','separable']:
                ps=[row['fits']['0.5']['p'] for row in record['geometry'] if row['map']==which and row['kind']=='feature_axes' and row['anchor']==0 and row['sign']==[1,1] and row['fits']['0.5']['p'] is not None]
                expected=S.median(ps) if ps else None
                assert expected==summary[which+'_p']
                sigma=.075 if label=='historical' else .35
                score=next(row for row in record['recovery'] if row['map']==which and row['sigma']==sigma)
                for metric in ['all_mse','active_mse','inactive_mse']:
                    max_summary_error=max(max_summary_error,abs(score[metric]-summary[which+'_'+metric]))
        diag=load(REPORTS/'toy-characterization-diagnostics-v1'/(family+'.json'))
        assert diag['source_sha256']==sha(ROOT/'scripts/diagnose_characterization.py')
        assert diag['model_frames']==120 and len(diag['records'])==2880
        assert diag['max_feature_plane_output_change_under_projection']<1e-9
        diag_raw=REPORTS/'toy-characterization-diagnostics-v1'/(family+'-errors.npz')
        assert sha(diag_raw)==diag['raw_sha256']
        with np.load(diag_raw) as raw:
            for row in diag['records']:
                li=[0,.15,.35,.6].index(row['sigma']);prefix=row['key']+f'_{li}_{row["map"]}_'
                for metric in ['all','active','inactive']:
                    expected=float(raw[prefix+metric+'_sum'].sum()/raw[prefix+metric+'_count'].sum())
                    max_secondary_error=max(max_secondary_error,abs(expected-row[metric+'_mse']))
        axis=load(REPORTS/'toy-characterization-axis-null-v1'/(family+'.json'))
        assert axis['all_control_identity_assertions_passed'] and axis['separable_max_abs_grid_identity_error']<1e-8
        assert axis['source_sha256']==sha(ROOT/'scripts/axis_null_characterization.py')
        assert len(axis['records'])==(1800 if family=='orthogonal' else 1440)
        pair=load(REPORTS/'toy-characterization-pair-diagnostic-v1'/(family+'.json'))
        assert pair['all_control_additivity_checks_passed'] and len(pair['records'])==480
        assert pair['source_sha256']==sha(ROOT/'scripts/pair_diagnostic_characterization.py')
        assert pair['existing_surface_reproduction_max_abs_error']<1e-8
        path=REPORTS/'toy-characterization-pair-diagnostic-v1'/(family+'.npz')
        assert sha(path)==pair['raw_sha256']
        with np.load(path) as raw:
            for key in ['response','vector_null']:
                assert raw[key].shape==(480,4,31,31) and np.isfinite(raw[key]).all() and (raw[key]>=0).all()
                assert raw[key][...,0,0].max()<1e-8
        validation=load(REPORTS/'toy-characterization-validation-v1'/(family+'.json'))
        assert validation['all_control_checks_passed'] and len(validation['recovery'])==480 and len(validation['geometry'])==240
        assert validation['source_sha256']==sha(ROOT/'scripts/validate_characterization.py')
        path=REPORTS/'toy-characterization-validation-v1'/(family+'-errors.npz')
        assert sha(path)==validation['raw_errors_sha256']
        with np.load(path) as raw:
            for row in validation['recovery']:
                prefix=row['network']['key']+'_'+row['map']+'_'
                for metric in ['all','active','inactive']:
                    expected=float(raw[prefix+metric+'_sum'].sum()/raw[metric+'_count'].sum())
                    max_secondary_error=max(max_secondary_error,abs(expected-row[metric+'_mse']))
        path=REPORTS/'toy-characterization-validation-v1'/(family+'-geometry.npz')
        assert sha(path)==validation['raw_geometry_sha256']
        with np.load(path) as raw:
            assert raw['response'].shape==(240,4,31,31) and np.isfinite(raw['response']).all()
            assert raw['response'][...,0,0].max()<1e-8
    assert max_summary_error<1e-12 and max_secondary_error<1e-12
    initial=load(REPORTS/'toy-characterization-v1/preflight-checks.json')
    coupled=load(REPORTS/'toy-characterization-coupled-v1/preflight-checks.json')
    assert initial['all_assertions_passed'] and coupled['all_assertions_passed']
    assert initial['source_sha256']==sha(ROOT/'scripts/check_characterization.py')
    assert coupled['runtime_sha256']==sha(ROOT/'scripts/coupled_characterization.py')
    oracle_path=REPORTS/'toy-characterization-oracle-geometry-v1/analysis.json';oracle=load(oracle_path)
    assert len(oracle['records'])==8 and oracle['source_sha256']==sha(ROOT/'scripts/oracle_geometry_characterization.py')
    assert sha(oracle_path.with_name('responses.npz'))==oracle['raw_sha256']
    fine=load(OUT/'fine-checks.json');assert len(fine)==80
    result={'all_assertions_passed':True,'fresh_trained_networks':120,'historical_networks_reused':30,
            'total_checkpoint_frames':810,'primary_surfaces':surfaces,'response_points_including_fine':points,
            'primary_flagged_fits_including_invalid':flagged,'primary_invalid_fits':invalid,
            'fine_check_count':80,'fine_max_abs_delta_p':max(abs(r['delta_p']) for r in fine if r['delta_p'] is not None),
            'max_table_vs_source_error':max_summary_error,'max_secondary_metric_reconstruction_error':max_secondary_error,
            'independent_MPS_NumPy_forward_max_error':max(r['independent_numpy_vs_saved_MPS_prediction_max_error'] for r in audits.values()),
            'exact_posterior_enumeration_max_error':max(initial['Bayes_vs_full_65536_latent_likelihood_enumeration_max_error'],coupled['coupled_posterior_vs_65536_state_enumeration_max_error']),
            'source_sha256':sha(__file__), 'scope':'Numerical/provenance/derived-value checks; fit failures retained and not promoted as scientific successes.'}
    (OUT/'quality-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
