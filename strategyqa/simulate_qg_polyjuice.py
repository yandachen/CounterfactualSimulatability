import pandas as pd
from tqdm import tqdm
import json
import pickle as pkl
import sys
sys.path.append('../../')
from polyjuice import Polyjuice


def pj_generate_simqns(orig_qns, top_p, num_samples, ctrl_code):
    pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True)
    sim_qns = []
    for question in tqdm(orig_qns):
        perturb_texts = pj.perturb(question, ctrl_code=ctrl_code,
                                   num_perturbations=num_samples, top_p=top_p, do_sample=True)
        sim_qns.append([{'question': qn} for qn in set(perturb_texts)])
    return sim_qns


if __name__ == '__main__':
    test_data = json.load(open('../data/data.json'))['test'][:400]
    control_code2exidx2simqns = {}
    for control_code in ["negation", "lexical", "resemantic", "insert"]:
        control_code2exidx2simqns[control_code] = {}
        orig_qns = [ex['question'] for ex in test_data]
        sim_qns = pj_generate_simqns(orig_qns=orig_qns, ctrl_code=control_code,
                                     num_samples=10, top_p=1.0)
        for exidx in range(len(sim_qns)):
            control_code2exidx2simqns[control_code][exidx] = sim_qns[exidx]
    control_code2exidx2simqns['all'] = {}
    for exidx in range(400):
        control_code2exidx2simqns['all'][exidx] = [qn for control_code in ["negation", "lexical", "resemantic", "insert"]
                                                   for qn in control_code2exidx2simqns[control_code][exidx]]
    # pkl.dump(control_code2exidx2simqns, open(f'../outputs/simqg_pj.pkl', 'wb'))
