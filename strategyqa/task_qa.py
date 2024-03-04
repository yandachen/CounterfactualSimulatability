import json
import sys
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task


def task_qa(model, expl_type, inputs):
    assert expl_type in ['cot', 'posthoc']
    # first deduplicate inputs, only run on different inputs
    distinct_qns = list(set([input['question'] for input in inputs]))
    distinct_inputs = [{'question': question} for question in distinct_qns]

    prompts = get_prompts_by_task(f'strategyqa-taskqa-{expl_type}',
                                  [{'question': input['question']} for input in distinct_inputs])
    pred_expls = multiprocess_api(model=model, prompts=prompts, bsz=8, num_processes=12,
                                 temperature=0, max_tokens=200, stop='\n\n')
    assert len(pred_expls) == len(prompts)
    if expl_type == 'cot':
        pred_answers = []
        for pred_expl in pred_expls:
            if pred_expl.endswith('So the answer is no.'):
                pred_answers.append('no')
            elif pred_expl.endswith('So the answer is yes.'):
                pred_answers.append('yes')
            else:
                pred_answers.append('neither')
        preds = [{'pred_ans': pred_ans, 'pred_expl': pred_expl.strip()} for pred_ans, pred_expl in
                 zip(pred_answers, pred_expls)]
    elif expl_type == 'posthoc':
        preds = []
        for pred_expl in pred_expls:
            lines = pred_expl.split('\n')
            if (len(lines) != 2) or (lines[0].strip() not in ['yes', 'no']) or (not lines[1].startswith('Justification: ')):
                preds.append({'pred_ans': 'neither', 'pred_expl': pred_expl})
            else:
                preds.append({'pred_ans': lines[0].strip(), 'pred_expl': lines[1][len('Justification: '):].strip()})
    else:
        raise NotImplementedError

    # return to duplicated questions
    assert len(preds) == len(distinct_inputs)
    qn2pred = {input['question']: pred for input, pred in zip(distinct_inputs, preds)}
    preds = [qn2pred[input['question']] for input in inputs]
    return preds


def task_qa_sim_inputs_list(model, expl_type, sim_inputs_list):
    all_sim_inputs = [input for sim_inputs in sim_inputs_list for input in sim_inputs]
    preds = task_qa(model, expl_type, all_sim_inputs)
    # regroup preds according to examples (multiple simulation inputs for each original input)
    example_preds = []
    cur = 0
    for ex_idx in range(len(sim_inputs_list)):
        example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
        cur += len(sim_inputs_list[ex_idx])
    assert cur == len(preds)
    return example_preds

