import json
import sys
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task


def task_qa(model, expl_type, inputs):
    assert expl_type in ['cot', 'posthoc']
    prompts = get_prompts_by_task(f'shp-taskqa-{expl_type}',
                [{'context': input['context'],
                  'response_0': input['options'][0], 'response_1': input['options'][1]}
                 for input in inputs])
    deduplicated_prompts = list(set(prompts))
    responses = multiprocess_api(model=model, prompts=deduplicated_prompts, bsz=20, num_processes=12,
                                 temperature=0, max_tokens=200, stop='\n\n')
    assert len(responses) == len(deduplicated_prompts)
    prompt2response = {prompt: response for prompt, response in zip(deduplicated_prompts, responses)}
    responses = [prompt2response[prompt] for prompt in prompts]
    assert len(responses) == len(inputs)
    answers = []
    if expl_type == 'cot':
        for response in responses:
            select0 = "Candidate Response 1 is more helpful" in response
            select1 = "Candidate Response 2 is more helpful" in response
            if select0 == select1:
                answers.append({'pred_ans': None, 'pred_expl': response})
            elif select0:
                answers.append({'pred_ans': 0, 'pred_expl': response})
            elif select1:
                answers.append({'pred_ans': 1, 'pred_expl': response})
    elif expl_type == 'posthoc':
        for response in responses:
            if response.startswith('Candidate Response 1 is more helpful'):
                answers.append({'pred_ans': 0, 'pred_expl': response})
            elif response.startswith('Candidate Response 2 is more helpful'):
                answers.append({'pred_ans': 1, 'pred_expl': response})
            else:
                answers.append({'pred_ans': None, 'pred_expl': response})
    return answers


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

