import sys
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
import json
import time
import pickle as pkl


def task_qa_posthoc_force_answer(model, questions, forced_answers):
    assert len(questions) == len(forced_answers)
    prompts = get_prompts_by_task(f'strategyqa-taskqa-posthoc-force-answer',
                                  [{'question': question, 'answer': answer}
                                   for question, answer in zip(questions, forced_answers)])
    pred_expls = multiprocess_api(model=model, prompts=prompts, bsz=8, num_processes=12,
                                 temperature=0, max_tokens=200, stop='\n\n')
    assert len(pred_expls) == len(prompts)
    pred_expls = [pred_expl.strip() for pred_expl in pred_expls]
    preds = [{'pred_expl': pred_expl, 'pred_ans': answer} for pred_expl, answer in zip(pred_expls, forced_answers)]
    assert len(preds) == len(questions) == len(forced_answers)
    return preds


if __name__ == '__main__':
    taskqa_model = 'gpt3'

    # filter out example idxs where both cot and posthoc are correct
    correct_ex_idxs = []
    taskqa_expl_type2preds = {}
    for taskqa_expl_type in ['cot', 'posthoc']:
        taskqa_preds = pkl.load(open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}.pkl', 'rb'))
        taskqa_expl_type2preds[taskqa_expl_type] = {exidx: taskqa_preds[exidx]['pred_ans'] for exidx in taskqa_preds}
    data = json.load(open('../data/data.json', 'r'))['test']
    for exidx in range(200):
        if taskqa_expl_type2preds['cot'][exidx] == taskqa_expl_type2preds['posthoc'][exidx] == data[exidx]['answer']:
            correct_ex_idxs.append(exidx)
    print(len(correct_ex_idxs))

    questions, forced_answers = [], []
    for exidx in correct_ex_idxs:
        questions.append(data[exidx]['question'])
        forced_answers.append({'no': 'yes', 'yes': 'no'}[data[exidx]['answer']])
    preds = task_qa_posthoc_force_answer('gpt3', questions, forced_answers)
    assert len(preds) == len(correct_ex_idxs)
    exidx2preds = {correct_ex_idxs[idx]: preds[idx] for idx in range(len(correct_ex_idxs))}
    pkl.dump(exidx2preds, open('../outputs/taskqa_gpt3_posthocforced.pkl', 'wb'))
