import json
import os

def _embed_prompt(
    instruction, template_with_label, template_no_label, dem_examples, test_example
):
    prompt = str(instruction)
    for dem in dem_examples:
        prompt += template_with_label.format(**dem)

    # prompt += template_no_label.format(**test_example).strip()
    prompt += template_no_label.format(**test_example)
    return prompt


def get_prompts_by_task(task, test_examples):
    prompt = json.load(open(os.path.join(os.path.dirname(__file__), 'prompts.json')))[task]
    # if len(test_examples) > 0:
    #     print(_embed_prompt(prompt['instruction'], prompt['template_with_label'],
    #                           prompt['template_no_label'], prompt['dem_examples'], test_examples[0]))
    #     exit()
    return [_embed_prompt(prompt['instruction'], prompt['template_with_label'],
                          prompt['template_no_label'], prompt['dem_examples'], test_example)
            for test_example in test_examples]

