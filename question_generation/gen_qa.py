import os
import sys
import json
import random
import logging
from datetime import datetime
import argparse
import utils
import ast
from utils import fraction_dict

logging.basicConfig(format='%(message)s', level=logging.DEBUG)


def choice_synonyms(synonyms):
    temp = {}
    for key in synonyms:
        temp[key] = random.choice(synonyms[key])
    return temp


def replace_by_synonym(key, synonyms):
    if key in synonyms:
        return synonyms[key]
    else:
        return key


def instantiate_templates(scene, template, synonyms):
    logging.debug(f'template: {template}')

    states = []
    for node_idx, node in enumerate(template['nodes']):
        logging.debug(f'node: {node}')
        if len(states) == 0:
            state = {'index': node_idx, 'inputs': {'plane': set(), 'object': set()}}
        else:
            state = {'index': node_idx, 'inputs': states[-1]['outputs']}

        state['node'] = node
        # inputs in node
        input_states = [states[i] for i in node.get("input_states", [])]
        # constraints
        constraints = template.get('constraints', [])
        logging.debug(f'input_states: {node.get("input_states", [])}')
        logging.debug(f'constraints: {constraints}')
        if node['type'] == 'scene':
            state['outputs'] = {
                'plane': set(range(len(scene['planes']))),
                'object': set(range(len(scene['objects'])))
            }
        elif node['type'].startswith('filter'):
            utils.filter_handler(state, scene, input_states, constraints)
        elif node['type'].startswith('relate_filter'):
            utils.relate_filter_handler(state, scene, input_states, constraints)
        elif node['type'].startswith('contain'):
            utils.contain_filter_handler(state, scene)
        elif node['type'].startswith('not_contain'):
            utils.not_contain_filter_handler(state, scene)
        elif node['type'].startswith('position_filter'):
            utils.position_filter_handler(state, scene)
        elif node['type'].startswith('different_attribute'):
            utils.different_attribute_handler(state, scene, input_states)
        elif node['type'].startswith('query'):
            utils.query_handler(state, scene, input_states)
        elif node['type'].startswith('exist'):
            utils.exist_handler(state, scene, input_states)
        elif node['type'].startswith('whether'):
            utils.whether_handler(state, scene, input_states)
        elif 'attributes_subset' in node['type']:
            utils.attributes_subset_handler(state, scene, input_states)
        elif '_relate' in node['type']:
            utils.q_relate_handler(state, scene, input_states)
        elif 'except' in node['type']:
            utils.except_handler(state, scene, input_states)
        elif '_A_' in node['type']:
            utils.compare_handler(state, scene, input_states)
        elif '_larger_' in node['type'] or '_smaller_' in node['type']:
            utils.compare_size_handler(state, scene, input_states)

        logging.debug(f'state: {state}')
        states.append(state)

    question_text = random.choice(template['text'])

    # ensure same val => text
    synonyms = choice_synonyms(synonyms)
    # program
    program = []
    obj_attrs = []
    for state in states:
        p = {
            '_output': {},
            'vals': []
        }
        if 'vals' in state:
            p['vals'] = list(state['vals'])
        if state['node']['type'].startswith('query') or \
                '_larger_' in state['node']['type'] or \
                '_smaller_' in state['node']['type'] or \
                state['node']['type'].startswith('different_attribute'):
            p['_output'] = state['out_program']
        else:
            if 'outputs' in state:
                p['_output'] = {
                    'plane': sorted(list(state['outputs']['plane'])),
                    'object': sorted(list(state['outputs']['object']))
                }
        program.append(p)

    for state in states:
        if 'value_inputs' in state['node']:
            if 'vals' not in state:
                continue
            for idx, value_input in enumerate(state['node']['value_inputs']):
                if value_input not in question_text:
                    continue
                if state['vals'][idx] is None:
                    if 'attributes_subset' not in state['node']['type'] and \
                            value_input.startswith('<S') and \
                            (
                                    state['node']['kind'] == 'object' or
                                    state['node']['type'].startswith('contain') or
                                    state['node']['type'].startswith('not_contain')
                            ):
                        question_text = question_text.replace(value_input, synonyms['object'])
                    else:
                        question_text = question_text.replace(value_input, '')
                else:
                    question_text = question_text.replace(value_input, replace_by_synonym(state['vals'][idx], synonyms))

    if ' geometric ' in question_text:
        if random.random() < 0.5:
            question_text = question_text.replace(' geometric ', ' non-white ')
    # if 'No ' in question_text:
    #     if random.random() < 0.5:
    #         question_text = question_text.replace('No ', '0 ')

    # process {}
    # pattern = re.compile(r"{.*}")
    # for found in re.finditer(pattern, question_text):
    #     if random.random() < 0.5:
    #         question_text = question_text.replace(found.group(), '')
    #     else:
    #         if 'planes' in question_text:
    #             question_text = question_text.replace(found.group(), found.group()[1:-1])
    #         else:
    #             question_text = question_text.replace(found.group(), '')
    #
    # # []
    # pattern1 = re.compile(r"\[.*]")
    # for found in re.finditer(pattern1, question_text):
    #     if 'planes' in question_text:
    #         question_text = question_text.replace(found.group(), found.group()[1:-1])
    #     else:
    #         question_text = question_text.replace(found.group(), '')

    if '<' in question_text:
        print('Have < in question text')
        print(question_text)
        print(template)
        sys.exit()

    question_text = utils.space_replace(question_text)
    question_text = utils.space_punc_replace(question_text)
    question_text = utils.upper_first_charc(question_text)
    logging.info(question_text)
    logging.info(states[-1]['answer'])
    return question_text, states[-1]['answer'], program


def read_synonyms(synonyms_json='./synonyms.json'):
    with open(synonyms_json, 'r') as f:
        synonyms = json.load(f)
    return synonyms


def merge_templates():
    templates = read_templates()
    temp = []
    for key in templates:
        template = templates[key]
        template['template_filename'] = key[0]
        template['template_family_index'] = key[1]
        template['template_index'] = key[2]
        temp.append(template)
    with open('templates.json', 'w') as f:
        json.dump(temp, f, indent=4)


def last_node_contains_numerical(template):
    if 'value_inputs' in template['nodes'][-1]:
        for value_input in template['nodes'][-1]['value_inputs']:
            if '<OP' in value_input or '<OC' in value_input:
                return True
    return False


def get_numerical_vals(template, p):
    numerical_vals = ''
    node, program = template['nodes'][-1], p[-1]
    assert len(node['value_inputs']) == len(program['vals'])
    for val_input, val in zip(node['value_inputs'], program['vals']):
        if '<OC' in val_input or '<OP' in val_input:
            if val in fraction_dict:
                val = fraction_dict[val]
            numerical_vals += val + ','
    return numerical_vals


def read_templates(template_dir='./templates'):
    templates = {}
    for filename in os.listdir(template_dir):
        if not filename.endswith('json'):
            continue
        with open(f'templates/{filename}', 'r') as f:
            # enumerate templates
            family_idx = 0
            for template in json.load(f):
                if type(template) == list:
                    idx_1 = 0
                    for t in template:
                        if type(t) == list:
                            idx_2 = 0
                            for t_ in t:
                                templates[(filename, family_idx, idx_1, idx_2)] = t_
                                idx_2 += 1
                        else:
                            templates[(filename, family_idx, idx_1)] = t
                        idx_1 += 1
                else:
                    templates[(filename, family_idx)] = template
                family_idx += 1
    print(f'Read {len(templates)} template from disk')
    return templates


def shuffle_templates(storage, template_list):
    templates_sorted = []
    random.shuffle(template_list)
    for key, template in template_list:
        if key in storage['templates']:
            if len(key) == 2:
                templates_sorted.append((key, template,
                                         (storage['templates'][key]['Total'],
                                         storage['templates'][key]['Total'],
                                         storage['templates'][key]['Total'])))
            elif len(key) == 3:
                family_total_dict = {k: v for k, v in storage['templates'].items() if k[0] == key[0] and k[1] == key[1]}
                family_total = sum([v['Total'] for v in family_total_dict.values()])
                templates_sorted.append((key, template,
                                         (family_total,
                                          storage['templates'][key]['Total'],
                                          storage['templates'][key]['Total'])))
            else:
                family_total_dict = {k: v for k, v in storage['templates'].items() if k[0] == key[0] and k[1] == key[1]}
                family_total = sum([v['Total'] for v in family_total_dict.values()])
                idx_1_total_dict = {k: v for k, v in storage['templates'].items() if k[0] == key[0] and k[1] == key[1] and k[2] == key[2]}
                idx_1_total = sum([v['Total'] for v in idx_1_total_dict.values()])
                templates_sorted.append((key, template,
                                         (family_total, idx_1_total, storage['templates'][key]['Total'])))
        else:
            if len(key) == 2:
                templates_sorted.append((key, template, (0, 0, 0)))
            elif len(key) == 3:
                family_total_dict = {k: v for k, v in storage['templates'].items() if k[0] == key[0] and k[1] == key[1]}
                family_total = sum([v['Total'] for v in family_total_dict.values()]) if len(family_total_dict) else 0
                templates_sorted.append((key, template, (family_total, 0, 0)))
            else:
                family_total_dict = {k: v for k, v in storage['templates'].items() if k[0] == key[0] and k[1] == key[1]}
                family_total = sum([v['Total'] for v in family_total_dict.values()]) if len(family_total_dict) else 0
                idx_1_total_dict = {k: v for k, v in storage['templates'].items() if k[0] == key[0] and k[1] == key[1] and k[2] == key[2]}
                idx_1_total = sum([v['Total'] for v in idx_1_total_dict.values()]) if len(idx_1_total_dict) else 0
                templates_sorted.append((key, template, (family_total, idx_1_total, 0)))
    templates_sorted = sorted(templates_sorted, key=lambda x: x[2], reverse=True)
    return templates_sorted


def generate_question(
        input_scene_file,
        synonyms_json,
        template_dir,
        output_questions_file,
        output_storage_file,
        scene_start_idx,
        num_scenes,
        num_retries=10,
        num_templates_per_image=10,
        instances_per_template=1,
        save_after_num_imgs=10
):
    today = datetime.today()

    # read scenes
    with open(input_scene_file, 'r') as f:
        imgs_info = json.load(f)

    scenes = imgs_info['scenes']
    scene_info = imgs_info['info']

    # questions
    if os.path.exists(output_questions_file):
        assert os.path.exists(output_storage_file)
        with open(output_questions_file, 'r') as f:
            questions = json.load(f)
        # last_index = max([i['question_index'] for i in questions['questions']])
        last_index = questions['questions'][-1]['question_index']
        assert last_index == len(questions['questions']) - 1
        scene_image_index = questions['questions'][-1]['image_index']
        if scene_start_idx is not None:
            assert scene_image_index == scene_start_idx - 1
        else:
            scene_start_idx = scene_image_index + 1
        print(last_index)

        # storage
        with open(output_storage_file, 'r') as f:
            storage_json = json.load(f)
        storage = {
            'templates': {ast.literal_eval(k): v for k, v in storage_json['templates'].items()},
            'numerical': {ast.literal_eval(k): v for k, v in storage_json['numerical'].items()}
        }
    else:
        questions = {
            'info': {
                'date': f'{today.month:02d}/{today.day:02d}/{today.year}',
                'version': scene_info['version'],
                'split': scene_info['split']
            },
            'questions': []
        }
        storage = {
            'templates': {},
            'numerical': {}
        }
        last_index = -1
        scene_start_idx = 0

    if num_scenes is None:
        num_scenes = len(scenes) - scene_start_idx
    else:
        assert num_scenes <= len(scenes)

    # read synonyms
    synonyms = read_synonyms(synonyms_json)
    # read templates
    templates = read_templates(template_dir)
    template_list = list(templates.items())
    # instantiation
    count = 1
    save_times = 0
    for scene_idx, scene in enumerate(scenes[scene_start_idx: scene_start_idx + num_scenes]):
        print('*'*100)
        print(f'scene - {scene_idx + scene_start_idx}')
        # shuffle the template list
        template_queue = shuffle_templates(storage, template_list)
        # print(template_list)
        template = None
        for i in range(num_templates_per_image):
            print(f'generating question - {i}')
            success = False
            question = {
                'question_index': last_index + count,
                'image_index': scene["image_index"],
                'question': '',
                'program': [],
                'answer': '',
                'template': ()
            }
            print('total question index - ', question['question_index'])
            while not success:
                retry_count = 0
                key, template, _ = template_queue.pop()
                # print(count)
                question['template'] = key
                # keep balance of answers
                if question['template'] in storage['templates']:
                    if storage['templates'][question['template']]['True'] > \
                            storage['templates'][question['template']]['False']:
                        expectation = False
                    elif storage['templates'][question['template']]['True'] == \
                            storage['templates'][question['template']]['False']:
                        expectation = False if random.random() < 0.5 else True
                    else:
                        expectation = True
                else:
                    expectation = None
                while True:
                    try:
                        if (question['template'][0] == 'different_attr.json' or question['template'][0] == 'relate.json') and len(scene['planes']) < 4:
                            break
                        if question['template'][0] == 'total.json' and len(scene['planes']) < 3:
                            break
                        if question['template'][0].startswith('each_') and question['template'][1] == 1 and len(scene['planes']) < 3:
                            break
                        q, a, p = instantiate_templates(scene, template, synonyms)
                        expectation_num = None
                        if last_node_contains_numerical(template):
                            # keep balance of same type questions which contain same number or fraction
                            if question['template'] in storage['numerical']:
                                numerical_vals = get_numerical_vals(template, p)
                                if numerical_vals in storage['numerical'][question['template']]:
                                    if storage['numerical'][question['template']][numerical_vals]['True'] > \
                                            storage['numerical'][question['template']][numerical_vals]['False']:
                                        expectation_num = False
                                    elif storage['numerical'][question['template']][numerical_vals]['True'] == \
                                            storage['numerical'][question['template']][numerical_vals]['False']:
                                        expectation_num = False if random.random() < 0.5 else True
                                    else:
                                        expectation_num = True

                        if expectation is not None:
                            assert a == expectation
                            if expectation_num is not None:
                                assert a == expectation_num
                    except:
                        retry_count += 1
                        if retry_count > num_retries:
                            break
                    else:
                        question['question'] = q
                        question['answer'] = a
                        question['program'] = p
                        questions['questions'].append(question)
                        success = True
                        # each template family will be just generated once in an image
                        for t in template_queue:
                            if t[0][0] == question['template'][0] and t[0][1] == question['template'][1]:
                                template_queue.remove(t)
                        print('template used - ', question['template'])
                        count += 1
                        if question['template'] in storage['templates']:
                            storage['templates'][question['template']]['Total'] += 1
                            if a:
                                storage['templates'][question['template']]['True'] += 1
                            else:
                                storage['templates'][question['template']]['False'] += 1
                        else:
                            storage['templates'][question['template']] = {
                                'True': 1 if a else 0,
                                'False': 1 if not a else 0,
                                'Total': 1
                            }
                        if last_node_contains_numerical(template):
                            numerical_vals = get_numerical_vals(template, p)
                            if question['template'] in storage['numerical']:
                                if numerical_vals in storage['numerical'][question['template']]:
                                    storage['numerical'][question['template']][numerical_vals]['Total'] += 1
                                    if a:
                                        storage['numerical'][question['template']][numerical_vals]['True'] += 1
                                    else:
                                        storage['numerical'][question['template']][numerical_vals]['False'] += 1
                                else:
                                    storage['numerical'][question['template']][numerical_vals] = {
                                        'True': 1 if a else 0,
                                        'False': 1 if not a else 0,
                                        'Total': 1
                                    }
                            else:
                                storage['numerical'][question['template']] = {}
                                storage['numerical'][question['template']][numerical_vals] = {
                                    'True': 1 if a else 0,
                                    'False': 1 if not a else 0,
                                    'Total': 1
                                }
                        break
            # repeat
            if success and instances_per_template > 1:
                # keep balance of answers
                family_balance_dict = {k: v for k, v in storage['templates'].items() if
                                       k[0] == question['template'][0] and k[1] == question['template'][1]}
                if len(family_balance_dict):
                    family_total_true = sum([v['True'] for v in family_balance_dict.values()])
                    family_total_false = sum([v['False'] for v in family_balance_dict.values()])
                    if family_total_true > family_total_false:
                        expectation = False
                    elif family_total_true == family_total_false:
                        expectation = False if random.random() < 0.5 else True
                    else:
                        expectation = True
                else:
                    expectation = None
                question['question_index'] += 1
                for j in range(1, instances_per_template):
                    while True:
                        try:
                            q, a, p = instantiate_templates(scene, template, synonyms)
                            if last_node_contains_numerical(template):
                                if question['template'] in storage['numerical']:
                                    numerical_vals = get_numerical_vals(template, p)
                                    if numerical_vals in storage['numerical'][question['template']]:
                                        if storage['numerical'][question['template']][numerical_vals]['True'] > \
                                                storage['numerical'][question['template']][numerical_vals]['False']:
                                            expectation = False
                                        elif storage['numerical'][question['template']][numerical_vals]['True'] == \
                                                storage['numerical'][question['template']][numerical_vals]['False']:
                                            expectation = False if random.random() < 0.5 else True
                                        else:
                                            expectation = True
                            if expectation is not None:
                                assert a == expectation
                        except:
                            pass
                        else:
                            question['question'] = q
                            question['answer'] = a
                            question['program'] = p
                            questions['questions'].append(question)
                            count += 1
                            if question['template'] in storage['templates']:
                                storage['templates'][question['template']]['Total'] += 1
                                if a:
                                    storage['templates'][question['template']]['True'] += 1
                                else:
                                    storage['templates'][question['template']]['False'] += 1
                            else:
                                storage['templates'][question['template']] = {
                                    'True': 1 if a else 0,
                                    'False': 1 if not a else 0,
                                    'Total': 1
                                }
                            if last_node_contains_numerical(template):
                                numerical_vals = get_numerical_vals(template, p)
                                if question['template'] in storage['numerical']:
                                    if numerical_vals in storage['numerical'][question['template']]:
                                        storage['numerical'][question['template']][numerical_vals]['Total'] += 1
                                        if a:
                                            storage['numerical'][question['template']][numerical_vals]['True'] += 1
                                        else:
                                            storage['numerical'][question['template']][numerical_vals]['False'] += 1
                                    else:
                                        storage['numerical'][question['template']][numerical_vals] = {
                                            'True': 1 if a else 0,
                                            'False': 1 if not a else 0,
                                            'Total': 1
                                        }
                                else:
                                    storage['numerical'][question['template']] = {}
                                    storage['numerical'][question['template']][numerical_vals] = {
                                        'True': 1 if a else 0,
                                        'False': 1 if not a else 0,
                                        'Total': 1
                                    }
                            break
        save_times += 1
        if save_times == save_after_num_imgs:
            save_times = 0
            # save storage
            storage['templates'] = dict(sorted(storage['templates'].items()))
            storage['numerical'] = dict(sorted(storage['numerical'].items()))
            with open(output_storage_file, 'w') as f:
                json.dump({'templates': {str(k): v for k, v in storage['templates'].items()},
                           'numerical': {str(k): v for k, v in storage['numerical'].items()}}, f, indent=2)
            # save questions
            with open(output_questions_file, 'w') as f:
                json.dump(questions, f)
            print('json file saved')
    if save_times != 0:
        # save storage
        storage['templates'] = dict(sorted(storage['templates'].items()))
        storage['numerical'] = dict(sorted(storage['numerical'].items()))
        with open(output_storage_file, 'w') as f:
            json.dump({'templates': {str(k): v for k, v in storage['templates'].items()},
                       'numerical': {str(k): v for k, v in storage['numerical'].items()}}, f, indent=2)
        # save questions
        with open(output_questions_file, 'w') as f:
            json.dump(questions, f)

        print('task finished and json file saved')


def test():
    with open('output/3d_scene/train/scenes_3d_train.json', 'r') as f:
        scenes = json.load(f)

    # read synonyms
    synonyms = read_synonyms('./synonyms.json')
    # read templates
    templates = []
    with open(f'templates/each_not_exactly.json', 'r') as f:
        for template in json.load(f):
            templates.append(template)

    scene = scenes['scenes'][0]
    template = templates[41][1]

    # a = False
    # while not a:
    #     try:
    #         _, a, _ = instantiate_templates(scene, template, synonyms)
    #     except Exception as e:
    #         logging.error(f'Error: {e}')
    #         # raise e
    #     logging.debug(f'{"*" * 100}')

    for i in range(50):
        try:
            instantiate_templates(scene, template, synonyms)
        except Exception as e:
            logging.error(f'Error: {e}')
            # raise e
        logging.debug(f'{"*" * 100}')



def test_template_valid(t):
    templates = []
    with open(t, 'r') as f:
        for template in json.load(f):
            templates.append(template)
    id = 0
    for t in templates:
        if isinstance(t, list):
            for t1 in t:
                if isinstance(t1, list):
                    for t2 in t1:
                        unique_input = []
                        for n in t2['nodes']:
                            if "value_inputs" in n:
                                for ui in n['value_inputs']:
                                    if ui not in unique_input:
                                        unique_input.append(ui)
                                    else:
                                        print(id)
                        id += 1
                else:
                    unique_input = []
                    for n in t1['nodes']:
                        if "value_inputs" in n:
                            for ui in n['value_inputs']:
                                if ui not in unique_input:
                                    unique_input.append(ui)
                                else:
                                    print(id)
                    id += 1
        else:
            unique_input = []
            for n in t['nodes']:
                if "value_inputs" in n:
                    for ui in n['value_inputs']:
                        if ui not in unique_input:
                            unique_input.append(ui)
                        else:
                            print(id)
            id += 1
    print(id)


def num_template(t):
    # read question templates
    templates = []
    with open(t, 'r') as f:
        for template in json.load(f):
            templates.append(template)
    logging.info(f'Read {len(templates)} template from disk')
    print(f'Read {len(templates)} template from disk')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA Generation')
    parser.add_argument('--input_scene_file', default='./data/3d_scene/val/scenes_3d_val.json', type=str,
                        help='input scene file')
    parser.add_argument('--synonyms_json', default='./synonyms.json', type=str,
                        help='synonyms file')
    parser.add_argument('--template_dir', default='./templates/', type=str,
                        help='path of template directory')
    parser.add_argument('--output_questions_file', default='./data/3d_scene/val/questions_val.json', type=str,
                        help='output questions file')
    parser.add_argument('--output_storage_file', default='./data/3d_scene/val/storage_val.json', type=str,
                        help='output storage file')
    parser.add_argument('--scene_start_idx', default=None, type=int,
                        help='start from which scene')
    parser.add_argument('--num_scenes', default=None, type=int,
                        help='number of scenes')
    parser.add_argument('--num_retries', default=2000, type=int,
                        help='number of max retries')
    parser.add_argument('--num_templates_per_image', default=10, type=int,
                        help='number of templates per image')
    parser.add_argument('--instances_per_template', default=1, type=int,
                        help='number of instances per template')
    parser.add_argument('--save_times', default=100, type=int,
                        help='Save json file again after finishing the generation of the given number of images')
    args = parser.parse_args()
    generate_question(
        input_scene_file=args.input_scene_file,
        synonyms_json=args.synonyms_json,
        template_dir=args.template_dir,
        output_questions_file=args.output_questions_file,
        output_storage_file=args.output_storage_file,
        scene_start_idx=args.scene_start_idx,
        num_scenes=args.num_scenes,
        num_retries=args.num_retries,
        num_templates_per_image=args.num_templates_per_image,
        instances_per_template=args.instances_per_template,
        save_after_num_imgs=args.save_times
    )

