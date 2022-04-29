import os
import re
import json
import random
import logging
from itertools import product, combinations
from collections import Counter
from fractions import Fraction

from num2words import num2words


fraction_dict = {}


def generate_fraction_map():
    n_map = {}
    for i in range(1, 63):
        if i == 1:
            n_map[i] = random.choice([f'{num2words(i)}-', 'a ', f'{num2words(i)} '])
        elif i < 20:
            n_map[i] = random.choice([f'{num2words(i)}-', f'{num2words(i)} '])
        else:
            n_map[i] = f'{num2words(i)} '
    # n_map_over = {}
    # for i in range(1, 63):
    #     n_map_over[i] = f'{num2words(i)} over '
    d_map = {}
    for i in range(2, 63):
        if i == 2:
            d_map[i] = 'half'
        elif i == 4:
            d_map[i] = random.choice(['fourth', 'quarter'])
        else:
            d_map[i] = f"{num2words(i, to='ordinal')}"
    # d_map_over = {}
    # for i in range(2, 63):
    #     d_map_over[i] = f'{num2words(i)}'
    return n_map, d_map


# numerator_map, denominator_map, numerator_over_map, over_denominator_map = generate_fraction_map()


def precompute_filter_options(scene):
    options = {'plane': {}, 'object': {}}

    # Attribute mapping for plane
    plane_attrs = ('color', 'material', 'shape')
    masks = list(product(range(2), repeat=len(plane_attrs)))
    for plane_idx, plane in enumerate(scene['planes']):
        keys = [tuple(plane[k] for k in plane_attrs)]
        for mask in masks:
            for key in keys:
                masked_key = []
                for a, b in zip(key, mask):
                    if b == 1:
                        if a == '':  # non-geometric plane
                            masked_key.append(None)
                        else:
                            masked_key.append(a)
                    else:
                        masked_key.append(None)
                masked_key = tuple(masked_key)
                if masked_key not in options['plane']:
                    options['plane'][masked_key] = set()
                options['plane'][masked_key].add(plane_idx)

    # Attribute mapping for object
    obj_attrs = ('size', 'color', 'material', 'shape')
    masks = list(product(range(2), repeat=len(obj_attrs)))
    for obj_idx, obj in enumerate(scene['objects']):
        keys = [tuple(obj[k] for k in obj_attrs)]
        for mask in masks:
            for key in keys:
                masked_key = []
                for a, b in zip(key, mask):
                    if b == 1:
                        masked_key.append(a)
                    else:
                        masked_key.append(None)
                masked_key = tuple(masked_key)
                if masked_key not in options['object']:
                    options['object'][masked_key] = set()
                options['object'][masked_key].add(obj_idx)
    # remove null filter
    # options['plane'].pop((None, None, None), None)
    # options['object'].pop((None, None, None, None), None)
    logging.debug(f'precompute filter options, {options}')
    scene['_filter_options'] = options


def space_replace(text):
    return re.sub(' +', ' ', text)


def space_punc_replace(text):
    return re.sub(' ([?.;])', r'\1', text)


def upper_first_charc(text):
    first_s = text[0]
    rest_s = text[1:]
    first_s = first_s.upper() if first_s.islower() else first_s
    text = first_s + rest_s
    return text


def key_contains(key1, key2):
    flag = True
    for i in key2:
        if i is None:
            continue
        if i not in key1:
            flag = False
            break
    return flag


def r_key_contains(key1, key2):
    flag = False
    r_contains = {
        'left': ['left', 'front left', 'behind left'],
        'right': ['right', 'front right', 'behind right'],
        'behind': ['behind', 'behind left', 'behind right'],
        'front': ['front', 'front left', 'front right'],
        'behind left': ['behind left'],
        'front left': ['front left'],
        'behind right': ['behind right'],
        'front right': ['front right']
    }
    if key2[0] in r_contains[key1[0]]:
        if key_contains(key2[1], key1[1]):
            flag = True
    return flag


def constraints_handler(state, options, constraints, input_states):
    for constraint in constraints:
        if constraint['type'] == 'NOT_NULL':
            for value_input in set(state['node'].get('value_inputs', [])) & set(constraint['params']):
                idx = state['node'].get('value_inputs', []).index(value_input)
                options['object'] = {k: v for k, v in options['object'].items() if k[idx] is not None}
        elif constraint['type'] == 'NULL':
            for value_input in set(state['node'].get('value_inputs', [])) & set(constraint['params']):
                idx = state['node'].get('value_inputs', []).index(value_input)
                options['object'] = {k: v for k, v in options['object'].items() if k[idx] is None}
        elif constraint['type'] == 'BOTH_SIZE_BUT_NULL':
            for value_input in set(state['node'].get('value_inputs', [])) & set(constraint['params']):
                assert value_input.startswith('<Z')
                idx = state['node'].get('value_inputs', []).index(value_input)
                op_objs = {}
                for k, v in options['object'].items():
                    if k[idx] is None:
                        k_large = list(k)
                        k_large[idx] = 'large'
                        if tuple(k_large) not in options['object'] or len(options['object'][tuple(k_large)]) == 0:
                            continue
                        k_small = list(k)
                        k_small[idx] = 'small'
                        if tuple(k_small) not in options['object'] or len(options['object'][tuple(k_small)]) == 0:
                            continue
                        op_objs[k] = v
                options['object'] = op_objs
        elif constraint['type'] == 'NOT_ALL_NULL':
            idxs = []
            for value_input in set(state['node'].get('value_inputs', [])) & set(constraint['params']):
                idx = state['node'].get('value_inputs', []).index(value_input)
                idxs.append(idx)
            if sum(idxs):
                new_options = dict()
                for k, v in options['object'].items():
                    all_null = True
                    for idx in idxs:
                        all_null &= (k[idx] is None)
                    if not all_null:
                        new_options[k] = v
                options['object'] = new_options
        elif constraint['type'] == 'EQ':
            first_val = None
            for input_state in input_states:
                for value_input in set(input_state['node'].get('value_inputs', [])) & set(constraint['params']):
                    first_idx = input_state['node'].get('value_inputs', []).index(value_input)
                    first_val = input_state['vals'][first_idx]
            second_idx = None
            for value_input in set(state['node'].get('value_inputs', [])) & set(constraint['params']):
                second_idx = state['node'].get('value_inputs', []).index(value_input)
            if first_val is not None and second_idx is not None:
                options['object'] = {k: v for k, v in options['object'].items() if k[second_idx] == first_val}
        elif constraint['type'] == 'UNQ_EQ':
            if state['index'] == constraint['params'][1]:
                assert input_states[0]['index'] == constraint['params'][0]
                v1 = input_states[0]['outputs']['object']
                k1 = input_states[0]['vals']
                options['object'] = {k: v for k, v in options['object'].items() if (v != v1) or (k == tuple(k1))}
        elif constraint['type'] == 'OUT_NEQ':
            if state['index'] == constraint['params'][1]:
                if input_states[0]['node']['type'] == 'contain_total_multiple_filter' or \
                        input_states[0]['node']['type'].startswith("contain_each") or \
                        input_states[0]['node']['type'].startswith("not_contain_each"):
                    first_vals = input_states[0]['vals'][1:5]
                elif input_states[0]['node']['type'].startswith("contain_between") or \
                        input_states[0]['node']['type'].startswith("not_contain_between"):
                    first_vals = input_states[0]['vals'][2:6]
                else:
                    first_vals = input_states[0]['vals'][:4]
                options['object'] = {k: v for k, v in options['object'].items() if k != tuple(first_vals)}
        elif constraint['type'] == 'NOT_SAME':
            if state['index'] == constraint['params'][1]:
                if input_states[0]['node']['type'] == 'contain_total_multiple_filter' or \
                        input_states[0]['node']['type'].startswith("contain_each") or \
                        input_states[0]['node']['type'].startswith("not_contain_each"):
                    first_vals = input_states[0]['vals'][1:5]
                elif input_states[0]['node']['type'].startswith("contain_between") or \
                        input_states[0]['node']['type'].startswith("not_contain_between"):
                    first_vals = input_states[0]['vals'][2:6]
                else:
                    first_vals = input_states[0]['vals'][:4]
                options['object'] = {k: v for k, v in options['object'].items() if not v & options['object'][tuple(first_vals)]}
        elif constraint['type'] == 'R_OUT_NEQ':
            if state['index'] == constraint['params'][1]:
                v1 = input_states[0]['outputs']['object']
                v2 = input_states[-1]['outputs']['object']
                if v1 == v2:
                    r_val = input_states[1]['vals'][0]
                    o_vals = input_states[1]['vals'][1:5]
                    vals = (r_val, tuple(o_vals))
                    options['object'] = {k: v for k, v in options['object'].items() if k != vals}
        elif constraint['type'] == 'NOT_CONTAIN':
            if state['index'] == max(constraint['params']):
                idx = constraint['params'].index(state['index'])
                if input_states[0]['node']['type'] == 'contain_total_multiple_filter'or \
                        input_states[0]['node']['type'].startswith("contain_each") or \
                        input_states[0]['node']['type'].startswith("not_contain_each"):
                    vals = input_states[0]['vals'][1:5]
                elif input_states[0]['node']['type'].startswith("contain_between") or \
                        input_states[0]['node']['type'].startswith("not_contain_between"):
                    vals = input_states[0]['vals'][2:6]
                else:
                    vals = input_states[0]['vals'][:4]
                if idx == 0:
                    options['object'] = {k: v for k, v in options['object'].items() if not key_contains(k, vals)}
                else:
                    options['object'] = {k: v for k, v in options['object'].items() if not key_contains(vals, k)}
        elif constraint['type'] == 'ATTR_NOT_CONTAIN':
            if state['index'] == max(constraint['params'][:2]):
                idx = constraint['params'].index(state['index'])
                if input_states[-1]['node']['type'] == 'contain_total_multiple_filter' or \
                        input_states[-1]['node']['type'].startswith("contain_each") or \
                        input_states[-1]['node']['type'].startswith("not_contain_each"):
                    vals = input_states[-1]['vals'][1:5]
                elif input_states[-1]['node']['type'].startswith("contain_between") or \
                        input_states[-1]['node']['type'].startswith("not_contain_between"):
                    vals = input_states[-1]['vals'][2:6]
                else:
                    vals = input_states[-1]['vals'][:4]
                # print(vals, vals[0])
                rpl_vals = vals.copy()
                if constraint['params'][2] == "Z" and rpl_vals[0] is not None:
                    rpl_vals[0] = None
                elif constraint['params'][2] == "C" and rpl_vals[1] is not None:
                    rpl_vals[1] = None
                elif constraint['params'][2] == "M" and rpl_vals[2] is not None:
                    rpl_vals[2] = None
                elif constraint['params'][2] == "S" and rpl_vals[3] is not None:
                    rpl_vals[3] = None
                if idx == 0:
                    options['object'] = {k: v for k, v in options['object'].items() if not key_contains(k, rpl_vals)}
                else:
                    options['object'] = {k: v for k, v in options['object'].items() if not key_contains(rpl_vals, k)}
        elif constraint['type'] == 'SIZE_NOT_CONTAIN':
            if state['index'] == max(constraint['params'][:2]):
                idx = constraint['params'].index(state['index'])
                if input_states[-1]['node']['type'] == 'contain_total_multiple_filter' or \
                        input_states[-1]['node']['type'].startswith("contain_each") or \
                        input_states[-1]['node']['type'].startswith("not_contain_each"):
                    vals = input_states[-1]['vals'][1:5]
                elif input_states[-1]['node']['type'].startswith("contain_between") or \
                        input_states[-1]['node']['type'].startswith("not_contain_between"):
                    vals = input_states[-1]['vals'][2:6]
                else:
                    vals = input_states[-1]['vals'][:4]
                # print(vals, vals[0])
                if vals[0] == constraint['params'][2]:
                    rpl_vals = vals.copy()
                    rpl_vals[0] = None
                    if idx == 0:
                        options['object'] = {k: v for k, v in options['object'].items() if not key_contains(k, rpl_vals)}
                    else:
                        options['object'] = {k: v for k, v in options['object'].items() if not key_contains(rpl_vals, k)}
        elif constraint['type'] == 'R_NOT_CONTAIN':
            if state['index'] == max(constraint['params']):
                v1 = input_states[0]['outputs']['object']
                v2 = input_states[-1]['outputs']['object']
                if v1 == v2:
                    idx = constraint['params'].index(state['index'])
                    r_val = input_states[1]['vals'][0]
                    o_vals = input_states[1]['vals'][1:5]
                    vals = (r_val, tuple(o_vals))
                    # print(vals)
                    if idx == 0:
                        options['object'] = {k: v for k, v in options['object'].items() if not r_key_contains(k, vals)}
                    else:
                        options['object'] = {k: v for k, v in options['object'].items() if not r_key_contains(vals, k)}


def filter_handler(state, scene, input_states, constraints):
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    options = {'plane': {}, 'object': {}}
    if state['node']['kind'] == 'plane':
        for k, v in scene['_filter_options']['plane'].items():
            # Only filter_all node could contain the description of 'non-geometric' and 'white'
            if not state['node']['type'].startswith('filter_all') and ('non-geometric' in k or 'white' in k):
                continue
            options['plane'][k] = state['inputs']['plane'] & v
            # logging.debug(options['plane'][k])
        if 'unique' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if len(v) == 1}
        elif 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if len(v) > 1}
        if 'valid' in state['node']['type']:
            options['plane'].pop((None, None, None), None)
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list(answer[0])
        # Control Singular and plural
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            # elif value_input.startswith('<sc'):
            #     if len(answer[1]) > 1:
            #         state['vals'].append('')
            #     else:
            #         state['vals'].append('s')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    else:
        objects = set()
        for plane_idx in state['inputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                objects.add(obj)
        for k, v in scene['_filter_options']['object'].items():
            intersection = objects & v
            if intersection:
                options['object'][k] = intersection
        # constraints
        constraints_handler(state, options, constraints, input_states)
        if 'unique' in state['node']['type']:
            options['object'] = {k: v for k, v in options['object'].items() if len(v) == 1}
        elif 'multiple' in state['node']['type']:
            options['object'] = {k: v for k, v in options['object'].items() if len(v) > 1}
        else:
            options['object'] = {k: v for k, v in options['object'].items() if len(v) > 0}
        if 'valid' in state['node']['type']:
            options['object'].pop((None, None, None, None), None)
        answer = random.choice(list(options['object'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = answer[0]
        state['outputs'] = {'plane': state['inputs']['plane'], 'object': answer[1]}


def relate_filter_handler(state, scene, input_states, constraints):
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    options = {'plane': {}, 'object': {}}
    unique = state['node']['type'].endswith('unique')
    multiple = 'multiple' in state['node']['type']
    valid = 'valid' in state['node']['type']
    not_none = 'not_none' in state['node']['type']
    opposites = {
        'left': ['right'],
        'right': ['left'],
        'behind': ['front'],
        'front': ['behind'],
        'behind left': ['front left', 'behind right'],
        'front left': ['behind left', 'front right'],
        'behind right': ['behind left', 'front right'],
        'front right': ['front left', 'behind right']
    }
    plane_rls = scene['plane_spatial_relations']
    obj_rls = scene['obj_spatial_relations']
    if state['node']['kind'] == 'plane':
        for relationship in plane_rls:
            related = set()
            for plane_idx in state['inputs']['plane']:
                for related_plane_idx in plane_rls[relationship][plane_idx]:
                    if related_plane_idx == plane_idx:
                        continue
                    related.add(related_plane_idx)
            for k, v in scene['_filter_options']['plane'].items():
                intersection = related & v
                if intersection:
                    if unique and len(intersection) != 1:
                        continue
                    if multiple and len(intersection) <= 1:
                        continue
                    options['plane'][(relationship, k)] = intersection
        for relationship in ['behind left', 'front left', 'behind right', 'front right']:
            r1, r2 = relationship.split()
            related1 = set()
            related2 = set()
            for plane_idx in state['inputs']['plane']:
                for related_plane_idx in plane_rls[r1][plane_idx]:
                    if related_plane_idx == plane_idx:
                        continue
                    related1.add(related_plane_idx)
                for related_plane_idx in plane_rls[r2][plane_idx]:
                    if related_plane_idx == plane_idx:
                        continue
                    related2.add(related_plane_idx)
            related = related1 & related2
            for k, v in scene['_filter_options']['plane'].items():
                intersection = related & v
                if intersection:
                    if unique and len(intersection) != 1:
                        continue
                    if multiple and len(intersection) <= 1:
                        continue
                    options['plane'][(relationship, k)] = intersection
        new_options = {'plane': {}, 'object': {}}
        for k, v in options['plane'].items():
            for k1, v1 in options['plane'].items():
                if k1[0] in opposites[k[0]] and k[1] == k1[1]:
                    new_options['plane'][k] = v
        options = new_options
        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = answer[0][0], *answer[0][1]
        state['vals'] = list(state['vals'])
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            if value_input.startswith('<pg'):
                if state['vals'][1] is None and state['vals'][2] is None and state['vals'][3] is None:
                    state['vals'].append('geometric')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    else:
        objects = set()
        for plane_idx in state['inputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                objects.add(obj)
        for k, v in scene['_filter_options']['object'].items():
            intersection = objects & v
            if intersection:
                for relationship in obj_rls:
                    related = set()
                    for obj_idx in state['inputs']['object']:
                        for related_obj_idx in obj_rls[relationship][obj_idx]:
                            if related_obj_idx == obj_idx:
                                continue
                            if related_obj_idx in intersection:
                                related.add(related_obj_idx)
                    if len(related):
                        if unique and len(related) != 1:
                            continue
                        if multiple and len(related) <= 1:
                            continue
                        options['object'][(relationship, k)] = related
                    else:
                        if 'unlimited' in state['node']['type']:
                            options['object'][(relationship, k)] = {}

                for relationship in ['behind left', 'front left', 'behind right', 'front right']:
                    r1, r2 = relationship.split()
                    related1 = set()
                    related2 = set()
                    for obj_idx in state['inputs']['object']:
                        for related_obj_idx in obj_rls[r1][obj_idx]:
                            if related_obj_idx == obj_idx:
                                continue
                            if related_obj_idx in intersection:
                                related1.add(related_obj_idx)
                        for related_obj_idx in obj_rls[r2][obj_idx]:
                            if related_obj_idx == obj_idx:
                                continue
                            if related_obj_idx in intersection:
                                related2.add(related_obj_idx)
                        related_sum = related1 & related2
                        if related_sum:
                            if unique and len(related_sum) != 1:
                                continue
                            if multiple and len(related_sum) <= 1:
                                continue
                            options['object'][(relationship, k)] = related_sum
                        else:
                            if 'unlimited' in state['node']['type']:
                                options['object'][(relationship, k)] = {}
        logging.debug(f'options: {options}')
        new_options = {'plane': {}, 'object': {}}
        total_objs_of_r = {}
        if 'unlimited' not in state['node']['type']:
            for k, v in options['object'].items():
                if k[1] == tuple([None] * 4):
                    total_objs_of_r[k] = v
                    if valid:
                        continue
                for k1, v1 in options['object'].items():
                    if k1[0] in opposites[k[0]] and k[1] == k1[1]:
                        new_options['object'][k] = v
        else:
            none_r_list = []
            for r in list(opposites.keys()):
                if (r, tuple([None] * 4)) in options['object'].keys() and len(
                        options['object'][(r, tuple([None] * 4))]) == 0:
                    none_r_list.append(r)
                    none_r_list.extend(opposites[r])
            new_options['object'] = {k: v for k, v in options['object'].items() if k[0] not in none_r_list}
            if not_none or multiple or unique:
                new_options['object'] = {k: v for k, v in new_options['object'].items() if len(v)}
            if valid:
                total_objs_of_r = {k: v for k, v in new_options['object'].items() if k[1] == tuple([None] * 4)}
                new_options['object'] = {k: v for k, v in new_options['object'].items() if k[1] != tuple([None] * 4)}
        options = new_options
        logging.debug(f'options: {options}')
        # constraints
        constraints_handler(state, options, constraints, input_states)
        answer = random.choice(list(options['object'].items()))
        if valid:
            state['total_objs_of_r'] = total_objs_of_r[(answer[0][0], tuple([None] * 4))]
        else:
            state['total_objs_of_r'] = options['object'][(answer[0][0], tuple([None] * 4))]
        logging.debug(f'answer: {answer}')
        state['vals'] = answer[0][0], *answer[0][1]
        state['outputs'] = {'plane': state['inputs']['plane'], 'object': answer[1]}


def contain_filter_handler(state, scene):
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    options = {'plane': {}, 'object': set()}
    assert state['node']['kind'] == 'plane'
    if 'multiple' in state['node']['type']:
        assert len(state['inputs']['plane']) > 2
    if state['node']['type'].startswith('contain_each'):
        for plane_idx in state['inputs']['plane']:
            plane = scene['planes'][plane_idx]
            for k, v in scene['_filter_options']['object'].items():
                intersection = set(plane['internal_objs']) & v
                num = len(intersection)
                key = tuple([num, *k])
                if key not in options['plane']:
                    options['plane'][key] = {plane_idx}
                else:
                    options['plane'][key].add(plane_idx)

        if 'least' in state['node']['type']:
            options = contain_at_least(options)
        elif 'most' in state['node']['type']:
            options = contain_at_most(options)

        def check(x, n):
            # 2: multiple; 1: unique
            if n == 2:
                if len(x) >= len(state['inputs']['plane']) or len(x) < n:
                    return False
            else:
                if len(x) >= len(state['inputs']['plane']) or len(x) != n:
                    return False
            return True

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if (check(v, 2) and k[0] != 0)}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if (check(v, 1) and k[0] != 0)}

        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list((str(answer[0][0]), *answer[0][1:]))
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sap'):
                if answer[0][0] > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sop'):
                if answer[0][0] > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    elif state['node']['type'].startswith('contain_total'):
        for i in range(1, len(state['inputs']['plane']) + 1):
            for pair in combinations(state['inputs']['plane'], i):
                for k, v in scene['_filter_options']['object'].items():
                    obj_in_pair = set()
                    for plane_idx in pair:
                        plane = scene['planes'][plane_idx]
                        for obj_idx in plane['internal_objs']:
                            obj_in_pair.add(obj_idx)
                    intersection = obj_in_pair & v
                    num = len(intersection)
                    key = tuple([num, *k])
                    if num > 0:
                        if key not in options['plane']:
                            options['plane'][key] = {pair}
                        else:
                            options['plane'][key].add(pair)

        def check(x, n):
            # 2: multiple; 1: unique
            for item in x:
                if n == 2:
                    if len(item) >= len(state['inputs']['plane']) or len(item) < n:
                        return False
                else:
                    if len(item) >= len(state['inputs']['plane']) or len(item) != n:
                        return False
            return True

        logging.debug(f'options: {options}')

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if (len(v) == 1 and check(v, 2))}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if (len(v) == 1 and check(v, 1))}

        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list((str(answer[0][0]), *answer[0][1:]))
        ans_plane = set(list(answer[1])[0])
        logging.debug(f'ans_plane: {ans_plane}')
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(ans_plane) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sap'):
                if answer[0][0] > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sop'):
                if answer[0][0] > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': ans_plane, 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    elif state['node']['type'].startswith('contain_some'):
        options = contain_some(state, scene)

        def check(x, n):
            # 2: multiple; 1: unique
            if n == 2:
                if len(x) >= len(state['inputs']['plane']) or len(x) < n:
                    return False
            else:
                if len(x) >= len(state['inputs']['plane']) or len(x) != n:
                    return False
            return True

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 2)}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 1)}

        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list(answer[0])
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    elif state['node']['type'].startswith('contain_between'):
        options = contain_between(state, scene)

        def check(x, n):
            # 2: multiple; 1: unique
            if n == 2:
                if len(x) >= len(state['inputs']['plane']) or len(x) < n:
                    return False
            else:
                if len(x) >= len(state['inputs']['plane']) or len(x) != n:
                    return False
            return True

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 2)}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 1)}

        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list((str(answer[0][0]), str(answer[0][1]), *answer[0][2:]))
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sap'):
                if answer[0][1] > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sop'):
                if answer[0][1] > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    else:
        raise Exception('contain no answer')


def not_contain_filter_handler(state, scene):
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    options = {'plane': {}, 'object': set()}
    assert state['node']['kind'] == 'plane'
    if 'multiple' in state['node']['type']:
        assert len(state['inputs']['plane']) > 2
    if state['node']['type'].startswith('not_contain_each'):
        for plane_idx in state['inputs']['plane']:
            plane = scene['planes'][plane_idx]
            for k, v in scene['_filter_options']['object'].items():
                intersection = set(plane['internal_objs']) & v
                num = len(intersection)
                key = tuple([num, *k])
                if key not in options['plane']:
                    options['plane'][key] = {plane_idx}
                else:
                    options['plane'][key].add(plane_idx)

        if 'least' in state['node']['type']:
            options = contain_at_least(options)
        if 'most' in state['node']['type']:
            options = contain_at_most(options)

        def check(x, n):
            # 2: multiple; 1: unique
            if n == 2:
                if len(x) >= len(state['inputs']['plane']) or len(x) < n:
                    return False
            else:
                if len(x) >= len(state['inputs']['plane']) or len(x) != n:
                    return False
            return True

        options['plane'] = {k: state['inputs']['plane'] - v for k, v in options['plane'].items()}

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if (check(v, 2) and k[0] != 0)}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if (check(v, 1) and k[0] != 0)}

        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list((str(answer[0][0]), *answer[0][1:]))
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sap'):
                if answer[0][0] > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sop'):
                if answer[0][0] > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    elif state['node']['type'].startswith('not_contain_any'):
        options = contain_some(state, scene)

        def check(x, n):
            # 2: multiple; 1: unique
            if n == 2:
                if len(x) >= len(state['inputs']['plane']) or len(x) < n:
                    return False
            else:
                if len(x) >= len(state['inputs']['plane']) or len(x) != n:
                    return False
            return True

        options['plane'] = {k: state['inputs']['plane'] - v for k, v in options['plane'].items()}

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 2)}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 1)}

        logging.debug(f'options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list(answer[0])
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            # elif value_input.startswith('<sap'):
            #     if answer[0][0] > 1:
            #         state['vals'].append('are')
            #     else:
            #         state['vals'].append('is')
            # elif value_input.startswith('<sop'):
            #     if answer[0][0] > 1:
            #         state['vals'].append('s')
            #     else:
            #         state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    elif state['node']['type'].startswith('not_contain_between'):
        options = contain_between(state, scene)

        def check(x, n):
            # 2: multiple; 1: unique
            if n == 2:
                if len(x) >= len(state['inputs']['plane']) or len(x) < n:
                    return False
            else:
                if len(x) >= len(state['inputs']['plane']) or len(x) != n:
                    return False
            return True

        options['plane'] = {k: state['inputs']['plane'] - v for k, v in options['plane'].items()}
        logging.debug(f'22222111111111options: {options}')

        if 'multiple' in state['node']['type']:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 2) and k[0] != 0}
        else:
            options['plane'] = {k: v for k, v in options['plane'].items() if check(v, 1) and k[0] != 0}

        logging.debug(f'111111111options: {options}')
        answer = random.choice(list(options['plane'].items()))
        logging.debug(f'answer: {answer}')
        state['vals'] = list((str(answer[0][0]), str(answer[0][1]), *answer[0][2:]))
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<sp'):
                if len(answer[1]) > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sap'):
                if answer[0][1] > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sop'):
                if answer[0][1] > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
        state['outputs'] = {'plane': answer[1], 'object': set()}
        for plane_idx in state['outputs']['plane']:
            for obj in scene['planes'][plane_idx]['internal_objs']:
                state['outputs']['object'].add(obj)
    else:
        raise Exception('not contain no answer')


def position_filter_handler(state, scene):
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    assert state['node']['kind'] == 'plane'
    unique = state['node']['type'].endswith('unique')
    pos_val = {
        'behind': 'at the back',
        'front': 'at the front',
        'left': 'on the leftmost',
        'right': 'on the rightmost'
    }
    options = {}
    plane_rls = scene['plane_spatial_relations']
    for relationship in plane_rls:
        for idx, related in enumerate(plane_rls[relationship]):
            if len(related) == 0:
                for k, v in scene['_filter_options']['plane'].items():
                    intersection = state['inputs']['plane'] & {idx} & v
                    if 'non-geometric' in k or 'white' in k:
                        continue
                    if unique and len(intersection) != 1:
                        continue
                    if len(v) > 1:
                        options[(relationship, k)] = intersection
    answer = random.choice(list(options.items()))
    logging.debug(f'answer: {answer}')
    state['vals'] = pos_val[answer[0][0]], *answer[0][1]
    state['vals'] = list(state['vals'])
    state['outputs'] = {'plane': answer[1], 'object': set()}
    for plane_idx in state['outputs']['plane']:
        for obj in scene['planes'][plane_idx]['internal_objs']:
            state['outputs']['object'].add(obj)


def different_attribute_handler(state, scene, input_states):
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    assert state['node']['kind'] == 'plane'
    assert len(input_states[0]['outputs']['plane']) > 3
    options = {}
    masks = [0, 0, 0]
    for param in state['node']['value_inputs']:
        if param == 'PS':
            masks[2] = 1
        elif param == 'PM':
            masks[1] = 1
        elif param == 'PC':
            masks[0] = 1
    for k, v in scene['_filter_options']['plane'].items():
        skip = False
        for i in range(3):
            if masks[i]:
                if k[i] is None or 'non-geometric' in k or 'white' in k:
                    skip = True
            else:
                if k[i] is not None:
                    skip = True
        if skip or len(v) != 1:
            continue
        options[k] = state['inputs']['plane'] & v
    if len(options) > 1:
        raise Exception('no answer')
    logging.debug(f'options: {options}')
    answer = random.choice(list(options.items()))
    logging.debug(f'answer: {answer}')
    state['outputs'] = {'plane': answer[1], 'object': set()}
    for plane_idx in state['outputs']['plane']:
        for obj in scene['planes'][plane_idx]['internal_objs']:
            state['outputs']['object'].add(obj)
    different_vals = []
    for param in state['node']['value_inputs']:
        if param == 'PS':
            different_vals.append(answer[0][2])
        elif param == 'PM':
            different_vals.append(answer[0][1])
        elif param == 'PC':
            different_vals.append(answer[0][0])
    if len(different_vals) == 1:
        state['out_program'] = different_vals[0]
    else:
        state['out_program'] = different_vals


def query_handler(state, scene, input_states):
    assert len(input_states) == 1
    answer = []
    if state['node']['kind'] == 'plane':
        for plane_idx in state['inputs']['plane']:
            attr = state['node']['value_inputs'][0]
            if attr == 'PS':
                answer.append(scene['planes'][plane_idx]['shape'])
            elif attr == 'PC':
                answer.append(scene['planes'][plane_idx]['color'])
            elif attr == 'PM':
                answer.append(scene['planes'][plane_idx]['material'])
            else:
                raise Exception('Not found queried attribute')
    else:
        for obj_idx in input_states[0]['outputs']['object']:
            attr = state['node']['value_inputs'][0]
            if attr == 'S':
                answer.append(scene['objects'][obj_idx]['shape'])
            elif attr == 'C':
                answer.append(scene['objects'][obj_idx]['color'])
            elif attr == 'M':
                answer.append(scene['objects'][obj_idx]['material'])
            elif attr == 'Z':
                answer.append(scene['objects'][obj_idx]['size'])
            else:
                raise Exception('Not found queried attribute')
    options = dict(sorted(dict(Counter(answer)).items(), key=lambda item: item[1]))
    logging.debug(f'options: {options}')
    logging.debug(f"objects: {input_states[0]['outputs']['object']}")

    if 'most' in state['node']['type']:
        if len(options) == 1:
            answer = list(options)[0]
        else:
            last_key = list(options)[-1]
            last_value = options[last_key]
            total = sum(options.values())
            if last_value > total - last_value:
                answer = last_key
            else:
                raise Exception('query most no answer')
    else:
        raise Exception('Not found query handler')
    # elif 'some' in state['node']['type']:
    #     answer = list(options.keys())
    # elif 'more_than' in state['node']['type']:
    #     if len(options) == 1:
    #         raise Exception('no answer')
    #     else:
    #         last_key = list(options)[-1]
    #         last_value = options[last_key]
    #         total = sum(options.values())
    #         last_second_key = list(options)[-2]
    #         last_second_value = options[last_second_key]
    #         if last_value != last_second_value:
    #             denominator = total
    #             Numerator = random.randint()
    # elif 'exact' in state['node']['type']:
    #     # get duplicate values and remove the keys
    #     dl = list(set([x for x in list(options.values()) if list(options.values()).count(x) > 1]))
    #     exact_options = {key: val for key, val in options.items() if val not in dl}
    #     logging.debug(f'exact options: {exact_options}')
    #     if len(exact_options) == 0:
    #         raise Exception('no answer')
    #     exact_answer = random.choice(list(exact_options.items()))
    #     logging.debug(f'exact answer: {exact_answer}')
    #     answer = [exact_answer[0]]
    #     state['vals'] = list((exact_answer[0], str(exact_answer[1])))
    #     for value_input in state['node']['value_inputs']:
    #         if value_input.startswith('<so'):
    #             if exact_answer[1] > 1:
    #                 state['vals'].append('s')
    #             else:
    #                 state['vals'].append('')
    #     # print(state['vals'])
    logging.debug(f'answer: {answer}')
    state['outputs'] = state['inputs']
    state['answer'] = answer
    state['out_program'] = answer


def exist_handler(state, scene, input_states):
    assert state['node']['kind'] == 'object'
    attr = input_states[0]['node']['value_inputs'][0]
    attr_value = input_states[0]['answer']

    logging.debug(f'attribute: {attr}')
    logging.debug(f'attribute value: {attr_value}')
    targets = []
    for obj_idx in input_states[1]['outputs']['object']:
        target = [obj_idx]
        if attr == 'S':
            target.append(scene['objects'][obj_idx]['shape'])
        elif attr == 'C':
            target.append(scene['objects'][obj_idx]['color'])
        elif attr == 'M':
            target.append(scene['objects'][obj_idx]['material'])
        elif attr == 'Z':
            target.append(scene['objects'][obj_idx]['size'])
        else:
            raise Exception('Not found queried attribute')
        targets.append(target)
    logging.debug(f'targets: {targets}')
    satisf_objects = set()
    for t in targets:
        if t[1] == attr_value:
            satisf_objects.add(t[0])
    logging.debug(f'satisfied objects: {satisf_objects}')
    logging.debug(f"total objects: {input_states[1]['outputs']['object']}")

    exist_num = len(satisf_objects)
    total = len(input_states[1]['outputs']['object'])
    if '_some' in state['node']['type']:
        answer = True if exist_num != 0 else False
    else:
        assert exist_num > 0
        if 'exact' in state['node']['type'] or \
                'at_most' in state['node']['type'] or \
                'at_least' in state['node']['type']:
            if len(input_states) == 3 and \
                    (input_states[2]['node']['type'] == 'contain_total_multiple_filter' or
                     input_states[2]['node']['type'] == 'contain_each_unique_filter' or
                     input_states[2]['node']['type'] == 'contain_each_at_most_unique_filter' or
                     input_states[2]['node']['type'] == 'contain_between_unique_filter'):
                if input_states[2]['node']['type'] == 'contain_total_multiple_filter' or \
                        input_states[2]['node']['type'] == 'contain_each_unique_filter' or \
                        input_states[2]['node']['type'] == 'contain_each_at_most_unique_filter':
                    p_val_num = input_states[2]['vals'][0]
                    p_val_keys = input_states[2]['vals'][1:5]
                else:
                    assert "between" in input_states[2]['node']['type']
                    p_val_num = input_states[2]['vals'][1]
                    p_val_keys = input_states[2]['vals'][2:6]
                o_val_keys = input_states[1]['vals']
                # print(int(p_val_num), p_val_keys, o_val_keys)
                if key_contains(o_val_keys, p_val_keys):
                    cons_total = min(total, int(p_val_num))
                    # print('b', exist_num, 1, cons_total)
                    if 'exist_at_most' == state['node']['type']:
                        answer, rdm_num = at_most(exist_num, cons_total)
                    elif 'exist_at_least' == state['node']['type']:
                        answer, rdm_num = at_least(exist_num, cons_total)
                    elif 'exist_exact' == state['node']['type']:
                        answer, rdm_num = exact(exist_num, cons_total, 1, 'contain')
                    else:
                        raise Exception('exist no handler')
                else:
                    if 'exist_at_most' == state['node']['type']:
                        answer, rdm_num = at_most(exist_num, total)
                    elif 'exist_at_least' == state['node']['type']:
                        answer, rdm_num = at_least(exist_num, total)
                    elif 'exist_exact' == state['node']['type']:
                        answer, rdm_num = exact(exist_num, total)
                    else:
                        raise Exception('exist no handler')
            else:
                if 'exist_at_most' == state['node']['type']:
                    answer, rdm_num = at_most(exist_num, total)
                elif 'exist_at_least' == state['node']['type']:
                    answer, rdm_num = at_least(exist_num, total)
                elif 'exist_exact' == state['node']['type']:
                    answer, rdm_num = exact(exist_num, total)
                else:
                    raise Exception('exist no handler')
            state['vals'] = [str(rdm_num)]
            for value_input in state['node']['value_inputs']:
                if value_input.startswith('<so'):
                    if rdm_num > 1:
                        state['vals'].append('s')
                    else:
                        state['vals'].append('')
                elif value_input.startswith('<sa'):
                    if rdm_num > 1:
                        state['vals'].append('are')
                    else:
                        state['vals'].append('is')
                elif value_input.startswith('<sh'):
                    if rdm_num > 1:
                        state['vals'].append('have')
                    else:
                        state['vals'].append('has')
        elif 'between' in state['node']['type']:
            assert exist_num > 0
            if len(input_states) == 3 and \
                    (input_states[2]['node']['type'] == 'contain_total_multiple_filter' or
                     input_states[2]['node']['type'] == 'contain_each_unique_filter' or
                     input_states[2]['node']['type'] == 'contain_each_at_most_unique_filter' or
                     input_states[2]['node']['type'] == 'contain_between_unique_filter'):
                if input_states[2]['node']['type'] == 'contain_total_multiple_filter' or \
                        input_states[2]['node']['type'] == 'contain_each_unique_filter' or \
                        input_states[2]['node']['type'] == 'contain_each_at_most_unique_filter':
                    p_val_num = input_states[2]['vals'][0]
                    p_val_keys = input_states[2]['vals'][1:5]
                else:
                    assert "between" in input_states[2]['node']['type']
                    p_val_num = input_states[2]['vals'][1]
                    p_val_keys = input_states[2]['vals'][2:6]
                o_val_keys = input_states[1]['vals']
                # print(int(p_val_num), p_val_keys, o_val_keys)
                if key_contains(o_val_keys, p_val_keys):
                    cons_total = min(total, int(p_val_num))
                    # print('b', exist_num, 1, cons_total, total)
                    min_num, max_num, answer = between(1, total, exist_num, cons_total, None)
                else:
                    min_num, max_num, answer = between(1, total, exist_num)
            else:
                min_num, max_num, answer = between(1, total, exist_num)
            state['vals'] = list((str(min_num), str(max_num)))
            for value_input in state['node']['value_inputs']:
                if value_input.startswith('<so'):
                    if max_num > 1:
                        state['vals'].append('s')
                    else:
                        state['vals'].append('')
                elif value_input.startswith('<sa'):
                    if max_num > 1:
                        state['vals'].append('are')
                    else:
                        state['vals'].append('is')
                elif value_input.startswith('<sh'):
                    if max_num > 1:
                        state['vals'].append('have')
                    else:
                        state['vals'].append('has')
        elif 'than' in state['node']['type']:
            if 'more' in state['node']['type']:
                answer, state['vals'] = more_than_frac(exist_num, total)
            elif 'fewer' in state['node']['type']:
                answer, state['vals'] = fewer_than_frac(exist_num, total)
            else:
                raise Exception('exist no handler')
        elif 'but' in state['node']['type']:
            if 'most' in state['node']['type']:
                answer, rdm_num = all_but_at_most(exist_num, total)
            elif 'least' in state['node']['type']:
                answer, rdm_num = all_but_at_least(exist_num, total)
            else:
                raise Exception('exist no handler')
            state['vals'] = [str(rdm_num)]
            for value_input in state['node']['value_inputs']:
                if value_input.startswith('<so'):
                    if rdm_num > 1:
                        state['vals'].append('s')
                    else:
                        state['vals'].append('')
                elif value_input.startswith('<sa'):
                    if rdm_num > 1:
                        state['vals'].append('are')
                    else:
                        state['vals'].append('is')
                elif value_input.startswith('<sh'):
                    if rdm_num > 1:
                        state['vals'].append('have')
                    else:
                        state['vals'].append('has')
        else:
            raise Exception('exist no handler')
    state['answer'] = answer
    logging.debug(f'answer: {answer}')
    state['outputs'] = {'plane': state['inputs']['plane'],
                        'object': satisf_objects if len(satisf_objects) != 0 else {}}


def attributes_subset_handler(state, scene, input_states):
    assert state['node']['kind'] == 'object'
    if '_filter_options' not in scene:
        precompute_filter_options(scene)
    options = {}
    objects = set()
    total_keys = []
    for plane_idx in state['inputs']['plane']:
        for obj in scene['planes'][plane_idx]['internal_objs']:
            objects.add(obj)
    for k, v in scene['_filter_options']['object'].items():
        intersection = objects & v
        if intersection:
            total_keys.append(k)
            if len(intersection) < len(objects):
                count = 0
                for a in k:
                    if a is not None:
                        count += 1
                if count > 1:
                    options[k] = []
                    for num in range(1, count):
                        for pair in combinations([a for a in k if a is not None], num):
                            k1 = [None] * 4
                            for a1 in pair:
                                k1[k.index(a1)] = a1
                            k1 = tuple(k1)
                            if objects & scene['_filter_options']['object'][k1] == intersection:
                                options[k].append(k1)
                    if len(options[k]) == 0:
                        options.pop(k, None)
    logging.debug(f'options: {options}')
    answer1 = random.choice(list(options.keys()))

    if 'no_except' in state['node']['type']:
        if 'contain' in input_states[0]['node']['type'] and not input_states[0]['node']['type'].startswith(
                'not_contain_each'):
            if input_states[0]['node']['type'].startswith('contain_each_unique') or \
                    input_states[0]['node']['type'].startswith('contain_each_multiple') or \
                    input_states[0]['node']['type'].startswith('contain_each_at_most') or \
                    input_states[0]['node']['type'].startswith('contain_each_at_least') or \
                    input_states[0]['node']['type'].startswith('contain_total'):
                pre_vals = input_states[0]['vals'][1:5]
            elif input_states[0]['node']['type'].startswith('contain_between') or \
                    input_states[0]['node']['type'].startswith('not_contain_between'):
                pre_vals = input_states[0]['vals'][2:6]
            else:
                pre_vals = input_states[0]['vals']
            logging.debug(f'pre_vals: {pre_vals}')
            answer2_list = []
            answer3_list = []
            for a2 in options[answer1]:
                valid_a2 = [a for a in a2 if a is not None]
                a3 = tuple([k if k not in valid_a2 else None for k in answer1])
                if key_contains(pre_vals, a3):
                    valid_a3 = [a for a in a3 if a is not None]
                    cmp_vals = [p if p not in valid_a3 else None for p in pre_vals]
                    if key_contains(cmp_vals, a2) or key_contains(a2, cmp_vals):
                        answer2_list.append(a2)
                        answer3_list.append(a3)
                else:
                    answer2_list.append(a2)
                    answer3_list.append(a3)
            answer2, answer3 = random.choice(list(zip(answer2_list, answer3_list)))
        else:
            answer2 = random.choice(list(options[answer1]))
            valid_answer2 = [a for a in answer2 if a is not None]
            answer3 = tuple([k if k not in valid_answer2 else None for k in answer1])
    else:
        # every except
        answer2_list = []
        answer3_list = []
        for a2 in options[answer1]:
            valid_a2 = [a for a in a2 if a is not None]
            a3 = tuple([k if k not in valid_a2 else None for k in answer1])
            for t in total_keys:
                if t == a3:
                    continue
                flag = True
                for i in range(len(a3)):
                    if a3[i] is None and t[i] is not None:
                        flag = False
                    if a3[i] is not None and t[i] is None:
                        flag = False
                if flag:
                    answer2_list.append(a2)
                    answer3_list.append(t)
        answer2, answer3 = random.choice(list(zip(answer2_list, answer3_list)))

    answer_obj_num = len(objects & scene['_filter_options']['object'][answer2])
    logging.debug(f'answer1: {answer1}, answer2: {answer2}, answer3: {answer3}')
    if answer2[3] is None:
        answer2_rep = list(answer2)
        answer2_rep[3] = random.choice(['one', 'object'])
        state['vals'] = answer2_rep
    else:
        state['vals'] = list(answer2)
    answer3_rep = list(answer3)
    if answer3[3] is None and random.random() < 0.5:
        answer3_rep[3] = 'object'
    state['vals'].extend(answer3_rep)

    for value_input in state['node']['value_inputs']:
        if value_input.startswith('<so'):
            if answer_obj_num > 1:
                state['vals'].append('s')
            else:
                state['vals'].append('')
        elif value_input.startswith('<sn'):
            if answer3_rep[3] is not None:
                state['vals'].append('s')
            else:
                state['vals'].append('')
        elif value_input.startswith('<se'):
            if answer3_rep[3] is not None:
                state['vals'].append('a')
            else:
                state['vals'].append('')
    state['vals'].append('object')
    state['answer'] = [answer2, answer3]
    state['outputs'] = {'plane': state['inputs']['plane'], 'object': {}}


def except_handler(state, scene, input_states):
    assert state['node']['kind'] == 'object'
    k1, k2 = input_states[0]['answer']
    objects = set()
    for plane_idx in state['inputs']['plane']:
        for obj in scene['planes'][plane_idx]['internal_objs']:
            objects.add(obj)
    ob_set1 = objects & scene['_filter_options']['object'][k1]
    ob_set2 = objects & scene['_filter_options']['object'][k2]
    if 'no' in state['node']['type']:
        if ob_set1 == ob_set2:
            state['answer'] = True
        else:
            state['answer'] = False
        for o in ob_set1:
            ob_set2.remove(o)
        state['outputs'] = {'plane': state['inputs']['plane'],
                            'object': ob_set2 if len(ob_set2) != 0 else {}}
    elif 'every' in state['node']['type']:
        ob_set4 = objects.copy()
        for o2 in ob_set2:
            ob_set4.remove(o2)
        if ob_set4 == ob_set1:
            state['answer'] = True
        else:
            state['answer'] = False
        for o1 in ob_set1:
            if o1 in ob_set4:
                ob_set4.remove(o1)
        state['outputs'] = {'plane': state['inputs']['plane'],
                            'object': ob_set4 if len(ob_set4) != 0 else {}}
    else:
        raise Exception("unrecognized except handler")


def whether_handler(state, scene, input_states):
    assert state['node']['kind'] == 'object'
    obj_key = input_states[-1]['vals']
    plane_objects = set()
    assert len(input_states) == 2
    for plane_idx in input_states[0]['outputs']['plane']:
        for obj in scene['planes'][plane_idx]['internal_objs']:
            plane_objects.add(obj)

    total_objects = set()
    for obj_idx in scene['_filter_options']['object'][obj_key]:
        total_objects.add(obj_idx)

    intersection = plane_objects & total_objects
    if state['node']['type'] == 'whether_all':
        state['answer'] = all_of(len(intersection), len(total_objects))
    elif state['node']['type'] == 'whether_not_all':
        state['answer'] = not_all_of(len(intersection), len(total_objects))
    elif state['node']['type'] == 'whether_some':
        state['answer'] = some(len(intersection))
    elif state['node']['type'] == 'whether_no':
        state['answer'] = no(len(intersection))
    elif state['node']['type'] == 'whether_some_but_not_all':
        state['answer'] = some_but_not_all(len(intersection), len(total_objects))
    elif 'whether_exact' == state['node']['type'] or \
            'at_most_number' in state['node']['type'] or \
            'at_least_number' in state['node']['type'] or \
            'more_than_number' in state['node']['type'] or \
            'fewer_than_number' in state['node']['type']:
        num = len(intersection)
        total = min(len(plane_objects), len(total_objects))
        if input_states[0]['node']['type'] == 'contain_total_multiple_filter' or \
                input_states[0]['node']['type'] == 'contain_each_unique_filter':
            p_val_num = input_states[0]['vals'][0]
            p_val_keys = input_states[0]['vals'][1:5]
            o_val_keys = input_states[1]['vals']
            # print(int(p_val_num), p_val_keys, o_val_keys)
            if key_contains(p_val_keys, o_val_keys):
                cons_min = int(p_val_num)
                # print('a', num, cons_min, total)
                if 'at_most_number' in state['node']['type']:
                    state['answer'], rdm_num = at_most(num, total, cons_min)
                elif 'at_least_number' in state['node']['type']:
                    state['answer'], rdm_num = at_least(num, total, cons_min)
                elif 'whether_exact' == state['node']['type']:
                    state['answer'], rdm_num = exact(num, total, cons_min)
                elif 'more_than_number' in state['node']['type']:
                    state['answer'], rdm_num = more_than_number(num, total, cons_min)
                else:
                    state['answer'], rdm_num = fewer_than_number(num, total, cons_min)
            elif key_contains(o_val_keys, p_val_keys):
                cons_total = min(total, int(p_val_num))
                # print('b', num, 1, cons_total)
                if 'at_most_number' in state['node']['type']:
                    state['answer'], rdm_num = at_most(num, cons_total)
                elif 'at_least_number' in state['node']['type']:
                    state['answer'], rdm_num = at_least(num, cons_total)
                elif 'whether_exact' == state['node']['type']:
                    state['answer'], rdm_num = exact(num, cons_total, 1, 'contain')
                elif 'more_than_number' in state['node']['type']:
                    state['answer'], rdm_num = more_than_number(num, cons_total)
                else:
                    state['answer'], rdm_num = fewer_than_number(num, cons_total)
            else:
                if 'at_most_number' in state['node']['type']:
                    state['answer'], rdm_num = at_most(num, total)
                elif 'at_least_number' in state['node']['type']:
                    state['answer'], rdm_num = at_least(num, total)
                elif 'whether_exact' == state['node']['type']:
                    state['answer'], rdm_num = exact(num, total)
                elif 'more_than_number' in state['node']['type']:
                    state['answer'], rdm_num = more_than_number(num, total)
                else:
                    state['answer'], rdm_num = fewer_than_number(num, total)
        else:
            if 'at_most_number' in state['node']['type']:
                state['answer'], rdm_num = at_most(num, total)
            elif 'at_least_number' in state['node']['type']:
                state['answer'], rdm_num = at_least(num, total)
            elif 'whether_exact' == state['node']['type']:
                state['answer'], rdm_num = exact(num, total)
            elif 'more_than_number' in state['node']['type']:
                state['answer'], rdm_num = more_than_number(num, total)
            else:
                state['answer'], rdm_num = fewer_than_number(num, total)
        state['vals'] = [str(rdm_num)]
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<so'):
                if rdm_num > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sa'):
                if rdm_num > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
    elif 'between' in state['node']['type']:
        num = len(intersection)
        total = min(len(plane_objects), len(total_objects))
        if input_states[0]['node']['type'] == 'contain_total_multiple_filter' or \
                input_states[0]['node']['type'] == 'contain_each_unique_filter':
            p_val_num = input_states[0]['vals'][0]
            p_val_keys = input_states[0]['vals'][1:5]
            o_val_keys = input_states[1]['vals']
            if key_contains(p_val_keys, o_val_keys):
                cons_min = int(p_val_num)
                min_num, max_num, state['answer'] = between(1, total, num, None, cons_min)
            elif key_contains(o_val_keys, p_val_keys):
                cons_total = min(total, int(p_val_num))
                min_num, max_num, state['answer'] = between(1, total, num, cons_total, None)
            else:
                min_num, max_num, state['answer'] = between(1, total, num)
        else:
            min_num, max_num, state['answer'] = between(1, total, num)
        state['vals'] = list((str(min_num), str(max_num)))
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<so'):
                if max_num > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sa'):
                if max_num > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sh'):
                if max_num > 1:
                    state['vals'].append('have')
                else:
                    state['vals'].append('has')
    elif 'fractions' in state['node']['type']:
        num = len(intersection)
        total = len(total_objects)
        if 'more_than_fractions' in state['node']['type']:
            state['answer'], state['vals'] = more_than_frac(num, total)
        elif 'fewer_than_fractions' in state['node']['type']:
            state['answer'], state['vals'] = fewer_than_frac(num, total)
        elif 'at_most_fractions' in state['node']['type']:
            state['answer'], state['vals'] = at_most_frac(num, total)
        elif 'at_least_fractions' in state['node']['type']:
            state['answer'], state['vals'] = at_least_frac(num, total)
        else:
            raise Exception('whether no handler')
    elif 'but' in state['node']['type']:
        num = len(intersection)
        total = len(total_objects)
        if 'most' in state['node']['type']:
            state['answer'], rdm_num = all_but_at_most(num, total)
        elif 'least' in state['node']['type']:
            state['answer'], rdm_num = all_but_at_least(num, total)
        else:
            raise Exception('whether no handler')
        state['vals'] = [str(rdm_num)]
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<so'):
                if rdm_num > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sa'):
                if rdm_num > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
    elif 'whether_most' in state['node']['type']:
        num = len(intersection)
        total = len(total_objects)
        state['answer'] = most(num, total)
    else:
        raise Exception('whether no handler')
    if 'but' in state['node']['type'] or \
            state['node']['type'] == 'whether_all' or \
            'fractions' in state['node']['type'] or \
            'whether_most' in state['node']['type']:
        diff_objs = total_objects ^ intersection
        state['outputs'] = {'plane': input_states[0]['outputs']['plane'],
                            'object': diff_objs if len(diff_objs) != 0 else {}}
    else:
        state['outputs'] = {'plane': input_states[0]['outputs']['plane'],
                            'object': intersection if len(intersection) != 0 else {}}


def compare_handler(state, scene, input_states):
    if state['node']['type'].startswith('more_'):
        if len(input_states[0]['outputs']['object']) > len(input_states[1]['outputs']['object']):
            state['answer'] = True
        else:
            state['answer'] = False
    elif state['node']['type'].startswith('fewer_'):
        if len(input_states[0]['outputs']['object']) < len(input_states[1]['outputs']['object']):
            state['answer'] = True
        else:
            state['answer'] = False
    elif state['node']['type'].startswith('equal_'):
        if len(input_states[0]['outputs']['object']) == len(input_states[1]['outputs']['object']):
            state['answer'] = True
        else:
            state['answer'] = False
    else:
        raise Exception('compare no answer')


def q_relate_handler(state, scene, input_states):
    assert state['node']['kind'] == 'object'
    objects = set()
    for plane_idx in state['inputs']['plane']:
        for obj in scene['planes'][plane_idx]['internal_objs']:
            objects.add(obj)

    # the unique object used to determine direction
    r_obj_id = input_states[-1]['inputs']['object']

    # all 'obj_vals' objects in the located planes
    obj_vals = tuple(input_states[-1]['vals'][1:5])
    scene_obj = scene['_filter_options']['object'][obj_vals]
    all_intersection = objects & scene_obj
    intersection = all_intersection ^ (r_obj_id & all_intersection)
    assert intersection, 'q_relate_handler no answer'
    obj_total_num = len(intersection)

    # all 'obj_vals' objects on the 'obj_r' direction of 'r_obj_id'
    obj_r_num = len(input_states[-1]['outputs']['object'])

    # all objects on the 'obj_r' direction of 'r_obj_id'
    all_objs_of_robj_num = len(input_states[-1]['total_objs_of_r'])

    logging.debug(f"all 'obj_vals' objects in the located planes: {objects & scene_obj}")
    logging.debug(f"the unique object used to determine direction: {r_obj_id}")
    logging.debug(f"all objects on the 'obj_r' direction of 'r_obj_id': {input_states[-1]['total_objs_of_r']}")
    logging.debug(f"all 'obj_vals' objects on the 'obj_r' direction of 'r_obj_id': {input_states[-1]['outputs']['object']}")
    min_num_objs = min(all_objs_of_robj_num, obj_total_num)

    if state['node']['type'].startswith('most_'):
        state['answer'] = most(obj_r_num, obj_total_num)
    elif state['node']['type'].startswith('all_'):
        state['answer'] = all_of(obj_r_num, obj_total_num)
    elif state['node']['type'].startswith('not_all_'):
        state['answer'] = not_all_of(obj_r_num, obj_total_num)
    elif state['node']['type'].startswith('no_'):
        state['answer'] = no(obj_r_num)
    elif state['node']['type'].startswith('some_but_not_all_'):
        state['answer'] = some_but_not_all(obj_r_num, obj_total_num)
    elif state['node']['type'] == 'some_relate':
        state['answer'] = some(obj_r_num)
    elif state['node']['type'].startswith('exact_') or \
            state['node']['type'].startswith('at_most_number_') or \
            state['node']['type'].startswith('at_least_number_') or \
            'more_than_number' in state['node']['type'] or \
            'fewer_than_number' in state['node']['type'] or \
            'but_' in state['node']['type']:
        if len(input_states) == 3 and \
                (input_states[0]['node']['type'] == 'contain_total_multiple_filter' or
                 input_states[0]['node']['type'] == 'contain_each_unique_filter' or
                 input_states[0]['node']['type'] == 'contain_each_at_most_unique_filter' or
                 input_states[0]['node']['type'] == 'contain_between_unique_filter'):
            if input_states[0]['node']['type'] == 'contain_total_multiple_filter' or \
                    input_states[0]['node']['type'] == 'contain_each_unique_filter' or \
                    input_states[0]['node']['type'] == 'contain_each_at_most_unique_filter':
                p_val_num = input_states[0]['vals'][0]
                p_val_keys = input_states[0]['vals'][1:5]
            else:
                assert "between" in input_states[0]['node']['type']
                p_val_num = input_states[0]['vals'][1]
                p_val_keys = input_states[0]['vals'][2:6]
            o_val_keys = input_states[1]['vals']
            o_of_r_val_keys = input_states[-1]['vals'][1:5]
            # print(int(p_val_num), p_val_keys, o_val_keys, o_of_r_val_keys)
            if key_contains(o_of_r_val_keys, p_val_keys):
                if key_contains(o_val_keys, p_val_keys):
                    cons_total = min(min_num_objs, int(p_val_num) - 2)
                    # print('c', obj_r_num, 1, cons_total)
                    if 'at_most_number_relate' == state['node']['type']:
                        state['answer'], rdm_num = at_most(obj_r_num, cons_total)
                    elif 'at_least_number_relate' == state['node']['type']:
                        state['answer'], rdm_num = at_least(obj_r_num, cons_total)
                    elif 'exact_relate' == state['node']['type']:
                        state['answer'], rdm_num = exact(obj_r_num, cons_total, 1, 'contain')
                    elif 'more_than_number_relate' == state['node']['type']:
                        state['answer'], rdm_num = more_than_number(obj_r_num, cons_total)
                    elif "fewer_than_number_relate" == state['node']['type']:
                        state['answer'], rdm_num = fewer_than_number(obj_r_num, cons_total)
                    elif 'but_most_relate' == state['node']['type']:
                        cons_total = min(obj_total_num, int(p_val_num) - 2)
                        state['answer'], rdm_num = all_but_at_most(obj_r_num, cons_total)
                    elif 'but_least_relate' == state['node']['type']:
                        cons_total = min(obj_total_num, int(p_val_num) - 2)
                        state['answer'], rdm_num = all_but_at_least(obj_r_num, cons_total)
                    else:
                        raise Exception('q_relate no handler')
                else:
                    cons_total = min(min_num_objs, int(p_val_num) - 1)
                    # print('b', obj_r_num, 1, cons_total)
                    if 'at_most_number_relate' == state['node']['type']:
                        state['answer'], rdm_num = at_most(obj_r_num, cons_total)
                    elif 'at_least_number_relate' == state['node']['type']:
                        state['answer'], rdm_num = at_least(obj_r_num, cons_total)
                    elif 'exact_relate' == state['node']['type']:
                        state['answer'], rdm_num = exact(obj_r_num, cons_total, 1, 'contain')
                    elif 'more_than_number_relate' == state['node']['type']:
                        state['answer'], rdm_num = more_than_number(obj_r_num, cons_total)
                    elif "fewer_than_number_relate" == state['node']['type']:
                        state['answer'], rdm_num = fewer_than_number(obj_r_num, cons_total)
                    elif 'but_most_relate' == state['node']['type']:
                        cons_total = min(obj_total_num, int(p_val_num) - 1)
                        state['answer'], rdm_num = all_but_at_most(obj_r_num, cons_total)
                    elif 'but_least_relate' == state['node']['type']:
                        cons_total = min(obj_total_num, int(p_val_num) - 1)
                        state['answer'], rdm_num = all_but_at_least(obj_r_num, cons_total)
                    else:
                        raise Exception('q_relate no handler')
            else:
                if 'at_most_number_relate' == state['node']['type']:
                    state['answer'], rdm_num = at_most(obj_r_num, min_num_objs)
                elif 'at_least_number_relate' == state['node']['type']:
                    state['answer'], rdm_num = at_least(obj_r_num, min_num_objs)
                elif 'exact_relate' == state['node']['type']:
                    state['answer'], rdm_num = exact(obj_r_num, min_num_objs)
                elif 'more_than_number_relate' == state['node']['type']:
                    state['answer'], rdm_num = more_than_number(obj_r_num, min_num_objs)
                elif "fewer_than_number_relate" == state['node']['type']:
                    state['answer'], rdm_num = fewer_than_number(obj_r_num, min_num_objs)
                elif 'but_most_relate' == state['node']['type']:
                    state['answer'], rdm_num = all_but_at_most(obj_r_num, obj_total_num)
                elif 'but_least_relate' == state['node']['type']:
                    state['answer'], rdm_num = all_but_at_least(obj_r_num, obj_total_num)
                else:
                    raise Exception('q_relate no handler')
        else:
            if 'at_most_number_relate' == state['node']['type']:
                state['answer'], rdm_num = at_most(obj_r_num, min_num_objs)
            elif 'at_least_number_relate' == state['node']['type']:
                state['answer'], rdm_num = at_least(obj_r_num, min_num_objs)
            elif 'exact_relate' == state['node']['type']:
                state['answer'], rdm_num = exact(obj_r_num, min_num_objs)
            elif 'more_than_number_relate' == state['node']['type']:
                state['answer'], rdm_num = more_than_number(obj_r_num, min_num_objs)
            elif "fewer_than_number_relate" == state['node']['type']:
                state['answer'], rdm_num = fewer_than_number(obj_r_num, min_num_objs)
            elif 'but_most_relate' == state['node']['type']:
                state['answer'], rdm_num = all_but_at_most(obj_r_num, obj_total_num)
            elif 'but_least_relate' == state['node']['type']:
                state['answer'], rdm_num = all_but_at_least(obj_r_num, obj_total_num)
            else:
                raise Exception('q_relate no handler')
        state['vals'] = [str(rdm_num)]
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<so'):
                if rdm_num > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sa'):
                if rdm_num > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
    elif state['node']['type'].startswith('between_'):
        if len(input_states) == 3 and \
                (input_states[0]['node']['type'] == 'contain_total_multiple_filter' or
                 input_states[0]['node']['type'] == 'contain_each_unique_filter' or
                 input_states[0]['node']['type'] == 'contain_each_at_most_unique_filter' or
                 input_states[0]['node']['type'] == 'contain_between_unique_filter'):
            if input_states[0]['node']['type'] == 'contain_total_multiple_filter' or \
                    input_states[0]['node']['type'] == 'contain_each_unique_filter' or \
                    input_states[0]['node']['type'] == 'contain_each_at_most_unique_filter':
                p_val_num = input_states[0]['vals'][0]
                p_val_keys = input_states[0]['vals'][1:5]
            else:
                assert "between" in input_states[0]['node']['type']
                p_val_num = input_states[0]['vals'][1]
                p_val_keys = input_states[0]['vals'][2:6]
            o_val_keys = input_states[1]['vals']
            o_of_r_val_keys = input_states[-1]['vals'][1:5]
            if key_contains(o_of_r_val_keys, p_val_keys):
                if key_contains(o_val_keys, p_val_keys):
                    cons_total = min(min_num_objs, int(p_val_num) - 2)
                    min_num, max_num, state['answer'] = between(1, min_num_objs, obj_r_num, cons_total, None)
                else:
                    cons_total = min(min_num_objs, int(p_val_num) - 1)
                    min_num, max_num, state['answer'] = between(1, min_num_objs, obj_r_num, cons_total, None)
            else:
                min_num, max_num, state['answer'] = between(1, min_num_objs, obj_r_num)
        else:
            min_num, max_num, state['answer'] = between(1, min_num_objs, obj_r_num)
        state['vals'] = list((str(min_num), str(max_num)))
        for value_input in state['node']['value_inputs']:
            if value_input.startswith('<so'):
                if max_num > 1:
                    state['vals'].append('s')
                else:
                    state['vals'].append('')
            elif value_input.startswith('<sa'):
                if max_num > 1:
                    state['vals'].append('are')
                else:
                    state['vals'].append('is')
            elif value_input.startswith('<sh'):
                if max_num > 1:
                    state['vals'].append('have')
                else:
                    state['vals'].append('has')
    elif 'fractions' in state['node']['type']:
        if 'more_than_fractions' in state['node']['type']:
            state['answer'], state['vals'] = more_than_frac(obj_r_num, obj_total_num)
        elif 'fewer_than_fractions' in state['node']['type']:
            state['answer'], state['vals'] = fewer_than_frac(obj_r_num, obj_total_num)
        elif 'at_most_fractions' in state['node']['type']:
            state['answer'], state['vals'] = at_most_frac(obj_r_num, obj_total_num)
        elif 'at_least_fractions' in state['node']['type']:
            state['answer'], state['vals'] = at_least_frac(obj_r_num, obj_total_num)
        else:
            raise Exception('q_relate no handler')
    else:
        raise Exception('q_relate no handler')

    state['outputs'] = {'plane': input_states[0]['outputs']['plane'],
                        'object': intersection if len(intersection) != 0 else {}}


def compare_size_handler(state, scene, input_states):
    assert state['node']['kind'] == 'object'
    node_type = state['node']['type']
    q1, cmp, q2 = node_type.split('_')

    def large_small_num(a, size):
        num = 0
        total_num = len(input_states[a]['outputs']['object'])
        size_list = {}
        valid_objs = []
        for obj_idx in input_states[a]['outputs']['object']:
            size_list[obj_idx] = scene['objects'][obj_idx]['size']
            if scene['objects'][obj_idx]['size'] == size:
                valid_objs.append(obj_idx)
                num += 1
        logging.debug(f"objects size: {total_num, num, size_list}")
        return num, total_num, valid_objs

    if cmp == 'larger':
        left_num, left_total_num, left_valid_objs = large_small_num(0, 'large')
        right_num, right_total_num, right_valid_objs = large_small_num(1, 'small')
    else:
        left_num, left_total_num, left_valid_objs = large_small_num(0, 'small')
        right_num, right_total_num, right_valid_objs = large_small_num(1, 'large')

    def compare_size(q, num, total, dirt):
        assert dirt == 'right' or dirt == 'left'
        vals = []
        if q == 'all':
            flag = all_of(num, total)
        elif q == 'some':
            flag = some(num)
        elif q == 'someButNotAll':
            flag = some_but_not_all(num, total)
        elif q == 'atMostNum':
            flag, rdm_num = at_most(num, total)
            vals.append(str(rdm_num))
            if dirt == 'left':
                for value_input in state['node']['value_inputs'][:3]:
                    if value_input.startswith('<so'):
                        if rdm_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if rdm_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
            else:
                for value_input in state['node']['value_inputs'][-3:]:
                    if value_input.startswith('<so'):
                        if rdm_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if rdm_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
        elif q == 'atLeastNum':
            flag, rdm_num = at_least(num, total)
            vals.append(str(rdm_num))
            if dirt == 'left':
                for value_input in state['node']['value_inputs'][:3]:
                    if value_input.startswith('<so'):
                        if rdm_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if rdm_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
            else:
                for value_input in state['node']['value_inputs'][-3:]:
                    if value_input.startswith('<so'):
                        if rdm_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if rdm_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
        elif q == 'exact':
            flag, rdm_num = exact(num, total)
            vals.append(str(rdm_num))
            if dirt == 'left':
                for value_input in state['node']['value_inputs'][:3]:
                    if value_input.startswith('<so'):
                        if rdm_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if rdm_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
            else:
                for value_input in state['node']['value_inputs'][-3:]:
                    if value_input.startswith('<so'):
                        if rdm_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if rdm_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
        elif q == 'between':
            min_num, max_num, flag = between(1, total, num)
            vals.extend([str(min_num), str(max_num)])
            if dirt == 'left':
                for value_input in state['node']['value_inputs'][:4]:
                    if value_input.startswith('<so'):
                        if max_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if max_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
            else:
                for value_input in state['node']['value_inputs'][-4:]:
                    if value_input.startswith('<so'):
                        if max_num > 1:
                            vals.append('s')
                        else:
                            vals.append('')
                    elif value_input.startswith('<sa'):
                        if max_num > 1:
                            vals.append('are')
                        else:
                            vals.append('is')
        elif 'Than' in q or 'Fractions' in q:
            if 'Fractions' in q:
                if 'moreThanFractions' in state['node']['type']:
                    flag, val = more_than_frac(num, total)
                elif 'fewerThanFractions' in state['node']['type']:
                    flag, val = fewer_than_frac(num, total)
                elif 'atMostFractions' in state['node']['type']:
                    flag, val = at_most_frac(num, total)
                elif 'atLeastFractions' in state['node']['type']:
                    flag, val = at_least_frac(num, total)
                else:
                    raise Exception('whether no handler')
                vals.append(val[0])
            else:
                if 'moreThanNumber' in state['node']['type']:
                    flag, rdm_num = more_than_number(num, total)
                elif 'fewerThanNumber' in state['node']['type']:
                    flag, rdm_num = fewer_than_number(num, total)
                else:
                    raise Exception('whether no answer')
                vals.append(str(rdm_num))
                if dirt == 'left':
                    for value_input in state['node']['value_inputs'][:3]:
                        if value_input.startswith('<so'):
                            if rdm_num > 1:
                                vals.append('s')
                            else:
                                vals.append('')
                        elif value_input.startswith('<sa'):
                            if rdm_num > 1:
                                vals.append('are')
                            else:
                                vals.append('is')
                else:
                    for value_input in state['node']['value_inputs'][-3:]:
                        if value_input.startswith('<so'):
                            if rdm_num > 1:
                                vals.append('s')
                            else:
                                vals.append('')
                        elif value_input.startswith('<sa'):
                            if rdm_num > 1:
                                vals.append('are')
                            else:
                                vals.append('is')
        # elif 'But' in q:
        #     if 'allButAtMost' in q:
        #         flag, rdm_num = all_but_at_most(num, total)
        #     elif 'allButAtLeast' in q:
        #         flag, rdm_num = all_but_at_least(num, total)
        #     else:
        #         raise Exception('whether no handler')
        #     vals.append(str(rdm_num))
        #     if dirt == 'left':
        #         for value_input in state['node']['value_inputs'][:3]:
        #             if value_input.startswith('<so'):
        #                 if rdm_num > 1:
        #                     vals.append('s')
        #                 else:
        #                     vals.append('')
        #             elif value_input.startswith('<sa'):
        #                 if rdm_num > 1:
        #                     vals.append('are')
        #                 else:
        #                     vals.append('is')
        #     else:
        #         for value_input in state['node']['value_inputs'][-3:]:
        #             if value_input.startswith('<so'):
        #                 if rdm_num > 1:
        #                     vals.append('s')
        #                 else:
        #                     vals.append('')
        #             elif value_input.startswith('<sa'):
        #                 if rdm_num > 1:
        #                     vals.append('are')
        #                 else:
        #                     vals.append('is')
        else:
            raise ValueError('no compare size handler')
        return flag, vals

    flag_left, vals_left = compare_size(q1, left_num, left_total_num, 'left')
    flag_right, vals_right = compare_size(q2, right_num, right_total_num, 'right')
    if len(vals_left) or len(vals_right):
        state['vals'] = []
        state['vals'].extend(vals_left)
        state['vals'].extend(vals_right)
    if flag_left and flag_right:
        state['answer'] = True
    else:
        state['answer'] = False

    state['out_program'] = [left_valid_objs, right_valid_objs]


def exact(num, total, min_num=1, flag=None):
    if random.random() < 0.5:
        if flag is not None:
            rdm_num = random.randint(max(min_num, num - 4), min(total, num + 4))
        else:
            if random.random() < 0.5:
                rdm_num = random.randint(max(min_num, num - 4), min(total + 2, num + 4))
            else:
                rdm_num = random.randint(max(min_num, num - 4), min(total, num + 4))
    else:
        rdm_num = random.randint(min_num, total)
    answer = True if num == rdm_num else False
    return answer, rdm_num


def between(first, second, num, max_left=None, min_right=None):
    assert num > 0
    assert second - first > 2
    if max_left is not None or min_right is not None:
        if min_right is not None:
            assert max_left is None
            if random.random() < 0.5:
                rand_num1 = random.randint(max(min_right, int(num / 2)), min(second, num * 2))
            else:
                rand_num1 = random.randint(min_right, second)
        else:
            assert min_right is None
            if random.random() < 0.5:
                rand_num1 = random.randint(max(first, int(num / 2)), min(max_left, num * 2))
            else:
                rand_num1 = random.randint(first, max_left)
    else:
        if random.random() < 0.5:
            rand_num1 = random.randint(max(first, int(num / 2)), min(second, num * 2))
        else:
            rand_num1 = random.randint(first, second)

    if random.random() < 0.5:
        rand_num2 = random.randint(max(first, int(num / 2)), min(second, num * 2))
    else:
        rand_num2 = random.randint(first, second)
    while abs(rand_num2 - rand_num1) <= 1 or (rand_num1 in [first, second] and rand_num2 in [first, second]):
        if random.random() < 0.5:
            rand_num2 = random.randint(max(first, int(num / 2)), min(second, num * 2))
        else:
            rand_num2 = random.randint(first, second)
    max_num = rand_num1 if rand_num1 > rand_num2 else rand_num2
    min_num = rand_num1 if rand_num1 < rand_num2 else rand_num2
    answer = True if min_num <= num <= max_num else False
    return min_num, max_num, answer


def more_than_number(num, total, min_num=1):
    assert num > 0
    if random.random() < 0.5:
        rdm_num = random.randint(max(min_num, num - 4), min(total - 1, num + 4))
    else:
        rdm_num = random.randint(min_num, total - 1)
    answer = True if num > rdm_num else False
    return answer, rdm_num


def fewer_than_number(num, total, min_num=1):
    assert num > 0
    if random.random() < 0.5:
        rdm_num = random.randint(max(min_num + 1, num - 4), min(total, num + 4))
    else:
        rdm_num = random.randint(min_num + 1, total)
    answer = True if num < rdm_num else False
    return answer, rdm_num


def at_most(num, total, min_num=1):
    if random.random() < 0.5:
        rdm_num = random.randint(max(min_num, num - 4), min(total - 1, num + 4))
    else:
        rdm_num = random.randint(min_num, total - 1)
    answer = True if num <= rdm_num else False
    return answer, rdm_num


def at_least(num, total, min_num=1):
    if random.random() < 0.5:
        rdm_num = random.randint(max(min_num + 1, num - 4), min(total, num + 4))
    else:
        rdm_num = random.randint(min_num + 1, total)
    answer = True if num >= rdm_num else False
    return answer, rdm_num


def all_of(num, total):
    answer = True if num == total else False
    return answer


def not_all_of(num, total):
    answer = True if total - num > 0 else False
    return answer


def some(num):
    answer = True if num > 0 else False
    return answer


def some_but_not_all(num, total):
    answer = True if num > 0 and total - num > 0 else False
    return answer


def no(num):
    answer = True if num == 0 else False
    return answer


def most(num, total):
    answer = True if num > total - num else False
    return answer


def all_but_at_most(num, total, min_num=1):
    assert total > num
    rdm_num = random.randint(min_num, total - 1)
    answer = True if total - num <= rdm_num else False
    return answer, rdm_num


def all_but_at_least(num, total, min_num=1):
    assert total > num
    rdm_num = random.randint(min_num, total - 1)
    answer = True if total - num >= rdm_num else False
    return answer, rdm_num


def more_than_frac(num, total):
    frac = get_frac(num, total)
    answer = True if Fraction(num, total) > frac else False
    # print(frac.numerator, frac.denominator)
    return answer, fraction_to_words(frac.numerator, frac.denominator)


def fewer_than_frac(num, total):
    frac = get_frac(num, total)
    answer = True if Fraction(num, total) < frac else False
    # print(frac.numerator, frac.denominator)
    return answer, fraction_to_words(frac.numerator, frac.denominator)


def at_most_frac(num, total):
    frac = get_frac(num, total)
    answer = True if Fraction(num, total) <= frac else False
    # print(frac.numerator, frac.denominator)
    return answer, fraction_to_words(frac.numerator, frac.denominator)


def at_least_frac(num, total):
    frac = get_frac(num, total)
    answer = True if Fraction(num, total) >= frac else False
    # print(frac.numerator, frac.denominator)
    return answer, fraction_to_words(frac.numerator, frac.denominator)


def fraction_to_words(numerator, denominator, ratio=0.5):
    numerator_map, denominator_map = generate_fraction_map()
    if random.random() < ratio:
        return [str(numerator) + '/' + str(denominator)]
    else:
        if numerator > 1:
            d_str = f'{denominator_map[denominator]}s'
        else:
            d_str = denominator_map[denominator]
        if numerator_map[numerator] == 'a ' and denominator_map[denominator][0] == 'e':
            n_str = 'an '
        else:
            n_str = numerator_map[numerator]
        key = n_str + d_str
        if key not in fraction_dict:
            fraction_dict[key] = str(numerator) + '/' + str(denominator)
        return [n_str + d_str]


def get_frac(num, total):
    assert num > 0
    if num == total:
        denominator = random.randint(2, 10)
        if random.random() < 0.5:
            numerator = random.randint(int(denominator / 2), denominator - 1)
        else:
            numerator = random.randint(1, denominator - 1)
    else:
        # denominator = total if total > 5 else 2 * total
        denominator = random.randint(max(num + 1, total - 3), total + 3) if total > 5 else random.randint(
            max(2 * num + 1, 2 * total - 3), 2 * total + 3)
        exact_num = num if total > 5 else 2 * num
        if random.random() < 0.5:
            numerator = random.randint(max(1, int(exact_num / 2)), min(denominator - 1, exact_num * 2))
        else:
            numerator = random.randint(1, denominator - 1)
    assert numerator < denominator
    frac = Fraction(numerator, denominator)
    return frac


def contain_between(state, scene):
    options = {'plane': {}, 'object': set()}
    obj_num_plane_dict = {}
    for plane_idx in state['inputs']['plane']:
        plane = scene['planes'][plane_idx]
        for k, v in scene['_filter_options']['object'].items():
            intersection = set(plane['internal_objs']) & v
            num = len(intersection)
            if k not in obj_num_plane_dict:
                obj_num_plane_dict[k] = {num: {plane_idx}}
            else:
                if num not in obj_num_plane_dict[k]:
                    obj_num_plane_dict[k][num] = {plane_idx}
                else:
                    obj_num_plane_dict[k][num].add(plane_idx)
    for key in obj_num_plane_dict:
        p = dict(sorted(obj_num_plane_dict[key].items()))
        num_list = list(p.keys())
        num_min = num_list[0]
        num_max = num_list[-1]
        num_left = max(0, num_min - 1)
        while num_left <= num_max:
            num_right = num_left + 2
            while num_right <= num_max + 1:
                new_key = tuple([num_left, num_right, *key])
                for i in num_list:
                    if num_left <= i <= num_right:
                        if new_key not in options['plane']:
                            options['plane'][new_key] = {*obj_num_plane_dict[key][i]}
                        else:
                            options['plane'][new_key].update({*obj_num_plane_dict[key][i]})
                num_right += 1
            num_left += 1

    return options


def contain_at_most(options):
    new_options = {'plane': {}, 'object': set()}
    for key1 in options['plane']:
        num1 = key1[0]
        obj1 = key1[1:]
        for key2 in options['plane']:
            if key2 == key1:
                continue
            num2 = key2[0]
            obj2 = key2[1:]
            if obj1 != obj2:
                continue
            for num3 in range(min(num1, num2), max(num1, num2) + 1):
                key3 = tuple([num3, *obj2])
                if num1 <= num3:
                    if key3 not in new_options['plane']:
                        new_options['plane'][key3] = options['plane'][key1].copy()
                    else:
                        for plane_idx in options['plane'][key1]:
                            new_options['plane'][key3].add(plane_idx)
                if num2 <= num3:
                    if key3 not in new_options['plane']:
                        new_options['plane'][key3] = options['plane'][key2].copy()
                    else:
                        for plane_idx in options['plane'][key2]:
                            new_options['plane'][key3].add(plane_idx)
    return new_options


def contain_at_least(options):
    new_options = {'plane': {}, 'object': set()}
    for key1 in options['plane']:
        num1 = key1[0]
        obj1 = key1[1:]
        for key2 in options['plane']:
            if key2 == key1:
                continue
            num2 = key2[0]
            obj2 = key2[1:]
            if obj1 != obj2:
                continue
            for num3 in range(min(num1, num2), max(num1, num2) + 1):
                key3 = tuple([num3, *obj2])
                if num1 >= num3:
                    if key3 not in new_options['plane']:
                        new_options['plane'][key3] = options['plane'][key1].copy()
                    else:
                        for plane_idx in options['plane'][key1]:
                            new_options['plane'][key3].add(plane_idx)
                if num2 >= num3:
                    if key3 not in new_options['plane']:
                        new_options['plane'][key3] = options['plane'][key2].copy()
                    else:
                        for plane_idx in options['plane'][key2]:
                            new_options['plane'][key3].add(plane_idx)
    return new_options


def contain_some(state, scene):
    options = {'plane': {}, 'object': set()}
    for plane_idx in state['inputs']['plane']:
        plane = scene['planes'][plane_idx]
        for k, v in scene['_filter_options']['object'].items():
            intersection = set(plane['internal_objs']) & v
            num = len(intersection)
            if num > 0:
                if k not in options['plane']:
                    options['plane'][k] = {plane_idx}
                else:
                    options['plane'][k].add(plane_idx)
    return options
