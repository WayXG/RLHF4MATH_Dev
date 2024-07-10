"""
This script support vllm inference and interactions with python interpreter.
The problem solving is formulated as a multi-turn chat task. For instance, we use the mistral model as an example.

Initially, the user prompt the model to solve a problem by:

<s> [INST] Tony loved to solve difficult pen and paper puzzles.  He did a warm-up puzzle that only took 10 minutes and after that he did 2 puzzles that each took 3 times as long.  How long did he spend solving puzzles? [/INST]

In response to the prompt, the model may generate some reasoning and code. If we detect the python code wrapped in ```python and ```, we run the code, get the returned message, and return the message as a user turn response to the model.

The process repeats for at most H rounds or stops if the model outputs the final answer wrapped in \\boxed. An example is as follows, where we process the chat into the standard format and omit some code.

[ { "content": "Tony loved to solve difficult pen and paper puzzles. He did a warm-up puzzle that only took 10 minutes and after that he did 2 puzzles that each took 3 times as long. How long did he spend solving puzzles?", "role": "user" }, 
{ "content": "Let's solve this problem using Python's sympy library.\n```python\nimport sympy as sp\n\n# let's denote time in minutes spent on solving puzzle\ntime_spent_on_warmup = 10\n .... time_spent_on_puzzle_1\ntime_spent_on_puzzles\n```", "role": "assistant" }, 
{ "content": "```output\n70\n```", "role": "user" }, 
{ "content": "Thus Tony spent \\boxed{70} minutes on solving puzzles.", "role": "assistant" } ]
"""

import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import requests
from eval.evaluate import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.python_executor import PythonExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_train_prompt_format", default=True, action="store_true")
    parser.add_argument("--ports", default=["8000", "8001", "8002", "8003", "8004", "8005", "8006", "8007"])
    parser.add_argument("--stop_tokens", default=['<end_of_turn>',"<eos>", "```output", '<start_of_turn>'])
    parser.add_argument("--answer_split", default='<start_of_turn>model\n')


    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args




def prepare_data(args):
    ds = load_dataset(args.data_name, split='train')
    examples = [sample for sample in ds]
    print(examples[0], len(examples))
    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_s_{dt_string}.jsonl'
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)

    # load all processed samples
    # find the files in ./output/llm-agents/tora-code-34b-v1.0/math/
    processed_files = [f for f in os.listdir(f"{args.output_dir}/{model_name}/{args.data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]
    processed_samples = []
    for f in processed_files:
        processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{model_name}/{args.data_name}/{f}")))
    #print('aaa', f"{args.output_dir}/{model_name}/{args.data_name}/")
    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    # if example has been inferenced...
    # we can prepare a xx_idxs to control the data to be inferenced...
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    if len(examples) == 0:
        pass
    else:
        print(examples[0])
    return examples, processed_samples, out_file




def main(args):
    ports = args.ports
    examples, processed_samples, out_file = prepare_data(args)

    # init python executor
    executor = PythonExecutor(get_answer_from_stdout=True)
    SamplingParams.seed = args.seed
    
    # The model is registered as a server externally, we determine the sampling parameters here
    if len(examples) > 0:
        default_args = {
            "use_beam_search": False,
            "n": 1,
            "temperature": args.temperature,
            "max_tokens": 1024,
            "seed": args.seed,
            "top_p": 1.0,
            "top_k": -1,
            "stop": args.stop_tokens #['<end_of_turn>',"<eos>", "```output", '<start_of_turn>']
        }
    


    def query_model(prompt, args, port):
        json = {
            **args,
            "prompt": prompt,
        }
        response = requests.post(url='http://localhost' + ":" + str(port) + "/generate", json=json)
        response_json = response.json()
        return [response_json["text"][i][len(prompt) :] for i in range(len(response_json["text"]))]


    samples = []

    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, args.data_name)
        gt_cot = ''
        gt_ans = ''
        full_prompt = construct_prompt(args, example)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}
        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    print("dataset:", args.data_name, "samples:", len(samples))
    if len(samples) > 0:
        print("-" * 50)
        print("sample:", samples[0]['prompt'])
        print("-" * 50)

    # repeat n times
    remain_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]#[:100]
    all_gts = [sample['gt'] for sample in samples for _ in range(args.n_sampling)]

    tmp_idx = list(range(len(all_gts)))
    all_gts = dict(zip(tmp_idx, all_gts))

    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 6

    ################################################################
    # start inference
    start_time = time.time()
    print('The maxmial function call is ', max_func_call)
    for epoch in range(max_func_call):  # the model can interact with the external environment for at most max_func_call rounds
        print("=" * 50, "Epoch", epoch)
        current_prompts = remain_prompts
        # if all the queries meet the stop criteria, break
        if len(current_prompts) == 0:
            break

        # get all outputs, each prompt is (idx, prompt_content)
        prompts = [item[1] for item in current_prompts]

        with ThreadPoolExecutor(512) as executor2:
            result = [
                executor2.submit(query_model, prompts[i], default_args, ports[i % len(ports)]) for i in range(len(prompts))
            ]
            # use tqdm to show progress
            for _ in tqdm(as_completed(result), total=len(result)):
                pass

            outputs = [r.result()[0] for r in result]

        # process all outputs
        remain_prompts = []
        remain_codes = []
        remain_gts = []

        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            # append the y_s to the current state (history)
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                # for cot, the prompt ends for one round
                end_prompts.append((i, query))
            elif ("boxed" not in output and output.endswith("```")):
                #print("i am here11")
                # the model does not output the final answer, meanwhile, a code needs to be executed
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
                #print(program)
            else:
                # the model outputs the final answer..
                end_prompts.append((i, query))

        # execute the codes and get the results
        # note that the order of remain_codes is the same as remain_prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            # for pot, there is only one round and we use the output of the code as the final answer
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"<end_of_turn>\n<start_of_turn>user\n```output\n{exec_result}\n```<end_of_turn>\n<start_of_turn>model\n"
            query += exec_result
            #print(query)
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)
    ################################################################
    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    ans_split = args.answer_split
    codes = [prompt.split(ans_split)[-1].strip() for _, prompt in end_prompts]

    # extract preds
    # run_execute will extract the code needed to run...
    # for tora, we only extract the final answer but do not run the code
    results = [run_execute(executor, code, args.prompt_type) for code in codes]
    #results = [run_execute(executor, code, 'tora') for code in codes]
    time_use = time.time() - start_time
    tmp_to_store = [z.split("---")[-1].strip() for _, z in end_prompts]
    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
       # code = end_prompts[i:(i+1)]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        response_tmp = tmp_to_store[i*args.n_sampling: (i+1)*args.n_sampling]
        sample.pop('prompt')
        sample.update({'my_solu':response_tmp, 'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    save_jsonl(all_samples, out_file)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
                                                         
