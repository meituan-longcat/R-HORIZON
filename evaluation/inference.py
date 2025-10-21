import argparse 
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import os
import requests
import time

def do_post(url, data, headers, model_name):
    print(url)
    response = requests.post(url, json=data, headers=headers, timeout=(600, 600))
    retry_times = 0
    while True:
        try:
            if response is None or response.text is None:
                text = response
            else:
                text = json.loads(response.text)
        except Exception as e:
            print(f'error!! {model_name} result error: {e}')
        answer = text
        if response.status_code == 200 and answer is not None and len(answer) > 0:
            return answer
        retry_times += 1
        if retry_times > 3:
            break
        time.sleep(60)
        response = requests.post(url, json=data, headers=headers, timeout=(600, 600))
    raise Exception(f"{model_name} no result")


def request_response(key, messages, config):
    if 'params' in config:
        request_params = config['params'].copy()
    else:
        request_params = {}
    request_params['prompt'] = messages
    request_params['model'] = config['model_name']
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    url = config['base_url']
    result = do_post(url, request_params, headers, config['model_name'])
    print(result)
    text = result['choices'][0]['text']
    return {
        "key" : key, 
        "response"  : text
    }


def inference(queries, fo, config, max_workers = 5):
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key, value in queries:
            futures.append((executor.submit(request_response, key, value, config)))

        for future in tqdm(concurrent.futures.as_completed(futures)):
            result = None
            try:
                result = future.result()
                item = {
                    'key' : result['key'],
                    'response' : result['response']
                }
                with lock:
                    fo.write(json.dumps(item, ensure_ascii = False) + '\n')
                    fo.flush()
                # print(f"完成处理: {result}")
            except Exception as e:
                print(f"task fail: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.json')
    parser.add_argument('--config', type=str, default='evaluation/config.json')
    parser.add_argument('--model_name', type=str, default='THUDM/chatglm-6b')
    args = parser.parse_args()
    print(args)
    model_configs = json.load(open(args.config, 'r'))
    assert "inference" in model_configs 
    assert args.model_name in model_configs['inference']
    config = model_configs['inference'][args.model_name]

    exists = set()
    if os.path.exists(args.output):
        for line in tqdm(open(args.output)):
            item = json.loads(line)
            key = item['key']
            exists.add(key)
    print(f"load exists {len(exists)} from {args.output}")

    query_lst = []
    cnt = 0
    for line in open(args.input, 'r'):
        item = json.loads(line)
        key = item['instanceId']
        if key in exists:
            continue
        prompt = item['input']
        if "prompt_prefix" in config:
            prompt = config['prompt_prefix'] + prompt
        if "prompt_suffix" in config:
            prompt = prompt + config['prompt_suffix']
        query_lst.append((key, prompt))
        cnt += 1
    print(f"{cnt} in total, load {len(query_lst)} new queries")

    with open(args.output, "a") as fo:
        inference(query_lst, fo, config)




