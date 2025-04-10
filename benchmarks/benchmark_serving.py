"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class BenchmarkMetrics:
    completed: int
    total_requests: int
    slo_attained: int
    total_input_tokens: int
    total_output_tokens: int
    slo_output_tokens: int
    benchmark_duration: float
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    slo_attainment: float
    goodput: float
    min_latency: float
    max_latency: float
    avg_latency: float
    avg_tpots: dict


def read_json_input(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)


async def get_request(
    input_requests: List[Dict[str, Any]],
    start_time: float,
) -> AsyncGenerator[Tuple[str, int, int, float], None]:
    last_emission_time = 0.0
    for request in input_requests:
        # Calculate the wait time as the difference between current and last emission time
        wait_time = (request['emission_time_ms'] - last_emission_time) / 1000
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        last_emission_time = request['emission_time_ms']

        prompt = request['prompt']
        input_len = request['input_length']
        output_len = request['output_length']
        yield prompt, input_len, output_len, request['slo_ratio']


def calculate_metrics(
    input_requests: List[Dict[str, Any]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    baseline_latency_per_token = 0.1  
) -> BenchmarkMetrics:
    
    completed = sum(1 for output in outputs if output.success)
    total_requests = len(input_requests)
    total_input_tokens = sum(request['input_length'] for request in input_requests)
    total_output_tokens = sum(len(output.itl) for output in outputs if output.success)
    
    slo_attained = 0
    slo_output_tokens = 0
    latencies = []
    avg_tpots = {}
    for i, output in enumerate(outputs):
        if output.success:
            request = input_requests[i]
            input_tokens = request['input_length']
            output_tokens = len(output.itl)
            
            decode_time = output.latency - output.ttft
            if (output_tokens != 0):
                tpot = decode_time / output_tokens
                if request['slo_ratio'] not in avg_tpots:
                    avg_tpots[request['slo_ratio']] = []
                avg_tpots[request['slo_ratio']].append(tpot)
                
                expected_latency = request['slo_ratio'] * baseline_latency_per_token if request['slo_ratio'] >= 0 else -request['slo_ratio']
                
                if tpot <= expected_latency:
                    slo_attained += 1
                    slo_output_tokens += output_tokens
                latencies.append(output.ttft)


    metrics = BenchmarkMetrics(
        completed=completed,
        total_requests=total_requests,
        slo_attained=slo_attained,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        slo_output_tokens=slo_output_tokens,
        benchmark_duration=dur_s,
        request_throughput=completed / dur_s,
        output_throughput=total_output_tokens / dur_s,
        total_token_throughput=(total_input_tokens + total_output_tokens) / dur_s,
        slo_attainment=slo_attained / total_requests,
        goodput=slo_output_tokens / dur_s,
        min_latency=min(latencies) if latencies else 0,
        max_latency=max(latencies) if latencies else 0,
        avg_latency=sum(latencies) / len(latencies) if latencies else 0,
        avg_tpots={k: sum(v) / len(v) for k, v in avg_tpots.items()}
    )

    return metrics

async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Dict[str, Any]],
    best_of: int,
    use_beam_search: bool,
    disable_tqdm: bool,
    baseline_latency_per_token: float,  
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt = input_requests[0]['prompt']
    test_input_len = input_requests[0]['input_length']
    test_output_len = input_requests[0]['output_length']
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_input_len,
        output_len=test_output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for prompt, input_len, output_len, slo_ratio in get_request(input_requests, benchmark_start_time):
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=input_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        baseline_latency_per_token=baseline_latency_per_token, 
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Total requests:", metrics.total_requests))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", metrics.benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input_tokens))
    print("{:<40} {:<10}".format("Total output tokens:", metrics.total_output_tokens))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))
    print("{:<40} {:<10.2f}".format("SLO Attainment:", metrics.slo_attainment))
    print("{:<40} {:<10.2f}".format("Goodput (tok/s):", metrics.goodput))
    print("{:<40} {:<10.2f}".format("Min Latency (s):", metrics.min_latency))
    print("{:<40} {:<10.2f}".format("Max Latency (s):", metrics.max_latency))
    print("{:<40} {:<10.2f}".format("Avg Latency (s):", metrics.avg_latency))
    print("{:<40} {}".format("Avg TPOTs (ms/token):", metrics.avg_tpots))

    print("=" * 50)

    return metrics

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    # Read the input file
    input_requests = read_json_input(args.input_file)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            disable_tqdm=args.disable_tqdm,
            baseline_latency_per_token=args.baseline_latency_per_token, 
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = len(input_requests)

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Benchmark results
        result_json["completed_requests"] = benchmark_result.completed
        result_json["total_requests"] = benchmark_result.total_requests
        result_json["benchmark_duration"] = benchmark_result.benchmark_duration
        result_json["request_throughput"] = benchmark_result.request_throughput
        result_json["output_throughput"] = benchmark_result.output_throughput
        result_json["total_token_throughput"] = benchmark_result.total_token_throughput
        result_json["slo_attainment"] = benchmark_result.slo_attainment
        result_json["goodput"] = benchmark_result.goodput
        result_json["avg_latency"] = benchmark_result.avg_latency
        result_json["avg_tpots"] = benchmark_result.avg_tpots

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the JSON input file containing requests.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--baseline-latency-per-token",
        type=float,
        default=0.1,
        help="Baseline latency per token in seconds.",
    )

    args = parser.parse_args()
    main(args)
