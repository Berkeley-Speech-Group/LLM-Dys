import os
import subprocess
import time
import logging
from typing import List, Tuple
import fcntl

def generate_ranges(start: int, end: int, step: int) -> List[Tuple[int, int, int]]:
    ranges = []
    gpu_count = 1
    current_gpu = 1
    
    for i in range(start, end, step):
        ranges.append((i, min(i + step - 1, end), current_gpu))
        current_gpu = (current_gpu + 1) % gpu_count
    return ranges

def run_processes():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


    start_id = 1
    end_id = 12000  # +1
    step = 1200
    max_concurrent_processes = 10


    vits_path = "/set path here/LLM-Dys/data_simulation/vits"  # root directory of vits project
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vits_path}:{env.get('PYTHONPATH', '')}"
    
    ranges = generate_ranges(start_id, end_id, step)
    active_processes = []

    for start, end, gpu in ranges:

        while len(active_processes) >= max_concurrent_processes:
            active_processes[:] = [p for p in active_processes if p.poll() is None]
            time.sleep(3)

        cmd = f"CUDA_VISIBLE_DEVICES=7 python  /set path here/LLM-Dys/vits/word_level/vctk_set_word_pau.py --start_id {start} --end_id {end} --gpu_id {gpu}"
        logging.info(f"Starting process: {cmd}")
        
        try:
            proc = subprocess.Popen(cmd, shell=True, env=env)
            active_processes.append(proc)
            logging.info(f"Active processes: {len(active_processes)}")
            time.sleep(3)
        except Exception as e:
            logging.error(f"Failed to start process: {e}")

    while active_processes:
        active_processes[:] = [p for p in active_processes if p.poll() is None]
        if active_processes:
            logging.info(f"Waiting for {len(active_processes)} processes to complete...")
            time.sleep()
    
    logging.info("All processes completed")

if __name__ == "__main__":
    run_processes()