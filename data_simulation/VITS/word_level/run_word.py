import os
import subprocess
import time
import logging
from typing import List, Tuple

def generate_ranges(start: int, end: int, step: int) -> List[Tuple[int, int, int]]:
    ranges = []
    gpu_count = 1
    current_gpu = 0
    
    for i in range(start, end, step):
        ranges.append((i, min(i + step - 1, end), current_gpu))
        current_gpu = (current_gpu + 1) % gpu_count
    return ranges

def wait_for_processes(processes, max_processes):
    while len(processes) >= max_processes:
        for proc in processes[:]:
            if proc.poll() is not None:
                processes.remove(proc)
        if len(processes) >= max_processes:
            time.sleep(5)

def run_processes():
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    start_id = 0
    end_id = 15000
    step = 500
    speaker_ids = range(0, 108)
    max_processes = 10  
    
    vits_path = "/set path here/LLM-Dys/data_simulation/vits"  # root directory of vits project
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vits_path}:{env.get('PYTHONPATH', '')}"
    
    for speaker_id in speaker_ids:
        ranges = generate_ranges(start_id, end_id, step)
        processes = []
        
        for start, end, gpu in ranges:

            wait_for_processes(processes, max_processes)
            
            cmd = f"CUDA_VISIBLE_DEVICES=7 python /set path here/LLM-Dys/vits/word_level/vctk_set_word.py \
                   --start_id {start} --end_id {end} --gpu_id {gpu} --speaker_id {speaker_id}"
            
            logging.info(f"Starting process for speaker {speaker_id}: {cmd}")
            try:
                proc = subprocess.Popen(cmd, shell=True, env=env)
                processes.append(proc)
                time.sleep(3)
            except Exception as e:
                logging.error(f"Error starting process: {e}")
  
        for proc in processes:
            proc.wait()
        logging.info(f"Completed all processes for speaker {speaker_id}")

if __name__ == "__main__":
    run_processes()