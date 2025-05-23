import subprocess
import argparse
import math
import time
import os
from concurrent.futures import ProcessPoolExecutor

def run_pau_add(start_id, end_id, speaker_id, config):
    """Run a single pau_add.py processing task"""
    # Build speaker-specific paths
    speaker_dir = f"speaker_{speaker_id:03d}"
    speaker_input = os.path.join(config['audio_dir'], speaker_dir)
    speaker_output = os.path.join(config['output_dir'], speaker_dir)
    ############################################################################################################
    json_path = os.path.join(speaker_input, "information/word_pau.json")    # phn_pau/ word_pau
    
    # Ensure output directory exists
    os.makedirs(speaker_output, exist_ok=True)
    
    cmd = [
        "python", config['script_path'],
        "--json_path", json_path,
        "--audio_dir", speaker_input,
        "--output_dir", speaker_output,
        "--start_id", str(start_id),
        "--end_id", str(end_id),
        "--pause_duration", str(config['pause_duration']),
        "--crossfade_ms", str(config['crossfade_ms'])
    ]
    
    print(f"Executing task: Speaker {speaker_id:03d}, ID {start_id}-{end_id}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    return process.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Run pau_add.py in parallel to process audio')
    parser.add_argument('--start_id', type=int, required=True, help='Start ID')
    parser.add_argument('--end_id', type=int, required=True, help='End ID')
    parser.add_argument('--start_speaker', type=int, default=0, help='Start speaker ID')
    parser.add_argument('--end_speaker', type=int, default=108, help='End speaker ID')
    parser.add_argument('--step_size', type=int, required=True, help='Number of IDs to process per task')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of parallel processes')
    parser.add_argument('--interval', type=float, default=2.0, help='Task start interval (seconds)')
    parser.add_argument('--script_path', type=str, required=True, help='Path to pau_add.py')
    parser.add_argument('--audio_dir', type=str, required=True, help='Base audio directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    parser.add_argument('--pause_duration', type=float, default=0.6, help='Pause duration (seconds)')
    parser.add_argument('--crossfade_ms', type=int, default=50, help='Crossfade time (milliseconds)')
    
    args = parser.parse_args()
    
    config = {
        'script_path': args.script_path,
        'audio_dir': args.audio_dir,
        'output_dir': args.output_dir,
        'pause_duration': args.pause_duration,
        'crossfade_ms': args.crossfade_ms
    }
    
    total_ids = args.end_id - args.start_id + 1
    num_tasks = math.ceil(total_ids / args.step_size)
    
    print(f"Total tasks: {num_tasks}")
    print(f"Parallel processes: {args.max_workers}")
    print(f"Task start interval: {args.interval} seconds")
    
    futures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for speaker_id in range(args.start_speaker, args.end_speaker + 1):
            for i in range(num_tasks):
                task_start = args.start_id + i * args.step_size
                task_end = min(task_start + args.step_size - 1, args.end_id)
                
                future = executor.submit(run_pau_add, task_start, task_end, speaker_id, config)
                futures.append(future)
                
                time.sleep(args.interval)
    
    results = [f.result() for f in futures]
    success = results.count(True)
    failed = results.count(False)
    
    print(f"\nExecution completed:")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()

'''
# Example usage commands:

python /set your path/LLM-Dys/data_simulation/vits/word_level/batch_pau_add.py \
    --start_id 1 \
    --end_id 13000 \
    --start_speaker 0 \
    --end_speaker 108 \
    --step_size 300 \
    --max_workers 64 \
    --interval 0.01\
    --script_path /set your path/LLM-Dys/data_simulation/vits/word_level/pau_add_word.py \
    --audio_dir /set your path/LLM_dys/word/word_pau_original \
    --output_dir /set your path/LLM_dys/word/word_pau \
    --crossfade_ms 30

'''