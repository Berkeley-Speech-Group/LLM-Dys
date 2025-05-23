import torch
import random
import json
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from typing import Union, List, Tuple, Optional
import os
import numpy as np
import torch
import json

class AudioSilenceInserter:
    def __init__(self, audio_path: str, output_dir: str = None):
        """
        Initialize audio processor
        
        Args:
            audio_path: Input audio path
            output_dir: Output directory path (optional)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        self.audio = AudioSegment.from_file(audio_path)
        self.original_path = audio_path
        self.output_dir = output_dir or os.path.dirname(audio_path)
        
        print("Analyzing ambient noise...")
        self.ambient_noise = self._extract_ambient_noise()
        
    def _extract_ambient_noise(self, sample_duration=200):
        """Extract longer ambient noise sample"""
        chunks = []
        chunk_size = 100
        
        for i in range(0, len(self.audio) - chunk_size, chunk_size):
            chunk = self.audio[i:i + chunk_size]
            if chunk.dBFS < -30:  # Only collect quieter segments
                chunks.append((chunk.dBFS, chunk))
                
        if not chunks:
            print("Warning: No significant ambient noise detected, using silence instead")
            return AudioSegment.silent(duration=sample_duration)
            
        # Get multiple quiet segments and concatenate
        chunks.sort(key=lambda x: x[0])
        ambient = sum([chunk[1] for chunk in chunks[:3]], AudioSegment.empty())
        return ambient[:sample_duration]

    def _create_natural_pause(self, duration_ms: int, crossfade_ms: int):
        """Create completely silent natural pause with smooth fade in and out"""
        
        # Create complete silence
        pause = AudioSegment.silent(duration=duration_ms)
        
        # Get audio data
        pause_array = np.array(pause.get_array_of_samples())
        
        # Ensure array length is a multiple of frame size
        samples_per_frame = pause.channels * pause.sample_width
        array_length = (len(pause_array) // samples_per_frame) * samples_per_frame
        pause_array = pause_array[:array_length]
        
        # Calculate fade in/out points for 22kHz sampling rate
        envelope_points = len(pause_array)
        samples_per_ms = 22  # 22kHz = 22 samples/ms
        fade_points = min(crossfade_ms * samples_per_ms, envelope_points // 4)
        
        # Create smooth fade in/out envelope
        fade_in = np.cos(np.linspace(np.pi, 2*np.pi, fade_points)) * 0.7 + 0.7 # 0->1.4
        fade_out = np.cos(np.linspace(0, np.pi, fade_points)) * 0.7 + 0.7 # 1.4->0
        
        # Build complete envelope
        envelope = np.concatenate([
            fade_in,  # Smooth fade in
            np.zeros(envelope_points - 2 * fade_points),  # Middle completely silent
            fade_out  # Smooth fade out
        ])
        
        # Apply envelope
        pause_array = (pause_array * envelope[:len(pause_array)]).astype(np.int16)
        
        # Create AudioSegment
        pause = AudioSegment(
            pause_array.tobytes(),
            frame_rate=22000,  # Explicitly specify 22kHz sampling rate
            sample_width=pause.sample_width,
            channels=pause.channels
        )
        
        return pause
        
    def _get_output_path(self, suffix=""):
        """Generate output file path"""
        path = Path(self.original_path)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{path.stem}{suffix}{path.suffix}"
        return str(output_path)

    def insert_natural_pause(self,
                            positions: Union[float, List[float], List[Tuple[float, float]]],
                            pause_duration: float = 1.0,
                            crossfade_ms: int = 50,
                            extend_duration_ms: int = 5,  # Extension area ##########previously 10
                            boost_gain_db: float = 3.0) -> str:
        """Insert precise duration pauses, extend and enhance audio before and after"""
        
        if isinstance(positions, (int, float)):
            positions = [(float(positions), pause_duration)]
        elif isinstance(positions, list):
            if not positions:
                raise ValueError("positions cannot be empty")
            if isinstance(positions[0], (int, float)):
                positions = [(p, pause_duration) for p in positions]
        
        positions = sorted(positions, key=lambda x: x[0])
        result = self.audio
        offset = 0
        
        for pos, duration in positions:
            pos_ms = int(pos * 1000)
            actual_pos = pos_ms + offset
            
            # Get audio segments before and after
            pre_audio = result[actual_pos - extend_duration_ms:actual_pos]
            post_audio = result[actual_pos:actual_pos + extend_duration_ms]
            
            # Extend audio segments (repeat 3 times)
            pre_extended = pre_audio * 3    ##############################################################
            post_extended = post_audio * 3   ##############################################################previously 5
            
            # Apply gain
            pre_extended = pre_extended + boost_gain_db
            post_extended = post_extended + boost_gain_db
            
            # Apply fade
            pre_extended = pre_extended.fade_out(crossfade_ms)
            post_extended = post_extended.fade_in(crossfade_ms)
            
            # Create silent segment
            pause = AudioSegment.silent(duration=int(duration * 1000))
            
            # Combine all parts
            first_part = result[:actual_pos - extend_duration_ms]
            last_part = result[actual_pos + extend_duration_ms:]
            
            result = (first_part + pre_extended + pause + post_extended + last_part)
            
            # Update offset
            offset += int(duration * 1000) + extend_duration_ms * 2
        
        output_path = self._get_output_path()
        result.export(output_path, format=Path(output_path).suffix.replace('.', ''))
        return output_path
    
class AudioPauseProcessor:
    def __init__(self, json_path: str, audio_base_dir: str, output_dir: str):
        """Initialize audio processor"""
        self.json_path = json_path
        self.audio_base_dir = Path(audio_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)

    def process_range(self, start_id: int, end_id: int, 
                     pause_duration: float = 0.6,
                     crossfade_ms: int = 50) -> List[str]:
        """Process audio files within specified range"""
        processed_files = []
        
        target_items = [item for item in self.data_list 
                       if start_id <= int(item['id']) <= end_id]
        
        for item in tqdm(target_items, desc="Processing audio"):
            try:
                output_path = self.process_single(item['id'], 
                                               pause_duration,
                                               crossfade_ms)
                processed_files.append(output_path)
            except Exception as e:
                print(f"Error processing ID {item['id']}: {str(e)}")
                continue
                
        return processed_files

    def process_single(self, audio_id: str,
                    crossfade_ms: int = 50) -> Optional[str]:
        """Process a single audio file"""
        # Get target data
        target_data = next((item for item in self.data_list 
                        if str(item['id']) == str(audio_id)), None)
        if not target_data:
            raise ValueError(f"Data with ID {audio_id} not found")

        # Print target information
        print(f"\nProcessing audio ID: {audio_id}")
        print(f"Insertion time points: {target_data['start_time']} -> {target_data['end_time']}")

        # Set random pause duration (range between 0.8 to 3.5 seconds)
        random_pause_duration = random.uniform(0.8, 3.5)
        print(f"Random pause duration: {random_pause_duration:.3f} seconds")

        # Get audio path
        audio_path = self.audio_base_dir / f"{audio_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        pause_position = float(target_data['start_time']) + \
                        (float(target_data['end_time']) - float(target_data['start_time'])) * 0.5
                        
        print(f"\nPause insertion position: {pause_position:.3f} seconds")
        
        processor = AudioSilenceInserter(str(audio_path), str(self.output_dir))
        output_path = processor.insert_natural_pause(pause_position, 
                                                random_pause_duration,  # Use random pause duration
                                                crossfade_ms)
        
        return output_path

if __name__ == "__main__":
    import argparse
    import logging
    from datetime import datetime
    
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Batch process audio files to insert pauses')
    parser.add_argument('--json_path', type=str, required=True, help='JSON file path with annotation data')
    parser.add_argument('--audio_dir', type=str, required=True, help='Audio file directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--start_id', type=int, default=1, help='Start ID')
    parser.add_argument('--end_id', type=int, default=None, help='End ID')
    parser.add_argument('--pause_duration', type=float, default=0.6, help='Pause duration (seconds)')
    parser.add_argument('--crossfade_ms', type=int, default=50, help='Crossfade time (milliseconds)')
    
    args = parser.parse_args()
    
    try:
        processor = AudioPauseProcessor(
            json_path=args.json_path,
            audio_base_dir=args.audio_dir,
            output_dir=args.output_dir
        )
        
        if args.end_id is None:
            args.end_id = max(int(item['id']) for item in processor.data_list)
        
        logging.info(f"Starting audio file processing (ID range: {args.start_id}-{args.end_id})")
        logging.info(f"Pause duration: {args.pause_duration} seconds")
        logging.info(f"Crossfade: {args.crossfade_ms} milliseconds")
        
        processed_files = processor.process_range(
            start_id=args.start_id,
            end_id=args.end_id,
            pause_duration=args.pause_duration,
            crossfade_ms=args.crossfade_ms
        )
        
        total_files = args.end_id - args.start_id + 1
        success_count = len(processed_files)
        
        logging.info(f"\nProcessing complete:")
        logging.info(f"Total files: {total_files}")
        logging.info(f"Successfully processed: {success_count}")
        logging.info(f"Failed: {total_files - success_count}")
        
    except Exception as e:
        logging.error(f"Error occurred during processing: {str(e)}", exc_info=True)
        raise