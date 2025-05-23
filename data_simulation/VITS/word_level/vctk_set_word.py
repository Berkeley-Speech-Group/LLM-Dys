import argparse
import matplotlib.pyplot as plt
import IPython.display as ipd
import random
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import fcntl
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

# Load model and configuration
hps = utils.get_hparams_from_file("../configs/vctk_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
net_g.cuda()
_ = net_g.eval()

# Load checkpoint
_ = utils.load_checkpoint("../pretrained/pretrained_vctk.pth", net_g, None)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)  # text_to_sequence_phn
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def synthesize_speech(text, speaker_id):
    stn_tst = get_text(text, hps)
    speaker = torch.LongTensor([speaker_id]).cuda()
    length_scale = random.uniform(1.3, 1.6)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, 
                           sid=speaker,
                           noise_scale=0.6, 
                           noise_scale_w=0.6,
                           length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
    return audio

def process_data(start_id, end_id, gpu_id, speaker_id, json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create speaker-specific output directory
    speaker_dir = os.path.join(output_dir, f'speaker_{speaker_id:03d}')
    os.makedirs(speaker_dir, exist_ok=True)
    
    for i in range(start_id, end_id + 1):
        if i >= len(data):
            break
            
        item = data[i]
        text = item['dysfluent']
        file_id = str(item['id'])
        
        try:
            audio = synthesize_speech(text, speaker_id)
            output_path = os.path.join(speaker_dir, f'{file_id}.wav')
            write(output_path, hps.data.sampling_rate, audio)
            print(f'Speaker {speaker_id} - Generated {file_id}.wav')
        except Exception as e:
            print(f"Error processing {file_id}: {str(e)}")

if __name__ == '__main__':
    type = "rep"  # [rep, del, sub]"
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, required=True)
    parser.add_argument('--end_id', type=int, required=True) 
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--speaker_id', type=int, required=True)
    parser.add_argument('--json', type=str, default=f"/set your path/LLM-Dys/labels/word_{type}.json")
    parser.add_argument('--output', type=str, default=f"/your path to save the dataset/LLM_dys/word/word_{type}")
    args = parser.parse_args()

    process_data(args.start_id, args.end_id, args.gpu_id, args.speaker_id,
                args.json, args.output)