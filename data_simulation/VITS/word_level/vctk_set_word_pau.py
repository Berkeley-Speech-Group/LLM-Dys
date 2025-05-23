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
from text import text_to_sequence, text_to_sequence_phn

from scipy.io.wavfile import write


hps = utils.get_hparams_from_file("../configs/vctk_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda(0)
_ = net_g.eval()

_ = utils.load_checkpoint("../pretrained/pretrained_vctk.pth", net_g, None)


def get_text(text, hps):
    text_norm = text_to_sequence_phn(text, hps.data.text_cleaners)  # text_to_sequence_phn
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def synthesize_speech(text, start_index, end_index, speaker_id):
    stn_tst = get_text(text, hps)
    length_scale = 1.5
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([speaker_id]).cuda()
        
        output = net_g.infer_pause(
                    x_tst, 
                    x_tst_lengths, 
                    sid=sid, 
                    noise_scale=0.6, 
                    noise_scale_w=0.6, 
                    length_scale=length_scale,
                    start_index=start_index,
                    end_index=end_index
                )
        audio = output[0][0,0].data.cpu().float().numpy()
        start_time = output[-2].item()
        end_time = output[-1].item()

    return audio, start_time, end_time

import fcntl

def process_data(start_id, end_id, gpu_id, json_path, base_output_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered_data = [item for item in data if start_id <= item['id'] <= end_id]
    

    for speaker_id in range(0,108):  
 
        speaker_dir = os.path.join(base_output_dir, f"speaker_{speaker_id:03d}") 
        audio_dir = speaker_dir
        info_dir = os.path.join(speaker_dir, "information")
        
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)
        
        final_json = os.path.join(info_dir, 'word_pau.json')
        
        if not os.path.exists(final_json):
            with open(final_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        for item in filtered_data:
            try:
                audio, start_time, end_time = synthesize_speech(
                    text=item['ipa_dysfluent'],
                    start_index=item['start_index'],
                    end_index=item['end_index'],
                    speaker_id=speaker_id
                )
                

                output_path = os.path.join(audio_dir, f"{item['id']}.wav")
                write(output_path, hps.data.sampling_rate, audio)
                

                with open(final_json, 'r+', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        final_data = json.load(f)
                        for final_item in final_data:
                            if final_item['id'] == item['id']:
                                final_item['start_time'] = float(start_time)
                                final_item['end_time'] = float(end_time)
                                break
                        
                        f.seek(0)
                        f.truncate()
                        json.dump(final_data, f, ensure_ascii=False, indent=2)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                print(f"success: speaker_{speaker_id:03d}/audio/{item['id']}.wav")
                
            except Exception as e:
                print(f"fail: speaker_{speaker_id:03d}/id_{item['id']}: {str(e)}")
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, required=True)
    parser.add_argument('--end_id', type=int, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--json', type=str, default=f"/set your path/LLM-Dys/labels/word_pau.json")
    parser.add_argument('--output', type=str, default=f"/your path to save the dataset/LLM_dys/word/word_pau_original")
    args = parser.parse_args()
    
    process_data(args.start_id, args.end_id, args.gpu_id, args.json, args.output)