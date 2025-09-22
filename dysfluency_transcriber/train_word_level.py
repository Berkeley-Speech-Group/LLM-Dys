import json
from pathlib import Path
import os
import gc
import warnings
import re
import pandas as pd
from datasets import Dataset, Audio
from transformers import (WhisperFeatureExtractor, WhisperTokenizer,
                          WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from torch.utils.data import Dataset as TorchDataset, DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import setproctitle
from tqdm import tqdm
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['HF_HOME'] = '...'

# setproctitle.setproctitle("0.005_r0.05")

def calculate_ter(prediction, label):
    pred_tokens = prediction.split()
    label_tokens = label.split()
    
    m, n = len(pred_tokens), len(label_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == label_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / max(m, n) if max(m, n) > 0 else 0

def calculate_token_distance(prediction, label):
    def get_token_positions(text):
        return [(m.group(1), m.start()) for m in re.finditer(r'\[(REP|PAU|INS)\]', text)]
    
    pred_tokens = get_token_positions(prediction)
    label_tokens = get_token_positions(label)
    
    total_distance = 0
    matched_pairs = 0
    
    for p_type, p_pos in pred_tokens:
        min_dist = float('inf')
        for l_type, l_pos in label_tokens:
            if p_type == l_type:
                dist = abs(p_pos - l_pos)
                min_dist = min(min_dist, dist)
        if min_dist != float('inf'):
            total_distance += min_dist
            matched_pairs += 1
    
    return total_distance / matched_pairs if matched_pairs > 0 else 0

def calculate_metrics_by_type(predictions, labels, token_type=None):

    total_tp = total_fp = total_fn = 0
    total_ter = 0
    total_td = 0
    valid_samples = 0
    evaluation_details = []

    for pred, label in zip(predictions, labels):
        sample_eval = {
            'prediction': pred,
            'label': label,
            'metrics': {}
        }
        
        if token_type:
            has_type_pred = bool(re.search(r'\[' + token_type + r'\]', pred))
            has_type_label = bool(re.search(r'\[' + token_type + r'\]', label))
            
            if has_type_label:
                if has_type_pred and token_type in pred:
                    tp = 1
                    fp = 0
                    fn = 0
                else:
                    tp = 0
                    fp = 0
                    fn = 1
            elif has_type_pred:
                tp = 0
                fp = 1
                fn = 0
            else:
                continue
                
        else:
            pred_tokens = [(m.group(1), m.start()) for m in re.finditer(r'\[(REP|PAU|INS)\]', pred)]
            label_tokens = [(m.group(1), m.start()) for m in re.finditer(r'\[(REP|PAU|INS)\]', label)]
            
            tp = len(set(pred_tokens) & set(label_tokens))
            fp = len(set(pred_tokens) - set(label_tokens))
            fn = len(set(label_tokens) - set(pred_tokens))

        ter = calculate_ter(pred, label)
        total_ter += ter
        
        if tp > 0:
            td = calculate_token_distance(pred, label)
            total_td += td
            valid_samples += 1
        else:
            td = 0
            
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        sample_eval['metrics'] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ter': ter,
            'td': td
        }
        evaluation_details.append(sample_eval)
            
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_ter = total_ter / len(evaluation_details) if evaluation_details else 0
    avg_td = total_td / valid_samples if valid_samples > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ter': avg_ter,
        'td': avg_td,
        'details': evaluation_details
    }




def create_dataset(csv_files, sample_ratios, valid_split=0.05, seed=42):

    all_data = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        ratio = sample_ratios.get(csv_file, 1.0)
        n_samples = int(len(df) * ratio)
        df_sampled = df.sample(n=n_samples, random_state=seed)
        
        for _, row in tqdm(df_sampled.iterrows(), total=n_samples,
                          desc=f"Processing {os.path.basename(csv_file)}"):
            item = {
                'audio': row['audio_path'],
                'sentence': row['label']
            }
            all_data.append(item)
    
    # Shuffle all the data
    all_df = pd.DataFrame(all_data).sample(frac=1, random_state=seed)
    
    # Split into train and validation
    split_idx = int(len(all_df) * (1 - valid_split))
    
    train_df = all_df.iloc[:split_idx]
    valid_df = all_df.iloc[split_idx:]
    
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    valid_dataset = Dataset.from_pandas(valid_df)
    valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return {"train": train_dataset, "validation": valid_dataset}

class SpeechDataset(TorchDataset):
    def __init__(self, dataset, feature_extractor, tokenizer):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_features = self.feature_extractor(
            item["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0)
        labels = self.tokenizer(
            item["sentence"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


metric = evaluate.load("wer")
tokenizer = None


trainer = None

def compute_metrics(pred):

    global tokenizer, trainer
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    results = {}
    metrics = {}
    
    # Calculate metrics for validation dataset
    type_results = {}
    type_counts = {
        'REP': sum(1 for label in label_str if '[REP]' in label),
        'PAU': sum(1 for label in label_str if '[PAU]' in label),
        'INS': sum(1 for label in label_str if '[INS]' in label)
    }
    total_samples = len(label_str)
    total_dys = sum(type_counts.values())
    
    # Calculate metrics for each type
    for token_type in ['REP', 'PAU', 'INS']:
        type_results[token_type.lower()] = calculate_metrics_by_type(pred_str, label_str, token_type)
    
    # Calculate weights
    type_weights = {
        'REP': type_counts['REP'] / total_dys if total_dys > 0 else 0,
        'PAU': type_counts['PAU'] / total_dys if total_dys > 0 else 0,
        'INS': type_counts['INS'] / total_dys if total_dys > 0 else 0
    }
    
    # Calculate weighted metrics
    weighted_metrics = {}
    for metric_name in ['precision', 'recall', 'f1']:
        weighted_metrics[metric_name] = sum(
            type_results[t.lower()][metric_name] * type_weights[t]
            for t in ['REP', 'PAU', 'INS']
        )
    
    for metric_name in ['ter', 'td']:
        weighted_metrics[metric_name] = sum(
            type_results[t.lower()][metric_name] * type_counts[t]
            for t in ['REP', 'PAU', 'INS']
        ) / total_samples if total_samples > 0 else 0
    
    results['validation'] = {
        'rep': type_results['rep'],
        'pau': type_results['pau'],
        'ins': type_results['ins'],
        'weighted': weighted_metrics,
        'distribution': {
            'type_counts': type_counts,
            'type_weights': type_weights,
            'total_samples': total_samples,
            'total_dys': total_dys
        }
    }
    
    metrics.update({
        'weighted_precision': weighted_metrics['precision'],
        'weighted_recall': weighted_metrics['recall'], 
        'weighted_f1': weighted_metrics['f1'],
        'weighted_ter': weighted_metrics['ter'],
        'weighted_td': weighted_metrics['td'],

        'rep_precision': type_results['rep']['precision'],
        'rep_recall': type_results['rep']['recall'],
        'rep_f1': type_results['rep']['f1'],
        'rep_ter': type_results['rep']['ter'],
        'rep_td': type_results['rep']['td'],
        
        'pau_precision': type_results['pau']['precision'],
        'pau_recall': type_results['pau']['recall'],
        'pau_f1': type_results['pau']['f1'],
        'pau_ter': type_results['pau']['ter'],
        'pau_td': type_results['pau']['td'],
        
        'ins_precision': type_results['ins']['precision'],
        'ins_recall': type_results['ins']['recall'],
        'ins_f1': type_results['ins']['f1'],
        'ins_ter': type_results['ins']['ter'],
        'ins_td': type_results['ins']['td']
    })

    try:
        if trainer and trainer.state.global_step > 0:
            checkpoint_dir = Path(trainer.args.output_dir) / f"checkpoint-{trainer.state.global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            results_file = checkpoint_dir / 'evaluation_results_validation.json'
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({'results': results}, f, indent=4, ensure_ascii=False)
            print(f"\nEvaluation results saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results to checkpoint directory: {str(e)}")

    return metrics

def main():

    csv_files_with_ratios = {
        '/data/jinming/LLM_dys/csv/train/word_ins_4000.csv': 1,
        '/data/jinming/LLM_dys/csv/train/word_rep_4000.csv': 1,
        '/data/jinming/LLM_dys/csv/train/word_pau_4000.csv': 1,      ## 240 0000*0.01=24000
        '/data/jinming/LLM_dys/csv/train/vctk.csv': 0.0135              ## 4 3800*0.03 = 1314
    }
    
    dataset = create_dataset(
        list(csv_files_with_ratios.keys()),
        sample_ratios=csv_files_with_ratios,
        valid_split=0.05
    )
    global tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", language="en", task="transcribe")
    new_tokens = ["[REP]",  "[PAU]", "[INS]"]
    tokenizer.add_tokens(new_tokens)


    train_dataset = SpeechDataset(dataset["train"], feature_extractor, tokenizer)
    valid_dataset = SpeechDataset(dataset["validation"], feature_extractor, tokenizer)


    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        
        output_dir=f"/the path to save your model/",
        per_device_train_batch_size=8, # 16 /all gpus
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=300, 
        max_steps=10000, 
        gradient_checkpointing=False, 
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8, 
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200, # 300
        eval_steps=200, # 300 
        logging_steps=100, # 25
        #report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_f1",  
        greater_is_better=True,  
        push_to_hub=False,
        dataloader_num_workers=2,
       # weight_decay=0.01,  
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    globals()['trainer'] = trainer

    processor.save_pretrained(training_args.output_dir)
    print('start training...')
    trainer.train()
    #trainer.train(resume_from_checkpoint="/data/jinming/whisper/turbo_bs16_wp1000/checkpoint-7400")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()


