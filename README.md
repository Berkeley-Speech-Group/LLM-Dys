# LLM-Dys

[![Demo](https://img.shields.io/badge/Demo-Listen_Online-blue)](https://Berkeley-Speech-Group.github.io/LLM-Dys/)  [![Dataset](https://img.shields.io/badge/Dataset-Google_Drive-orange)](https://drive.google.com/drive/folders/14LlchEh2PJqhpewztIDh-9hUFF2AAkYr?usp=sharing)  [![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Dataset-yellow)](https://huggingface.co/datasets/tong0/LLM_Dys)  [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2505.22029)

## Overview

LLM-Dys uses large language models for realistic dysfluent speech synthesis. Check out the [demo](https://Berkeley-Speech-Group.github.io/LLM-Dys/) to hear some examples.


## Dysfluency Types

We support dysfluencies at both word and phoneme levels:

### Word-level Dysfluencies
- **Repetition (REP)**: Repetition of single word or phrase 
  - *Example*: "The conference will feature keynote speeches **from, from** leading experts in the field."
- **Insertion (INS)**: Insertion of single word or common phrases 
  - *Example*: "Don't forget to **you know,** set your clocks forward this weekend."
- **Deletion (DEL)**: Omission of words from expected speech 
  - *Example*: "The client wants us **(to)** deliver the product by next month." 
- **Pause (PAU)**: Extended pauses between words 
  - *Example*: "The team is working hard to **&lt;pause&gt;** finish the project on time."
- **Substitution (SUB)**: Replacement of target words 
  - *Example*: "The patient needs immediate medical retention **(attention)**." 

### Phoneme-level Dysfluencies
- **Repetition (REP)**: Repetition of single syllables 
  - *Example*: "ðeɪ ɑːɹ **plˈeɪ...plˈeɪ**ɪŋ ɪnðə pˈɑːɹk." (They are playing in the park.)
- **Insertion (INS)**: Insertion of single phoneme 
  - *Example*: "ɑːɹ juː fɹˈiː ðɪs wˈiːk**m**ɛnd fɚɹə hˈaɪk?" (Are you free this weekend for a hike?)
- **Deletion (DEL)**: Omission of single phoneme
  - *Example*: "dˈɑːɹk stˈoːɹm klˈaʊdz ɡˈæðɚd k **(w)** ˈɪkli." (Dark storm clouds gathered quickly.)
- **Pause (PAU)**: Extended pauses between phonemes within a word
  - *Example*: "ʃiː ɪz pɹɪ **&lt;pause&gt;** pˈɛɹɪŋ fɚðə pɹˌɛzəntˈeɪʃən təmˈɑːɹoʊ." (She is preparing for the presentation tomorrow.)
- **Substitution (SUB)**: Replacement of single phoneme 
  - *Example*: "ˈaɪ wˈɪʃ tə wˈɑːʃ maɪ ˈaɪɹɪʃ ɹˈɪstwɑː**s(tʃ)**." (I wish to wash my Irish wristwatch.)
- **Prolongation (PRO)**: Extended duration of specific phonemes 
  - *Example*: "wiː nˈiː **&lt;prolong&gt;** d tʊ ɪmpɹˈuːv pɹədˈʌkʃən ɪfˈɪʃənsi." (We need to improve production efficiency.)


## Features

- Natural dysfluency patterns using LLMs
- Word and phoneme level dysfluencies (REP, INS, DEL, PAU, SUB, PRO)
- Large-scale dataset (~12,790 hours)
- Multiple speaker support via VCTK dataset




## Dataset

- **Sample Data**: [Google Drive](https://drive.google.com/drive/folders/14LlchEh2PJqhpewztIDh-9hUFF2AAkYr?usp=sharing) (4000 samples per type)
- **Full Dataset**: ~5TB (12,790 hours)

### Generating the Complete Dataset

The full dataset is too large to distribute directly. Generate it yourself:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anonymousmmp/LLM-Dys.git
   cd LLM-Dys
   ```

2. **Set up the environment**
   ```bash
   cd data_simulation/VITS
   pip install -r environment.yml
   ```

3. **Configure VITS**
   
   Follow the setup instructions from [VITS](https://github.com/jaywalnut310/vits)

   > Note: We use the VCTK dataset for multi-speaker generation.

## Data Generation

### Word-level Synthesis

```bash
# Standard word-level synthesis
cd word_level
# Set 'path' and 'type' variables in vctk_set_word.py and run_word.py
python run_word.py

# Pause-type synthesis
# Set 'path' in vctk_set_word_pau.py and run_word_pau.py
python run_word_pau.py
python batch_pau_add.py # Refers to example usage commands in batch_pau_add.py

# For repetition-type synthesis, we recommend using [E2-TTS](https://github.com/SWivid/F5-TTS)
```

### Phoneme-level Synthesis

```bash
# Standard phoneme-level synthesis
cd phoneme_level
# Configure all path and type variables （the same in word-level synthesis)
python run_phn.py

# Pause-type synthesis
python run_phn_pau.py
python batch_pau_add.py # Refers to example usage commands in batch_pau_add.py

# Prolongation-type synthesis
python run_phn_pro.py
```

## Dysfluency Transcriber

### Inference

1. Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1feIZcFZeKPQKwQ6_7d6ddyBFGM4mOE3h?usp=sharing)
2. Put them in `dysfluency_transcriber/finetuned_model/`
3. Run `inference.ipynb`

### Training

```bash
cd dysfluency_transcriber
pip install -r environment.yml

# Train on your own data
python train_word_level.py  # word-level
# or
python train_phn_level.py   # phoneme-level
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{zhang25u_interspeech,
  title     = {{Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection}},
  author    = {Jinming Zhang and Xuanru Zhou and Jiachen Lian and Shuhe Li and William Li and Zoe Ezzes and Rian Bogley and Lisa Wauters and Zachary Miller and Jet Vonk and Brittany Morin and Maria Gorno-Tempini and Gopala Anumanchipalli},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1853--1857},
  doi       = {10.21437/Interspeech.2025-2658},
  issn      = {2958-1796},
}
```
