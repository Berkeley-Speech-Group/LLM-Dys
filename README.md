# LLM-Dys

[![Demo](https://img.shields.io/badge/Demo-Listen_Online-blue)](https://anonymousmmp.github.io/LLM-Dys/)  [![Dataset](https://img.shields.io/badge/Dataset-Google_Drive-orange)](https://drive.google.com/drive/folders/14LlchEh2PJqhpewztIDh-9hUFF2AAkYr?usp=sharing)


## 🔊 Overview

**LLM-Dys** is an innovative project that leverages large language models to help  realistic dysfluent speech synthesis. Experience our technology through our [ demo](https://anonymousmmp.github.io/LLM-Dys/) showcasing various audio examples.


## 🔍 Dysfluency Types

Our system supports multiple types of dysfluency at different linguistic levels:

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
  - *Example*: "The patient needs immediate medical **attention(retention)**." 

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


## ✨ Key Features

- **Natural and authentic dysfluency patterns** leveraging advanced LLM technology
- **Comprehensive support for all dysfluency types** at both word and phoneme levels
- **Extensive dataset** with over 10,000 hours of data that can be easily scaled
- **High-quality speech synthesis** with excellent performance in evaluation metrics
- **Multiple speaker capability** through VCTK dataset integration




## 📊 Dataset

Our comprehensive dataset enables advanced research in speech synthesis:

- **Sample Dataset**: [Google Drive](https://drive.google.com/drive/folders/14LlchEh2PJqhpewztIDh-9hUFF2AAkYr?usp=sharing) (4000 samples of each type)
- **Full Dataset Size**: ~5TB （12790 hours)

### 🚀 Accessing the Complete Dataset

Due to the large size (~5TB), only sample data is directly provided. To generate the complete dataset:

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
   
   Follow other configuration steps from [VITS](https://github.com/jaywalnut310/vits)

   > **Note**: The VCTK dataset is used to generate multiple speaker variations.

## 🛠️ Data Generation Guide

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

# 🔄 Dysfluency Transcriber

### Training the Transcriber Model

```bash
# Navigate to the transcriber directory
cd dysfluency_transcriber

# Install dependencies
pip install -r environment.yml

# Prepare your training and validation datasets
# Then train the model using either the phoneme or word level script
python train_word_level.py  # For word-level transcription
# OR
python train_phn_level.py   # For phoneme-level transcription
```
<!-- ## 📝 Citation

If you use this dataset in your research, please cite:

```
@misc{LLM-Dys,
  author = {Anonymous Authors},
  title = {LLM-Dys: Dysfluent Speech Synthesis Using Large Language Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Anonymousmmp/LLM-Dys}
}
```

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or support, please [open an issue](https://github.com/Anonymousmmp/LLM-Dys/issues) on our GitHub repository. -->
