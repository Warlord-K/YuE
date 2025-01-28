import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import argparse
import torch
import numpy as np
import json
from omegaconf import OmegaConf
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf

import uuid
from tqdm import tqdm
from einops import rearrange
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import glob
import time
import copy
from collections import Counter
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched
import re

class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics






