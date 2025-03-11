#coding: utf-8
import os
import os.path as osp
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

from char_indexer import VanillaCharacterIndexer, BertCharacterIndexer

np.random.seed(1)
random.seed(1)

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 ):

        _data_list = [l.strip().split('|') for l in data_list]
        if validation: _data_list = random.sample(_data_list, min(100, len(_data_list)))

        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.char_indexer = VanillaCharacterIndexer()
        self.bert_char_indexer = BertCharacterIndexer()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        
        # Load original sample
        wave, text_tensor, bert_text_tensor, speaker_id = self._load_tensor(data)
        
        # Process original audio into mel spectrogram
        mel_tensor = preprocess(wave).squeeze()
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # Get reference sample from the same speaker
        same_speaker_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(same_speaker_data[:3])
        
        # Get out-of-distribution (OOD) text
        ood_text = ""
        
        while len(ood_text) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ood_text = self.ptexts[rand_idx]
            
            ood_text_indices = self.char_indexer(ood_text)
            ood_text_indices.insert(0, 0)  # Add start token
            ood_text_indices.append(0)     # Add end token

            ood_text_tensor = torch.LongTensor(ood_text_indices)
        
        return speaker_id, acoustic_feature, text_tensor, bert_text_tensor, ood_text_tensor, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000) #TODO: be careful with resampling here
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.char_indexer(text)
        bert_text = self.bert_char_indexer(text)
        
        text.insert(0, 0); bert_text.insert(0, 0)
        text.append(0); bert_text.append(0)
        
        text = torch.LongTensor(text)
        bert_text = torch.LongTensor(bert_text)

        return wave, text, bert_text, speaker_id

    def _load_data(self, data):
        wave, _, _, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[4].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        bert_texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, bert_text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            bert_texts[bid, :text_size] = bert_text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, bert_texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels

def build_dataloader(path_list, root_path, OOD_data, min_length, batch_size, num_workers, device, validation=False, collate_config={}, dataset_config={}, **kwargs):
    
    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

