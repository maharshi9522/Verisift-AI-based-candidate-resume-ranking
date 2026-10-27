"""
Text Preprocessor Service for Verisift.
Generated during Stage 4B: Text Preprocessing.
Ensures identical text normalization and label encoding across training and API endpoints.
"""
import re
import unicodedata
from typing import Tuple, Dict, List

class TextPreprocessor:
    LABEL2ID: Dict[str, int] = {
        'No Fit': 0,
        'Potential Fit': 1,
        'Good Fit': 2
    }
    ID2LABEL: Dict[int, str] = {0: 'No Fit', 1: 'Potential Fit', 2: 'Good Fit'}
    
    @classmethod
    def normalize(cls, text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\ufffd', ' ')
        text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2012', '-')
        text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\u25CB\u25CF\u25AA\u25AB\uF0B7]', ' - ', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()

    @classmethod
    def encode_label(cls, label: str) -> int:
        return cls.LABEL2ID[label]

    @classmethod
    def decode_label(cls, label_id: int) -> str:
        return cls.ID2LABEL[label_id]

    @classmethod
    def preprocess_pair(cls, resume_text: str, job_text: str) -> Tuple[str, str]:
        return cls.normalize(resume_text), cls.normalize(job_text)
