from typing import Generator

from ctranslate2 import Translator as CTranslator
from huggingface_hub import snapshot_download
from transformers.models.nllb.tokenization_nllb_fast import NllbTokenizerFast

from server.config import Config


class Translator:
    """
    Summary
    -------
    a static class for the NLLB translator

    Methods
    -------
    translate(input: str, source_language: str, target_language: str) -> str
        translate the input from the source language to the target language
    """
    tokeniser: NllbTokenizerFast
    translator: CTranslator

    @classmethod
    def load(cls):
        """
        Summary
        -------
        download and load the model
        """
        model_path = snapshot_download('winstxnhdw/nllb-200-distilled-1.3B-ct2-int8')
        device = 'cuda' if Config.use_cuda else 'cpu'

        cls.translator = CTranslator(model_path, device=device, compute_type='auto', inter_threads=Config.worker_count)
        cls.tokeniser = NllbTokenizerFast.from_pretrained(model_path, local_files_only=True)


    @classmethod
    def translate(cls, text: str, source_language: str, target_language: str) -> Generator[str, None, None]:
        """
        Summary
        -------
        translate the input from the source language to the target language

        Parameters
        ----------
        input (str) : the input to translate
        source_language (str) : the source language
        target_language (str) : the target language

        Returns
        -------
        translated_text (str) : the translated text
        """
        cls.tokeniser.src_lang = source_language

        lines = [line for line in text.splitlines() if line]

        return (
            f'{cls.tokeniser.decode(cls.tokeniser.convert_tokens_to_ids(result.hypotheses[0][1:]))}\n'
            for result in cls.translator.translate_iterable(
                (cls.tokeniser(line).tokens() for line in lines),
                ([target_language] for _ in lines),
                beam_size=1
            )
        )
