"""Unlearning data module."""

from pathlib import Path
from torch.utils.data import Dataset
import torch
import pandas as pd
import random

IDK_PATH = Path(__file__).with_name("idk.jsonl")

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer + tokenizer.eos_token
    num_question_tokens = len(tokenizer(question, add_special_tokens=False)['input_ids']) #this is important, we 
    encoded = tokenizer(
        full_text,
        add_special_tokens=False, #this is important, we keep false cause we already added the special tokens from template
        max_length=max_length,
        truncation=True,
    )
    input_ids = encoded['input_ids']
    pad_length = max_length - len(input_ids)
    pad_input_ids = encoded['input_ids']  + [tokenizer.pad_token_id] * pad_length
    pad_attention_mask = [1] * len(input_ids) + [0] * pad_length

    labels = list(input_ids) + [-100] * pad_length

    #change label to -100 for question tokens, including assistant header and end of header.
    for i in range(min(num_question_tokens, len(labels))): labels[i] = -100
    assert len(pad_input_ids) == max_length
    assert len(labels) == max_length
    assert len(pad_attention_mask) == max_length
    return torch.tensor(pad_input_ids),torch.tensor(labels),torch.tensor(pad_attention_mask)


class ForgetOnlyDataset(Dataset):
    def __init__(self, forget_data,
                 tokenizer,
                 max_length=512,
                 question_key = 'question',
                 answer_key = 'answer'):
        """
        Initializes the dataset for gradient ascent finetuning

        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
            template_format (str, optional): format template for structuring input
        """
        self.data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx][self.qk]
        answer = self.data.iloc[idx][self.ak]
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            question=question,
            answer=answer
        )

# TOFU implementation
class ForgetRetainDataset(Dataset):
    """
    TOFU way of implementation.

    Args:
        forget_data (pd.DataFrame): DataFrame for forgetting.
        retain_data (pd.DataFrame): DataFrame for retaining.
        tokenizer: tokenizer instance to process text.
        max_length (int): maximum sequence length.
        question_key (str): column name for questions.
        answer_key (str): column name for answers.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        forget_idx = idx
        # A new random sample is chosen every time __getitem__ is called.
        retain_idx = torch.randint(0, len(self.retain), (1,)).item()

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)


class IdkForgetOnlyDataset(ForgetOnlyDataset):
    """Forget samples paired with a preferred answer, without a retain set.

    The preferred ("alternate") answer comes from an `alternate` column when the forget
    data has one, otherwise a random line of idk.jsonl is sampled per access.
    """

    def __init__(self, *args, idk_path=None, alternate_key="alternate", **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_key = (
            alternate_key if alternate_key in self.data.columns else None
        )
        path = Path(idk_path) if idk_path else IDK_PATH
        self.idk_responses = (
            [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if self.alternate_key is None else []
        )
        if self.alternate_key is None and not self.idk_responses:
            raise ValueError(f"No alternate answers available: {path} is empty.")

    def _alternate_answer(self, idx):
        if self.alternate_key is not None:
            return self.data.iloc[idx][self.alternate_key]
        pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        return self.idk_responses[pos]

    def __getitem__(self, idx):
        forget_data = super().__getitem__(idx)

        alternate_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.data.iloc[idx][self.qk],
            self._alternate_answer(idx),
        )

        return (forget_data, alternate_data)


class IdkForgetRetainDataset(IdkForgetOnlyDataset):
    """DPO preference pairs with an independently sampled retain example."""

    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key="question", answer_key="answer", **kwargs):
        super().__init__(forget_data, tokenizer, max_length,
                         question_key, answer_key, **kwargs)
        self.retain_dataset = ForgetOnlyDataset(
            retain_data, tokenizer, max_length, question_key, answer_key
        )

    def __getitem__(self, idx):
        forget_data, alternate_data = super().__getitem__(idx)
        retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
        return forget_data, alternate_data, self.retain_dataset[retain_idx]
