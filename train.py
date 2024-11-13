#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy  # 用于深拷贝
import logging  # 用于日志记录
from dataclasses import dataclass, field  # 用于定义数据类，简化类的定义
from typing import Dict, Optional, Sequence

import torch
import transformers  # 用于加载预训练模型和分词器
import utils  # 自定义，用于加载数据
from torch.utils.data import Dataset  # 用于定义数据集
from transformers import Trainer  # 用于训练模型

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-0.5B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    smart_tokenizer_and_embedding_resize 函数的主要作用是调整 tokenizer 和模型的嵌入层，以适应新的特殊标记。
    具体来说，它包括以下几个步骤：
    1. 添加特殊标记: 将新的特殊标记添加到 tokenizer 中。
    2. 调整嵌入层大小: 调整模型的嵌入层大小，使其与新的 tokenizer 大小匹配。
    3. 初始化新嵌入: 对于新增的标记，使用已有嵌入的平均值进行初始化，以避免随机初始化带来的不稳定。
    注意，embedding layer的参数也会随着训练而更新，因此这里只是初始化了新的embedding，而不是固定了embedding。
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    # 对 strings 列表中的每个字符串进行分词
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",  # 返回 PyTorch 张量
            padding="longest",  # 将所有序列填充到最长序列的长度
            max_length=tokenizer.model_max_length,  # 设置最大长度为 tokenizer 的模型最大长度
            truncation=True,  # 如果序列超过最大长度，则进行截断
        )
        for text in strings
    ]

    # 从分词结果中提取 input_ids，并将其赋值给 input_ids 和 labels
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    # 计算每个 input_ids 中非填充标记的数量，即有效标记的长度
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id)
        .sum()
        .item()  # 判断是否为填充标记，并计算非填充标记的数量
        for tokenized in tokenized_list
    ]

    # 返回一个字典，包含 input_ids、labels、input_ids_lens 和 labels_lens
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # 将 sources 和 targets 按顺序拼接，生成 examples
    examples = [s + t for s, t in zip(sources, targets)]

    # 对 examples 和 sources 分别进行分词，生成 examples_tokenized 和 sources_tokenized
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]

    # 从 examples_tokenized 中提取 input_ids
    input_ids = examples_tokenized["input_ids"]

    # 深拷贝 input_ids，生成 labels
    labels = copy.deepcopy(input_ids)  # 深拷贝，避免修改原始数据造成影响

    # 遍历 labels 和 sources_tokenized["input_ids_lens"]，将 labels 中对应输入部分的位置设置为 IGNORE_INDEX
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    # 返回一个字典，包含 input_ids 和 labels
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    继承自 PyTorch 的 Dataset 类。它负责加载数据、格式化输入、分词，并返回输入ids和标签。
    必须实现 __len__ 和 __getitem__ 方法，以便能够使用 PyTorch 的 DataLoader 进行数据加载。
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        # 1. 加载数据: 使用 utils.jload(data_path) 从指定路径加载数据，返回一个包含多个样本的字典列表 list_data_dict。
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        # 2. 格式化输入: 根据样本是否有输入部分，选择合适的提示模板（prompt_input 或 prompt_no_input），并格式化输入。
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            (
                prompt_input.format_map(example)
                if example.get("input", "") != ""
                else prompt_no_input.format_map(example)
            )
            for example in list_data_dict
        ]

        # 3. 生成目标: 将样本的输出部分与 tokenizer.eos_token 拼接，生成目标序列。
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        # 4. 分词: 调用 preprocess 函数对输入和目标进行分词，返回包含 input_ids 和 labels 的字典 data_dict。
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        # 5. 存储数据: 将分词后的 input_ids 和 labels 存储在类的属性中。
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.
    一个用于监督微调的数据整理器（Data Collator），继承自 object。它的主要作用是将数据集中的样本整理成批次，
    并进行必要的填充和掩码操作。
    """

    # 类的属性，表示用于分词的 transformers.PreTrainedTokenizer 对象
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
        instances (Sequence[Dict]): 包含多个字典的序列，每个字典包含 input_ids 和 labels。

        Returns:
        Dict[str, torch.Tensor]: 返回一个包含 input_ids、labels 和 attention_mask 的字典，每个值都是 torch.Tensor 类型。
        """

        # 从 instances 列表中提取 input_ids 和 labels
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # 使用 torch.nn.utils.rnn.pad_sequence 函数将 input_ids 填充到相同长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # 使用 torch.nn.utils.rnn.pad_sequence 函数将 labels 填充到相同长度
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # 返回一个字典，包含 input_ids、labels 和 attention_mask
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(
                self.tokenizer.pad_token_id
            ),  # 生成注意力掩码，指示哪些位置是有效的输入，哪些位置是填充的
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # Resize tokenizer and embedding based on special tokens
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # get train_dataset, eval_dataset, data_collator
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()  # 保存训练状态为trainer_state.json，输出到training_args.output_dir
    trainer.save_model(
        training_args.output_dir
    )  # 保存模型权重到training_args.output_dir


if __name__ == "__main__":
    train()
