import os

import pytorch_lightning as pl
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer

from utils import print_msg

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Dataset(Dataset):
    def __init__(self, dataset: DatasetDict):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, idx):
        return self.dataset[idx]


class Dataloader(pl.LightningDataModule):
    def __init__(self, mode, conf):
        """_summary_
        Args:
            conf (dict): configuration dictionary
            mode (str, optional): select mode about [train] or [inference]. Defaults to "inference".
        """
        super().__init__()
        self.mode = mode
        self.doc_stride = conf.doc_stride
        self.pad_to_max_length = conf.pad_to_max_length
        self.max_length = conf.max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            conf.tokenizer_name, max_length=self.max_length, local_files_only=True
        )
        self.batch_size = conf.batch_size

        self.dataset_path = conf.train_path

        self.dataset = None  # train dataset & inference dataset
        self.val_dataset = None
        self.dataloader = None

    def load_data(self) -> DatasetDict:
        dataset = load_from_disk(self.dataset_path)
        return dataset

    def preprocessing(self, dataset):
        def set_features(column_names):
            question_column_name = "question" if "question" in column_names else column_names[0]
            context_column_name = "context" if "context" in column_names else column_names[1]
            answer_column_name = "answers" if "answers" in column_names else column_names[2]
            pad_on_right = (
                self.tokenizer.padding_side == "right"
            )  # set padding option : (question|context) or (context|question)
            features = (
                question_column_name,
                context_column_name,
                answer_column_name,
                pad_on_right,
                last_checkpoint,
                max_seq_length,
            )
            return features

        def preprocess_common(examples):
            tokenized_examples = self.tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if self.pad_to_max_length else False,
            )

            # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            return tokenized_examples, sample_mapping

        def preprocess_for_train(examples):
            tokenized_examples, sample_mapping = preprocess_common(examples)
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # 데이터셋에 "start position", "enc position" label을 부여합니다.
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]

                # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # text에서 정답의 Start/end character index
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # text에서 current span의 Start token index
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # text에서 current span의 End token index
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                        # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples

        def preprocess_for_validation(examples):
            tokenized_examples, sample_mapping = preprocess_common(examples)
            # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
            # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples

        if self.mode == "train":
            column_names = dataset["train"].column_names
            train_features = set_features(column_names)
            (
                question_column_name,
                context_column_name,
                answer_column_name,
                pad_on_right,
                last_checkpoint,
                max_seq_length,
            ) = train_features
            train_dataset = dataset["train"].map(
                preprocess_for_train,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
            )
            self.train_dataset = Dataset(train_dataset)

        column_names = dataset["validation"].column_names
        val_features = set_features(column_names)
        (
            question_column_name,
            context_column_name,
            answer_column_name,
            pad_on_right,
            last_checkpoint,
            max_seq_length,
        ) = val_features
        val_dataset = dataset["validation"].map(
            preprocess_for_validation,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
        )
        self.val_dataset = Dataset(val_dataset)

    def setup(self, stage="fit"):
        if stage == "fit":
            print_msg("Loading Dataset...", "INFO")
            dataset = self.load_data(self.data_path)
            self.preprocessing(dataset)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=default_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=default_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=default_collate,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=default_collate,
        )
