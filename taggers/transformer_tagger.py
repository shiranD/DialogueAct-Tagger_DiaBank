import os
import pickle
from .dialogue_act_tagger import DialogueActTagger
from corpora.taxonomy import Tag
from corpora.corpus import Utterance
from typing import List
from config import TransformerConfig
from typing import Optional, Union
import json
import logging

from torchtext.legacy.data import Field
import torch
from utils import DataFrameDataset
import pandas as pd
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from corpora.taxonomy import Taxonomy
import sys
#sys.exit()
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class BERT(nn.Module):
    """
    The BERT module from Hugging Face provides a pre-trained transformer with
    BERT sentence embeddings
    """

    def __init__(self, n_classes):
        super(BERT, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.num_labels = n_classes
        self.encoder = BertForSequenceClassification(config)

    def forward(self, text, label):
        print("fw")
        print(text[0][0].size())
        print(text.size())
        print(label.size())
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea


class TransformerTagger(DialogueActTagger):
    @staticmethod
    def from_folder(folder: str) -> "TransformerTagger":
        with open(f"{folder}/config.json") as f:
            config = json.load(f)
        return TransformerConfig.from_dict(config)
        #return TransformerTagger(TransformerConfig.from_dict(config))

    #def __init__(self, cfg: TransformerConfig):
    def __init__(self, cfg_path: str):
        #DialogueActTagger.__init__(self, cfg)
        self.config = self.from_folder(cfg_path)
        #self.acceptance_threshold = cfg.acceptance_threshold
        self.models = {}
        self.history: List[Utterance] = []
        path1="/projects/shdu9019/DA_tagger/DialogueAct-Tagger_DiaBank/models/transformer_example/"
        self.config.pipeline_files = ["dimension", "comm_1", "comm_2", "comm_3", "comm_0"]
        #self.config.pipeline_files = ["comm_1", "comm_2", "comm_3"]
        for pipeline in self.config.pipeline_files:
            try:
                if "dimension" in pipeline: 
                    dimension_values = list(self.config.taxonomy.value.get_dimension_taxonomy().values().keys())
                    model = BERT(len(dimension_values)).to(self.config.device)
                    pipeline_path = path1+pipeline
                    self.load_checkpoint(pipeline_path, model, self.config.device)
                    self.models[pipeline] = model
                if "comm" in pipeline: 
                    # get comm values
                    comm_values = self.config.comm_n_cls[pipeline] 
                    if comm_values == {}:
                        comm_values = {"comm_0": 0, "comm_1": 7, "comm_2": 4, "comm_3": 2} 
                    model = BERT(comm_values).to(self.config.device)
                    pipeline_path = path1+pipeline
                    self.load_checkpoint(pipeline_path, model, self.config.device)
                    self.models[pipeline] = model
                 
            except OSError:
                logging.error(
                    "The model folder does not contain the required models to run the DA tagger"
                )
                logging.error(
                    "Please run the train() method of the "
                    "DialogueActTrain class to obtain the required models"
                )
                exit(1)

    @staticmethod
   #def load_checkpoint(model, load_path, device):
    def load_checkpoint(model_path, model, device):
        """
        Support method to load a checkpoint into a pytorch model
        :param model: the model to update
        :param load_path: path of the checkpoint
        :param device: device used for the training
        :return:
        """
        if model_path is None:
            return
        state_dict = torch.load(model_path+".pt", map_location=device)
        logger.info(f"Model loaded from <== {model_path}")
        model.load_state_dict(state_dict["model_state_dict"])
        return state_dict["valid_loss"]

    @staticmethod
    def build_features(tagged_utterances: List[Utterance], config: TransformerConfig):
        label_field = Field(
            sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
        )
        text_field = Field(
            use_vocab=False,
            tokenize=config.tokenizer.encode,
            lower=False,
            include_lengths=False,
            batch_first=True,
            fix_length=config.max_seq_len,
            pad_token=config.pad_index,
            unk_token=config.unk_index,
        )
        fields = {"Text": text_field, "Label": label_field}
        train_df = pd.DataFrame(
            [[utt.text, utt.tags[0]] for utt in tagged_utterances],
            columns=["Text", "Label"],
        )
        train_set = DataFrameDataset(train_df, fields)
        return train_set

    @staticmethod
    def stringify_tags(
        dataset: List[Utterance],
        attribute: str,
        filter_attr: Optional[str] = None,
        filter_value: Optional[str] = None,
    ):
        stringified_dataset = []
        for utterance in dataset:
            new_tags = []
            new_context = []
            for tag in utterance.tags:
                if (
                    filter_value is None
                    or getattr(tag, filter_attr).__str__() == filter_value
                ):
                    new_tags.append(getattr(tag, attribute).__str__())
            for tag in utterance.context[0].tags:
                if (
                    filter_value is None
                    or getattr(tag, filter_attr).__str__() == filter_value
                ):
                    new_context.append(getattr(tag, attribute).__str__())
            if len(new_tags) > 0:
                stringified_dataset.append(
                    Utterance(
                        speaker_id=utterance.speaker_id,
                        tags=new_tags,
                        context=new_context,
                        text=utterance.text,
                    )
                )
        return stringified_dataset

    #def tag(self, sentence: Union[Utterance, str]) -> List[Tag]:
        #if type(sentence) == str:
           # sentence = Utterance(
           #     text=sentence, tags=[], context=self.history, speaker_id=0
           # )
    def tag(self, utterance: Utterance) -> List[Tag]:
        assert type(utterance.text) == str
        tags = []
        if self.config.taxonomy == Taxonomy.ISO:
            pdb.set_trace()
            dimension_prediction = self.models["dimension"](utterance.text, utterance.tags)
            #dimension_prediction = self.models["dimension"](sentence.text)[1]
            print(dimension_prediction)
            return []
        #     features = self.build_features([sentence], self.config)[0]
        #     task_dim = self.models['dimension_task'].predict_proba(features)[0][1]
        #     som_dim = self.models['dimension_som'].predict_proba(features)[0][1]
        #     fb_dim = self.models['dimension_fb'].predict_proba((features)[0])[0][0]
        #     if task_dim > self.acceptance_threshold:
        #         tags.append(
        #             ISOTag(dimension=ISODimension.Task,
        #                    comm_function=ISOTaskFunction(self.models['comm_task'].predict(features)[0]))
        #         )
        #     if som_dim > self.acceptance_threshold:
        #         tags.append(
        #             ISOTag(dimension=ISODimension.SocialObligation,
        #                    comm_function=ISOSocialFunction(self.models['comm_som'].predict(features)[0]))
        #         )
        #     if fb_dim > self.acceptance_threshold:
        #         tags.append(
        #             ISOTag(dimension=ISODimension.Feedback,
        #                    comm_function=ISOFeedbackFunction(self.models['comm_fb'].predict(features)[0]))
        #         )
        # elif self.config.taxonomy == Taxonomy.AMI:
        #     features = SVMTagger.build_features([sentence], self.config)[0]
        #     tags.append(AMITag(comm_function=AMIFunction(self.models['comm_all'].predict(features)[0])))
        # return tags
