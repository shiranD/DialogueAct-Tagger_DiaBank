from taggers.dialogue_act_tagger import DialogueActTagger
from taggers.transformer_tagger import BERT
from typing import List
import torch
from torchtext.legacy.data import BucketIterator
from utils import stringify_tags
from corpora.corpus import Corpus
from corpora.corpus import Utterance
from config import TransformerConfig
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import random
import json
import pdb

class DialogueActTester:
    """
    A testing utility for Dialogue Act Tagger.
    Provides comparison of different DA tagging architectures on the same test set and
    in-depth statistics on a single classifier's performances
    """

    def __init__(self, corpora: List[Corpus], cfg_path: str):
        self.test_set = []
        for corpus in corpora:
            self.test_set = self.test_set + corpus.get_test_split()
        random.shuffle(self.test_set)
        self.test_set = self.test_set[0:1000]
        self.config = self.from_folder(cfg_path)

    @staticmethod
    def from_folder(folder: str) -> "TransformerTagger":
        with open(f"{folder}/config.json") as f:
            config = json.load(f)
        return TransformerConfig.from_dict(config)

    def tag_test(self, tagger: DialogueActTagger, test_set: List[Utterance]):
        """
        Evalautes batch size test sets
        """
        pipeline = "dimension"
        model = tagger.models[pipeline]
        test_iter = BucketIterator(
        tagger.build_features(test_set, self.config),
        batch_size=self.config.batch_size,
        sort_key=lambda x: len(x.Text),
        device=self.config.device,
        train=False,
        sort=True,
        sort_within_batch=True,
        ) 
        #pdb.set_trace()
        model.eval()
        test_running_loss = 0
        all_preds = 
        all_labels = 
        with torch.no_grad():

            # test loop
            for (text, labels), _ in test_iter:
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.config.device)
                text = text.type(torch.LongTensor)
                text = text.to(self.config.device)
                output = model(text, labels)
                loss, preds = output
                y_pred = torch.argmax(preds, dim=1)
                test_running_loss += loss.item()
        # return y_hats
        return loss, y_pred, labels

    def test(self, tagger: DialogueActTagger):
        #pdb.set_trace()
        facets = ["dimension", "comm_0", "comm_1", "comm_2", "comm_3"]
        facets = ["dimension"]
        classes = {"dimension": ["unknown", "task", "social obligation", "feedback"]}
        for facet in facets:
            test_set = stringify_tags(self.test_set, facet)
            loss, y_preds, y_true = self.tag_test(tagger, test_set)
            print(facet)
            print(classification_report(y_true, y_preds, target_names=classes[facet]))
            print(confusion_matrix(y_true, y_pred, labels=classes[facet]))
            print("test loss is {loss}")


        #y_true = [u.tags for u in self.test_set]
        #y_preds = [tagger.tag(u) for u in self.test_set]

   #     if "dimension" in tagger.classifiers:
   #         # 1) Compare dimension results
   #         y_dim_true = [[t.dimension.value for t in tags] for tags in y_true]
   #         y_dim_pred = [[t.dimension.value for t in tags] for tags in y_pred]
   #         binarizer = MultiLabelBinarizer()
   #         binarizer.fit(y_dim_true + y_dim_pred)

   #         # target_names = list(tagger.config.taxonomy.value.get_dimension_taxonomy().values().values())
   #         # labels = list(tagger.config.taxonomy.value.get_dimension_taxonomy().values().keys())
   #         # print("TARGET:", target_names)
   #         # print("LABELS:", labels)
   #         #

   #         print("Dimension Classification Report")
   #         print(
   #             classification_report(
   #                 binarizer.transform(y_dim_true), binarizer.transform(y_dim_pred)
   #             )
   #         )
   #         for dimension in tagger.config.taxonomy.value.get_dimension_taxonomy():
   #             if dimension.value > 0:
   #                 y_comm_true = []
   #                 y_comm_pred = []
   #                 for idx, datapoint in enumerate(y_true):
   #                     if any(t.dimension == dimension for t in datapoint):
   #                         y_comm_true.append(
   #                             [
   #                                 t.comm_function.value
   #                                 for t in datapoint
   #                                 if t.dimension == dimension
   #                             ][0]
   #                         )
   #                         try:
   #                             y_comm_pred.append(
   #                                 [
   #                                     t.comm_function.value
   #                                     for t in y_pred[idx]
   #                                     if t.dimension == dimension
   #                                 ][0]
   #                             )
   #                         except IndexError:
   #                             y_comm_pred.append(0)  # unknown
   #                 print(f"Communication Function Report for {dimension}")
   #                 print(classification_report(y_comm_true, y_comm_pred))
   #                 # labels=labels, target_names=target_names))

   # def test_compare(self, taggers: List[DialogueActTagger]):
   #     raise NotImplementedError()


