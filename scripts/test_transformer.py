import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpora.taxonomy import Taxonomy
from corpora.maptask import Maptask
from corpora.switchboard import Switchboard
from corpora.ami import AMI
from corpora.midas import MIDAS
from corpora.daily_dialog import DailyDialog
from taggers.transformer_tagger import TransformerTagger
from tester import DialogueActTester
from pathlib import Path
import pdb

if __name__ == "__main__":
    # load transformer tagger model
    model_path = "/projects/shdu9019/DA_tagger/DialogueAct-Tagger_DiaBank/models/transformer_example"
    tagger = TransformerTagger(model_path)
    tester = DialogueActTester(
        corpora=[
    #        Maptask(str(Path("data/Maptask").resolve()), Taxonomy.ISO),
    #        AMI(str(Path("data/AMI/corpus").resolve()), Taxonomy.ISO),
            Switchboard(str(Path("data/Switchboard").resolve()), Taxonomy.ISO),
    #        DailyDialog(str(Path("data/DailyDialog").resolve()), Taxonomy.ISO),
        ], cfg_path=model_path)
    #pdb.set_trace()
    t = tester.test(tagger)
