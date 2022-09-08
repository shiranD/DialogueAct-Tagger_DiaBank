import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpora.maptask import Maptask
from corpora.switchboard import Switchboard
from corpora.ami import AMI
from corpora.midas import MIDAS
from corpora.daily_dialog import DailyDialog
from taggers.transformer_tagger import BERT
from tester import DialogueActTester
from pathlib import Path


if __name__ == "__main__":
    model_path = "/projects/shdu9019/DA_tagger/DialogueAct-Tagger_DiaBank/models/transformer_example"
    tester = DialogueActTester(
        corpora=[
    #        (Maptask, str(Path("data/Maptask").resolve())),
    #        (AMI, str(Path("data/AMI/corpus").resolve())),
            (Switchboard, str(Path("data/Switchboard").resolve())),
    #        (DailyDialog, str(Path("data/DailyDialog").resolve())),
        ], path=model_path)
    #sys.exit()
    t = tester.test()
