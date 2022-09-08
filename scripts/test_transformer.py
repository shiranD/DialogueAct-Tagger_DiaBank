import sys
import os
from taggers.transformer_tagger import BERT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tester import DialogueActTester
from pathlib import Path


if __name__ == "__main__":
    model_path = ""
    tester = DialogueActTester(
        corpora_list=[
    #        (Maptask, str(Path("data/Maptask").resolve())),
    #        (AMI, str(Path("data/AMI/corpus").resolve())),
            (Switchboard, str(Path("data/Switchboard").resolve())),
    #        (DailyDialog, str(Path("data/DailyDialog").resolve())),
        ], model_path
    )
    #sys.exit()
    t = tester.test()
