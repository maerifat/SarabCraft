"""Text attack implementations."""

from .deepwordbug import run_deepwordbug
from .textbugger import run_textbugger
from .hotflip import run_hotflip
from .pruthi2019 import run_pruthi2019
from .textfooler import run_textfooler
from .bert_attack import run_bert_attack
from .bae import run_bae
from .pwws import run_pwws
from .alzantot_ga import run_alzantot_ga
from .faster_alzantot_ga import run_faster_alzantot_ga
from .iga import run_iga
from .pso import run_pso
from .clare import run_clare
from .back_translation import run_back_translation
from .a2t import run_a2t
from .checklist import run_checklist
from .stresstest import run_stresstest
from .uat import run_uat
from .scpn import run_scpn
from .input_reduction import run_input_reduction
from .kuleshov2017 import run_kuleshov2017
from .seq2sick import run_seq2sick
from .morpheus import run_morpheus

__all__ = [
    "run_deepwordbug",
    "run_textbugger",
    "run_hotflip",
    "run_pruthi2019",
    "run_textfooler",
    "run_bert_attack",
    "run_bae",
    "run_pwws",
    "run_alzantot_ga",
    "run_faster_alzantot_ga",
    "run_iga",
    "run_pso",
    "run_clare",
    "run_back_translation",
    "run_a2t",
    "run_checklist",
    "run_stresstest",
    "run_uat",
    "run_scpn",
    "run_input_reduction",
    "run_kuleshov2017",
    "run_seq2sick",
    "run_morpheus",
]
