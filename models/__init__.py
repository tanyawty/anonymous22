from .stgnn import STGNN_LearnOnly
from .fouriergnn import FourierGNN_LearnOnly
try:
    from .patchtst import PatchTST_Baseline
except Exception:
    pass
from .classical_baselines import GRU_Baseline, LSTM_Baseline, TCN_Baseline, Transformer_Baseline, MLP_Baseline
from .gp_mech_stgnn import GPMechSTGNN
