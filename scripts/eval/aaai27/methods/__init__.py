# Method implementations

# Basic methods
from scripts.eval.aaai27.methods.raw_qwen import RawQwenMethod
from scripts.eval.aaai27.methods.textmas_qwen import TextMASQwenMethod
from scripts.eval.aaai27.methods.latentmas import LatentMASMethod
from scripts.eval.aaai27.methods.latentweave import LatentWeaveMethod

# RWKV ablation methods
from scripts.eval.aaai27.methods.raw_rwkv import RawRWKVMethod
from scripts.eval.aaai27.methods.state_only_rwkv import StateOnlyRWKVMethod
from scripts.eval.aaai27.methods.anchored_relay_rwkv import AnchoredRelayRWKVMethod
from scripts.eval.aaai27.methods.pca_rwkv import PCARWKVMethod
from scripts.eval.aaai27.methods.anchored_pca_relay_state import AnchoredPCARelayStateMethod
from scripts.eval.aaai27.methods.direct_relay_rwkv import DirectRelayRWKVMethod
from scripts.eval.aaai27.methods.renormalized_relay_rwkv import RenormalizedRelayRWKVMethod
from scripts.eval.aaai27.methods.anchored_relay_state_rwkv import AnchoredRelayStateRWKVMethod

# HiddenBench-specific methods
from scripts.eval.aaai27.methods.best_local_agent import BestLocalAgentMethod
from scripts.eval.aaai27.methods.full_information_raw import FullInformationRawMethod
from scripts.eval.aaai27.methods.direct_latent_avg import DirectLatentAverageMethod
from scripts.eval.aaai27.methods.residual_fusion_k0 import ResidualFusionK0Method
from scripts.eval.aaai27.methods.shuffled_plan import ShuffledPlanMethod

# Long-form generation methods (PG-19)
from scripts.eval.aaai27.methods.raw_rwkv_long import RawRWKVLongMethod
from scripts.eval.aaai27.methods.single_plan_rwkv_long import SinglePlanRWKVLongMethod
from scripts.eval.aaai27.methods.latentweave_anchored_long import LatentWeaveAnchoredLongMethod
from scripts.eval.aaai27.methods.raw_qwen_long import RawQwenLongMethod
from scripts.eval.aaai27.methods.latentmas_qwen_long import LatentMASQwenLongMethod
