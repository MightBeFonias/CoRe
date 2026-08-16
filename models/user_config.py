from enum import Enum

from models.hf_causal_model import HFCausalLMModel


class Models(Enum):
    """Registry mapping a config's `model:` key to an implementation class.

    Values are (name, class) tuples rather than the bare class so that two
    entries pointing at the same class stay distinct members: with identical
    values, Python would silently alias the second name to the first and
    iteration would skip it.
    """

    baseline_qwen_3_5_9b = ("baseline_qwen_3_5_9b", HFCausalLMModel)

    # Gemma baseline (paper Table 2, macro-F1 0.522 val)
    gemma_baseline = ("gemma_baseline", HFCausalLMModel)

    # Single-policy ablations (paper Table 1 and 2)
    qwen_all_cot = ("qwen_all_cot", HFCausalLMModel)
    qwen_all_direct = ("qwen_all_direct", HFCausalLMModel)

    # CoRe routing and its two refinements (paper Table 2)
    core_base = ("core_base", HFCausalLMModel)
    core = ("core", HFCausalLMModel)
    core_company_cot = ("core_company_cot", HFCausalLMModel)

    def __init__(self, key: str, model_cls: type):
        self.model_cls = model_cls

    @staticmethod
    def get_model(model_name: str):
        try:
            return Models[model_name].model_cls
        except KeyError:
            available = ", ".join(m.name for m in Models)
            raise ValueError(
                f"Model `{model_name}` not found. Available: {available}"
            )
