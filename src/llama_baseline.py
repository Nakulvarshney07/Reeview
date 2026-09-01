import json
from typing import Dict, List, Any

class LlamaBaseline:
    """
    Baseline Llama 3.2 Aspect Generator.
    Represents zero-shot / baseline LLM prompting without reinforcement reward alignment.
    """
    def __init__(self):
        pass

    def get_baseline_aspects(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves baseline Llama 3.2 predictions from product dataset or constructs a baseline set.
        """
        llama_list = product_data.get("llama_output", [])

        # Format into standard structure
        extracted = []
        for item in llama_list:
            if isinstance(item, dict):
                extracted.append({
                    "aspect": item.get("aspect", ""),
                    "subaspects": item.get("subaspects", [])
                })

        return {
            "model_type": "Llama 3.2 (Baseline Zero-Shot LLM)",
            "extracted_aspects": extracted
        }
