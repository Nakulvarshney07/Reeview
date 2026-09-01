import os
import json
from typing import Dict, List, Any, Union

class Preprocessor:
    """
    Combined Product Document Preprocessor.
    
    Accepts raw product metadata, specifications, and star-grouped customer reviews,
    formatting them into a structured text document while preserving review star signals.
    """
    
    @staticmethod
    def combine_product_text(data: Dict[str, Any]) -> str:
        """
        Combines product name, specs, features, and star-wise customer reviews into a single document string.
        """
        sections = []
        
        # 1. Title / Name
        name = data.get("name") or data.get("title") or "Product"
        sections.append(f"Product Title:\n{name}")
        
        # 2. Specifications
        specs = data.get("specs", {})
        if isinstance(specs, dict) and specs:
            spec_lines = [f"- {k}: {v}" for k, v in specs.items()]
            sections.append(f"Specifications:\n" + "\n".join(spec_lines))
        elif isinstance(specs, str) and specs.strip():
            sections.append(f"Specifications:\n{specs.strip()}")
            
        # 3. Features / Description
        desc = data.get("description") or data.get("features") or ""
        if isinstance(desc, list):
            desc = "\n".join([f"- {item}" for item in desc])
        if desc and str(desc).strip():
            sections.append(f"Description / Features:\n{str(desc).strip()}")
            
        # 4. Star-grouped Customer Reviews
        reviews_dict = data.get("reviews", {})
        if isinstance(reviews_dict, dict) and reviews_dict:
            rev_sections = []
            for star_key in ["5_star", "4_star", "3_star", "2_star", "1_star"]:
                if star_key in reviews_dict and reviews_dict[star_key]:
                    star_num = star_key.split("_")[0]
                    lines = [f"{star_num}-star reviews:"]
                    for rev in reviews_dict[star_key]:
                        title = rev.get("title", "").strip()
                        body = rev.get("body", "").strip()
                        if title and body:
                            lines.append(f"- \"{title}\": {body}")
                        elif body:
                            lines.append(f"- {body}")
                        elif title:
                            lines.append(f"- {title}")
                    if len(lines) > 1:
                        rev_sections.append("\n".join(lines))
            if rev_sections:
                sections.append("\n\n".join(rev_sections))
                
        return "\n\n".join(sections)

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleanup without removing domain semantic signals."""
        if not text:
            return ""
        # Preserve newlines and punctuation, clean extraneous whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
