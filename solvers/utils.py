from pydantic import BaseModel


class ResponseFormat(BaseModel):
    answer: str
    reasoning: str

# Base Prompt
PROMPT = """
Please solve the rebus puzzle represented by the image. Respond with ONLY a valid JSON object containing two keys:
1. 'answer': the string value of your solution
2. 'reasoning': a detailed explanation of how you arrived at this answer, including the meaning of each visual element and how they combine
"""

# Prompt used for In-Context Learning (ICL)
ICL_PROMPT = """
You will be given one example:
<Image>
Response (json):
{
    "answer": "For once in my life",
    "reasoning": "The text 'M1Y L1I1F1E' places the digit '1' between each letter of 'MY LIFE', resulting in four '1's embedded in the phrase. This visually represents 'four 1s in my life', which sounds like 'for once in my life.'",
}
"""
ICL_PATH = "1.png"

# Prompt used for caption only
CAPTION_PROMPT = """
Please solve the rebus puzzle represented by the text given: {answer_text}. Respond with ONLY a valid JSON object containing two keys:
1. 'answer': the string value of your solution
2. 'reasoning': a detailed explanation of how you arrived at this answer, including the meaning of each visual element and how they combine
"""

SKILLS_ID_TO_NAME = {
    0: 'Absence or Negation: You need to identify missing elements or crossed-out text (e.g., a gap = "invisible").',
    1: "Font Style/Size: You need to recognize different font styles, sizes, or colors.",
    2: "Image Recognition: You need to identify objects, people, actions, or symbols in the image.",
    3: "Letter and Word Manipulation: You need to identify overlapping, hidden, or repeated letters that form new meanings.",
    4: "Phonetics and Wordplay: You need to recognize homophones, puns, and similar-sounding words that create new meanings.",
    5: 'Quantitative or Mathematical Reasoning: You need to interpret math symbols and count objects (e.g., "1 2 3" + foot = "three feet").',
    6: "Spatial and Positional Reasoning: You need to understand how object placement and relationships (e.g., above/below, inside/outside) affect meaning.",
    7: 'Symbolic Substitution: You need to identify how replacing numbers, letters, or emojis (e.g., "4" → "for") might form meaning.',
    8: "Text Orientation: You need to understand text direction (e.g., upside down, rotated) and how it affects meaning.",
    9: "Text Recognition (OCR + Typography/Layout): You need to detect written words, fonts, capitalization, or stylized text.",
    10: 'Visual Metaphors and Cultural References: You need to identify idioms, memes, or visual sayings ("water" shaped like a waterfall).',
}

def skill_informed_prompt(skills: list[int], prompt: str) -> str:
    """
    Given a list of skills and a prompt, return a skill-informed prompt.
    """
    skill_names = [SKILLS_ID_TO_NAME[skill] for skill in skills]
    skill_names_str = "\n".join(skill_names)
    return f"Please solve the rebus puzzle represented by the image.\n\nThe skills you need to apply are:\n{skill_names_str}\n{prompt}"
