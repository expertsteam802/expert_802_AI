"""Transparent English conversation engine; deliberately not marketed as human-level AI."""
import re
from .language import tokens, load_dictionary, correct_common_grammar, dictionary_lookup

class ExpertLanguageEngine:
    def __init__(self):
        self.dictionary = load_dictionary()

    def reply(self, message, history=None):
        text = message.strip()
        lower = text.lower()
        words = tokens(text)
        if not text:
            return {"reply": "Please type a message so I can try to understand it.", "engine": "python-english-v1"}
        if re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening)[!. ]*", lower):
            answer = "Hello! I'm Expert 802 AI. We can talk in English. What would you like to discuss?"
        elif re.search(r"\b(who are you|what are you|your name)\b", lower):
            answer = "I'm Expert 802 AI. I am a project-owned English language program. I process words, grammar patterns, and conversation context using code and data in this repository. I am still an early system, not a human or a large trained language model."
        elif re.search(r"\b(what does|meaning of|define|definition of)\b", lower):
            answer = self._define(text, words)
        elif re.search(r"\b(check|correct|grammar|grammatically|is this sentence)\b", lower):
            corrected, notes = correct_common_grammar(text)
            if notes:
                answer = "A possible correction is:\n“" + corrected + "”\n\n" + " ".join(notes)
            else:
                answer = "I did not detect an error covered by my current grammar rules. That does not prove the sentence is fully correct; my grammar coverage is still limited."
        elif re.search(r"\b(noun|verb|adjective|adverb|pronoun|tense)\b", lower) and re.search(r"\b(what is|explain|define)\b", lower):
            answer = self._explain_grammar(lower)
        elif re.search(r"\b(affect and effect|difference between affect and effect)\b", lower):
            answer = "“Affect” is usually a verb meaning to influence: “Rain can affect traffic.” “Effect” is usually a noun meaning a result: “The effect was a delay.”"
        elif re.search(r"\b(thank you|thanks)\b", lower):
            answer = "You're welcome. What else would you like to talk about?"
        elif re.search(r"\b(you are wrong|that's wrong|you don't understand|not what i asked)\b", lower):
            answer = "Thanks for correcting me. I may have interpreted your words incorrectly. Please explain what you intended. This version can use the correction in the current request, but it does not permanently update its trained weights from feedback."
        elif re.search(r"\b(what is|who is|where is|when did|how many|capital of|president of|latest|today)\b", lower):
            answer = "I recognise this as a factual question, but my current English training data is small and does not contain a reliable world-knowledge system. I should not guess. You can teach me facts by adding trusted English training text and retraining the language model."
        elif len(words) <= 2:
            answer = "Could you add a little more detail? A complete sentence or a little context will help me interpret your meaning."
        else:
            corrected, notes = correct_common_grammar(text)
            if notes:
                answer = "I can identify a possible grammar issue. A suggested version is:\n“" + corrected + "”\n\n" + " ".join(notes) + "\n\nIs that what you meant?"
            else:
                answer = "I can process the English words and some sentence patterns in your message, but I cannot confidently infer the full meaning yet. Could you give me more context or ask a specific question? I would rather ask than invent an answer."
        return {"reply": answer, "engine": "python-english-v1", "language": "English"}

    def _define(self, text, words):
        quoted = re.findall(r"['\"“]([a-zA-Z'-]+)['\"”]", text)
        candidates = quoted + list(reversed(words))
        for word in candidates:
            definition = self.dictionary.get(word.lower())
            if definition:
                return f"“{word.lower()}” means {definition}"
        return "I could not find that word in my starter English dictionary. Add a dictionary entry or training text containing the word and its meaning. A dictionary is a growing resource, not a complete representation of language understanding."

    @staticmethod
    def _explain_grammar(text):
        explanations = {
            "noun": "A noun names a person, place, thing, or idea. Examples: teacher, Kigali, book, freedom.",
            "verb": "A verb expresses an action, event, or state. Examples: run, think, become, is.",
            "adjective": "An adjective describes a noun. In “a bright room”, bright describes room.",
            "adverb": "An adverb modifies a verb, adjective, or another adverb. In “She speaks clearly”, clearly tells how she speaks.",
            "pronoun": "A pronoun can stand in for a noun or noun phrase. Examples include I, you, she, it, we, and they.",
            "tense": "Verb tense helps locate an action or state in time. Examples: I work (present), I worked (past), I will work (future construction).",
        }
        for key, value in explanations.items():
            if key in text:
                return value
        return "English grammar describes how words combine into meaningful sentences: word classes, word order, agreement, tense, and punctuation."
