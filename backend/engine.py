"""Transparent English conversation engine with dictionary and spelling recovery."""
import re
import json
from pathlib import Path
from .language import (
    tokens, load_dictionary, correct_common_grammar, correct_spelling,
    dictionary_lookup,
)

class ExpertLanguageEngine:
    def __init__(self):
        self.dictionary = load_dictionary()
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        path = Path(__file__).resolve().parent / "language_model.json"
        try:
            with path.open(encoding="utf-8") as stream:
                model = json.load(stream)
            return model if model.get("format") == "expert802-word-bigram-v1" else None
        except (OSError, ValueError):
            return None

    def _predict_next_word(self, phrase):
        if not self.model:
            return "The hosted training workflow has not produced a model artifact yet."
        words = tokens(phrase)
        if not words:
            return "Give me a short English phrase after the words 'next word after'."
        options = self.model.get("bigram", {}).get(words[-1], {})
        if not options:
            return f"I have not learned a next-word pattern after '{words[-1]}' yet."
        word = max(options.items(), key=lambda item: item[1])[0]
        return f"From my training examples, a likely next word after '{words[-1]}' is '{word}'. This is a statistical guess, not a guarantee of meaning."

    def reply(self, message, history=None):
        original = (message or "").strip()
        if not original:
            return {"reply": "Please type a message so I can try to understand it.", "engine": "python-english-v1"}
        normalized, spelling_changes = correct_spelling(original)
        text = normalized
        lower = text.lower()
        words = tokens(text)
        correction_note = ""
        if spelling_changes:
            correction_note = "I think you meant: **" + text + "**\n\n"

        if re.search(r"what word comes next|predict the next word|next word after", lower):
            phrase = text.split("after", 1)[-1].strip(" :") if "after" in lower else ""
            answer = self._predict_next_word(phrase)
        elif re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening)[!. ]*", lower):
            answer = "Hello! I'm Expert 802 AI. We can talk in English. What would you like to discuss?"
        elif re.search(r"\b(who are you|what are you|your name)\b", lower):
            answer = "I'm Expert 802 AI, a project-owned English language program. I use dictionaries, spelling recovery, grammar rules, and language patterns stored in this project. I am still developing and do not have human-level understanding."
        elif re.search(r"\b(what does|meaning of|define|definition of)\b", lower):
            answer = self._define(text, words)
        elif re.search(r"\b(check|correct|grammar|grammatically|is this sentence)\b", lower):
            corrected, notes = correct_common_grammar(text)
            if notes:
                answer = "A possible correction is:\n“" + corrected + "”\n\n" + " ".join(notes)
            elif spelling_changes:
                answer = "The spelling and word spacing look corrected above. I did not find another error covered by my current grammar rules."
            else:
                answer = "I did not detect an error covered by my current grammar rules. That does not prove the sentence is fully correct; my grammar coverage is still growing."
        elif re.search(r"\b(noun|verb|adjective|adverb|pronoun|tense|preposition|conjunction|article|clause|phrase|subject|predicate|object|word order|sentence structure|active voice|passive voice|conditional|modal verb|gerund|infinitive|punctuation|agreement)\b", lower) and re.search(r"\b(what is|explain|define|meaning|how does|how do)\b", lower):
            answer = self._explain_grammar(lower)
        elif re.search(r"\b(i need my game right now|i need a game right now)\b", lower):
            answer = "Got it—you need your game right now. Are you trying to open a game you already have, download one, or find a game to play?"
        elif re.search(r"\b(affect and effect|difference between affect and effect)\b", lower):
            answer = "“Affect” is usually a verb meaning to influence: “Rain can affect traffic.” “Effect” is usually a noun meaning a result: “The effect was a delay.”"
        elif re.search(r"\b(thank you|thanks)\b", lower):
            answer = "You're welcome. What else would you like to talk about?"
        elif re.search(r"\b(you are wrong|that's wrong|you don't understand|not what i asked)\b", lower):
            answer = "Thanks for correcting me. I may have interpreted your words incorrectly. Please explain what you intended. I can use that correction in this conversation, but this version does not permanently retrain itself from feedback."
        elif re.search(r"\b(what is|who is|where is|when did|how many|capital of|president of|latest|today)\b", lower):
            answer = "I recognise this as a factual question, but my current training does not provide a reliable world-knowledge system. I should not guess. I can explain English wording and definitions from my local knowledge."
        elif len(words) <= 2:
            answer = "Could you add a little more detail? A complete sentence or a little context will help me interpret your meaning."
        else:
            corrected, notes = correct_common_grammar(text)
            if notes:
                answer = "I can identify a possible grammar issue. A suggested version is:\n“" + corrected + "”\n\n" + " ".join(notes) + "\n\nIs that what you meant?"
            elif spelling_changes:
                answer = "I recovered the likely spelling and word boundaries, but I am not yet confident about the request. Could you tell me what you want to do?"
            else:
                answer = "I can process the English words and some sentence patterns in your message, but I cannot confidently infer the full meaning yet. Could you give me more context or ask a specific question? I would rather ask than invent an answer."
        return {
            "reply": correction_note + answer,
            "engine": "python-english-v1",
            "language": "English",
            "trained_model": bool(self.model),
            "training_vocabulary": self.model.get("vocabulary_size", 0) if self.model else 0,
            "spelling_recovered": bool(spelling_changes),
        }

    def _define(self, text, words):
        quoted = re.findall(r"['\"“]([a-zA-Z'-]+)['\"”]", text)
        requested = re.search(r"what does ([a-zA-Z'-]+) mean|meaning of ([a-zA-Z'-]+)|define ([a-zA-Z'-]+)", text, re.I)
        extracted = next((part for part in requested.groups() if part), None) if requested else None
        candidates = ([extracted] if extracted else []) + quoted + list(reversed(words))
        for word in candidates:
            definition = dictionary_lookup(word.lower())
            if definition:
                if isinstance(definition, dict):
                    definitions = definition.get("definitions", [])
                    pos = definition.get("part_of_speech", "word")
                    synonyms = definition.get("synonyms", [])[:5]
                    if definitions:
                        extra = (" Related words: " + ", ".join(synonyms) + ".") if synonyms else ""
                        return f"“{word.lower()}” is a {pos}. {definitions[0]}{extra}"
                return f"“{word.lower()}” means {definition}"
        return "I could not find that word in my local dictionary yet. The dictionary is expandable, and I will say when I do not have a reliable definition."

    @staticmethod
    def _explain_grammar(text):
        explanations = {
            "noun": "A noun names a person, place, thing, or idea. Examples: teacher, Kigali, book, freedom.",
            "verb": "A verb expresses an action, event, or state. Examples: run, think, become, is.",
            "adjective": "An adjective describes a noun. In “a bright room”, bright describes room.",
            "adverb": "An adverb modifies a verb, adjective, or another adverb. In “She speaks clearly”, clearly tells how she speaks.",
            "pronoun": "A pronoun can stand in for a noun or noun phrase. Examples include I, you, she, it, we, and they.",
            "tense": "Verb tense helps locate an action or state in time: “I work” (present), “I worked” (past), “I will work” (future construction).",
            "preposition": "A preposition shows a relationship, often of place, time, or direction. Examples: in, on, under, before, after. “The book is on the table.”",
            "conjunction": "A conjunction connects words, phrases, or clauses. Examples: and, but, because, although. “I stayed because it rained.”",
            "article": "The articles are “a”, “an”, and “the”. “A/an” are indefinite; “the” points to something specific.",
            "clause": "A clause is a group of words containing a subject and a verb. An independent clause can stand as a sentence; a dependent clause cannot.",
            "phrase": "A phrase is a group of related words that works as a unit but does not contain a complete subject–verb clause.",
            "subject": "The subject tells who or what the sentence is about. In “The child laughed”, “The child” is the subject.",
            "predicate": "The predicate says something about the subject and normally contains the verb. In “The child laughed”, “laughed” is the predicate.",
            "object": "An object receives the action of a verb or completes its meaning. In “She kicked the ball”, “the ball” is the direct object.",\n            "word order": "The usual English statement order is Subject–Verb–Object (SVO): “Maya (subject) reads (verb) books (object).” Questions often change the order: “Does Maya read books?” Adjectives usually come before the nouns they describe.",\n            "sentence structure": "A basic English sentence needs a subject and a verb. Common patterns include S–V (“Birds fly”), S–V–O (“She reads books”), and S–V–complement (“He is tired”). Clauses can be joined with conjunctions.",\n            "active voice": "In the active voice, the subject performs the action: “The teacher explained the lesson.” This is usually the clearest direct form.",\n            "passive voice": "In the passive voice, the subject receives the action: “The lesson was explained by the teacher.” Form: be + past participle; use it when the receiver or result is more important.",\n            "conditional": "Conditionals describe possible or imagined situations. Zero: “If you heat ice, it melts.” First: “If it rains, I will stay.” Second: “If I had time, I would travel.” Third: “If I had studied, I would have passed.”",\n            "modal verb": "Modal verbs express ability, possibility, permission, advice, or obligation: can, could, may, might, must, should, will, would. They are followed by the base verb: “She can swim.”",\n            "gerund": "A gerund is an -ing verb form used as a noun: “Swimming is fun.” It can be the subject or object of a sentence.",\n            "infinitive": "An infinitive is usually “to” + the base verb: to learn, to speak, to build. Example: “I want to learn English.” Some verbs take a bare infinitive, such as “can go.”",\n            "punctuation": "Punctuation marks organize writing. Use a period for a statement, a question mark for a direct question, commas to separate items or clauses, and apostrophes for contractions and possession.",\n            "agreement": "Subject–verb agreement means the verb matches its subject: “She works” but “They work”; “I am” but “You are.” In the present simple, he/she/it usually takes -s on a main verb."
        }
        for key, value in explanations.items():
            if re.search(r"\b" + key + r"\b", text):
                return value
        return "English grammar explains how words form phrases and sentences through word classes, word order, agreement, tense, clauses, and punctuation."
