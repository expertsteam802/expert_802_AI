"""English language assistant: normalization, definitions, grammar and dialogue routing."""
import re
import json
from pathlib import Path
from .model_client import is_configured, configured_model_name, generate_reply
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

    @staticmethod
    def _history_text(history):
        if not isinstance(history, list):
            return []
        return [str(item.get("text", "")) for item in history[-12:] if isinstance(item, dict)]

    def _in_game_context(self, history):
        previous = self._history_text(history)
        return any(re.search(r"\b(game|gta|video game|play)\b", item, re.I) for item in previous[-5:])

    def reply(self, message, history=None):
        original = (message or "").strip()
        if not original:
            return {"reply": "Please enter a word or message.", "engine": "dictionary-only"}

        # Preserve ordered processing when a user pastes several separate turns.
        batch_parts = [
            part.strip(" \t\r\n")
            for part in re.split(
                r"\s{2,}|(?<=[?!])\s+(?=(?:what|who|where|when|why|how)\b)",
                original,
                flags=re.I,
            )
            if part.strip(" \t\r\n")
        ]
        if len(batch_parts) > 1:
            running_history = list(history) if isinstance(history, list) else []
            rendered = []
            for part in batch_parts:
                result = self.reply(part, running_history)
                answer = result.get("reply", "")
                rendered.append("You: " + part + "\nExpert 802 AI: " + answer)
                running_history.extend([
                    {"role": "user", "text": part},
                    {"role": "assistant", "text": answer},
                ])
            return {
                "reply": "\n\n".join(rendered),
                "engine": "dictionary-only" if not is_configured() else "qwen3-self-hosted",
                "language": "English",
                "batch_processed": True,
                "batch_items": len(batch_parts),
            }

        if is_configured():
            try:
                answer = generate_reply(original, history)
                return {
                    "reply": answer,
                    "engine": "qwen3-self-hosted",
                    "model": configured_model_name(),
                    "language": "English",
                    "trained_model": True,
                    "knowledge_source": "local_dictionary_and_wordnet",
                }
            except Exception:
                return {
                    "reply": "The pretrained model endpoint is configured but did not respond. Please check the model host; no scripted fallback was used.",
                    "engine": "qwen3-self-hosted",
                    "model": configured_model_name(),
                    "model_error": True,
                }

        normalized, spelling_changes = correct_spelling(original)
        text = normalized.strip()
        words = tokens(text)
        target = self._definition_target(text)
        if target:
            answer = self._define_target_from_dictionary(target)
        elif len(words) == 1:
            answer = self._define_target_from_dictionary(words[0])
        else:
            answer = (
                "The pretrained conversational model is not connected yet. "
                "I can retrieve definitions from the local dictionary and WordNet, "
                "but I will not pretend to understand a full sentence using scripted replies."
            )
        correction_note = ("Recovered spelling: " + text + "\n\n") if spelling_changes else ""
        return {
            "reply": correction_note + answer,
            "engine": "dictionary-only",
            "language": "English",
            "model_configured": False,
            "spelling_recovered": bool(spelling_changes),
        }

    @staticmethod
    def _definition_target(text):
        value = re.sub(r"[?!.]+$", "", text.strip())
        patterns = [
            r"^what is (?:the definition of )?(.+?)(?:\s+in full)?$",
            r"^what does (.+?) mean$",
            r"^define (.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, value, re.I)
            if match:
                return match.group(1).strip(" '“”\"")
        return None

    def _define_target(self, target):
        return self._define_target_from_dictionary(target)

    def _define(self, text, words):
        target = self._definition_target(text)
        if target:
            return self._define_target_from_dictionary(target)
        quoted = re.findall(r"['\"“]([a-zA-Z'-]+)['\"”]", text)
        candidates = quoted + list(reversed(words))
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
        return "I could not find that word in my local dictionary yet. I can still help if you give me an example sentence."

    def _define_target_from_dictionary(self, target):
        key = re.sub(r"\s+", " ", target.lower().strip(" ?.!'“”\\""))
        definition = dictionary_lookup(key)
        if isinstance(definition, dict):
            definitions = definition.get("definitions", [])
            pos = definition.get("part_of_speech", "word")
            synonyms = definition.get("synonyms", [])[:5]
            if definitions:
                extra = (" Related words: " + ", ".join(synonyms) + ".") if synonyms else ""
                return f"“{key}” is a {pos}. {definitions[0]}{extra}"
        if isinstance(definition, str) and definition:
            return f"“{key}” means {definition}"
        return f"I do not have a reliable dictionary entry for “{key}” yet."

    @staticmethod
    def _explain_grammar(text):
        explanations = {
            "noun": "A noun names a person, place, thing, or idea. Examples: teacher, Kigali, book, freedom.",
            "verb": "A verb expresses an action, event, or state. Examples: run, think, become, is.",
            "adjective": "An adjective describes a noun. In “a bright room”, bright describes room.",
            "adverb": "An adverb modifies a verb, adjective, or another adverb. In “She speaks clearly”, clearly tells how she speaks.",
            "pronoun": "A pronoun can stand in for a noun or noun phrase. Examples include I, you, she, it, we, and they.",
            "tense": "Verb tense helps locate an action or state in time: “I work” (present), “I worked” (past), “I will work” (future construction).",
            "preposition": "A preposition shows a relationship, often of place, time, or direction. Examples: in, on, under, before, after.",
            "conjunction": "A conjunction connects words, phrases, or clauses. Examples: and, but, because, although.",
            "article": "The articles are “a”, “an”, and “the”. “A/an” are indefinite; “the” points to something specific.",
            "clause": "A clause is a group of words containing a subject and a verb. An independent clause can stand as a sentence; a dependent clause cannot.",
            "phrase": "A phrase is a group of related words that works as a unit but does not contain a complete subject–verb clause.",
            "subject": "The subject tells who or what the sentence is about. In “The child laughed”, “The child” is the subject.",
            "predicate": "The predicate says something about the subject and normally contains the verb.",
            "object": "An object receives the action of a verb or completes its meaning.",
            "word order": "The usual English statement order is Subject–Verb–Object (SVO): “Maya reads books.” Questions often change the order.",
            "sentence structure": "Common English patterns include S–V (“Birds fly”), S–V–O (“She reads books”), and S–V–complement (“He is tired”).",
            "active voice": "In the active voice, the subject performs the action: “The teacher explained the lesson.”",
            "passive voice": "In the passive voice, the subject receives the action: “The lesson was explained.” Form: be + past participle.",
            "conditional": "Conditionals describe possible or imagined situations. Example: “If it rains, I will stay.”",
            "modal verb": "Modal verbs express ability, possibility, permission, advice, or obligation: can, could, may, might, must, should, will, would.",
            "gerund": "A gerund is an -ing verb form used as a noun: “Swimming is fun.”",
            "infinitive": "An infinitive is usually “to” + the base verb: to learn, to speak, to build.",
            "punctuation": "Punctuation marks organize writing. Use periods for statements, question marks for direct questions, and commas to separate items or clauses.",
            "agreement": "Subject–verb agreement means the verb matches its subject: “She works” but “They work”; “I am” but “You are.”"
        }
        for key, value in explanations.items():
            if re.search(r"\b" + key + r"\b", text):
                return value
        return "English grammar explains how words form phrases and sentences through word classes, word order, agreement, tense, clauses, and punctuation."
