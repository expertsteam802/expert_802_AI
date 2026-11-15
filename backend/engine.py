"""English language assistant: normalization, definitions, grammar and dialogue routing."""
import re
import json
from pathlib import Path
from .model_client import is_configured, configured_model_name, generate_reply\nfrom .language import (
    tokens, load_dictionary, correct_common_grammar, correct_spelling,
    dictionary_lookup,
)

CURATED_DEFINITIONS = {
    "ai": ("abbreviation", "AI stands for artificial intelligence: computer systems designed to perform tasks that normally require human intelligence, such as understanding language, recognizing patterns and solving problems."),
    "food": ("noun", "Food is any nutritious substance that people or animals eat or drink to provide energy and support life."),
    "car": ("noun", "A car is a road vehicle, usually with four wheels, powered by an engine or electric motor and used to carry people."),
    "artificial intelligence": ("noun phrase", "Artificial intelligence (AI) is the field of creating computer systems that can perform tasks associated with human intelligence."),
    "gta": ("abbreviation / title", "GTA commonly refers to Grand Theft Auto, a video-game series. The exact game can be identified by its subtitle or number."),
    "fbi": ("abbreviation", "FBI stands for Federal Bureau of Investigation, the United States federal law-enforcement and domestic intelligence agency."),
    "kill": ("verb", "To kill means to cause the death of a living thing. The word can also have figurative meanings, such as stopping or defeating something."),
}

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
            return {"reply": "Please type a message so I can try to understand it.", "engine": "python-english-v2"}

        # If several utterances are pasted into one message, process them in order.
        # Double-space/newline separators are treated as deliberate boundaries;
        # question punctuation also separates consecutive questions.
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
                "engine": "python-english-v2",
                "language": "English",
                "batch_processed": True,
                "batch_items": len(batch_parts),
            }

        # Prefer a real pretrained model when a self-hosted inference server is configured.
        # The dictionary is retrieved inside model_client and supplied as evidence, not as canned replies.
        if is_configured():
            try:
                answer = generate_reply(original, history)
                return {
                    "reply": answer,
                    "engine": "qwen3-self-hosted",
                    "model": configured_model_name(),
                    "language": "English",
                    "trained_model": True,
                    "knowledge_source": "local_dictionary",
                }
            except Exception:
                return {
                    "reply": "The pretrained language model is configured but its inference server did not respond. Please check the model host; I will not silently substitute scripted answers.",
                    "engine": "qwen3-self-hosted",
                    "model": configured_model_name(),
                    "model_error": True,
                }

        normalized, spelling_changes = correct_spelling(original)
        text = normalized.strip()
        lower = text.lower()
        words = tokens(text)
        correction_note = ("I think you meant: " + text + "\n\n") if spelling_changes else ""
        history_text = self._history_text(history)
        previous_user = history_text[-1].strip().lower() if history_text else ""
        game_context = self._in_game_context(history)

        # Interpret replies using the assistant's immediately preceding question.
        last_assistant = next(
            (str(item.get("text", "")).strip().lower()
             for item in reversed(history or [])
             if isinstance(item, dict) and item.get("role") == "assistant"),
            ""
        )
        game_choice_context = game_context and any(
            phrase in last_assistant
            for phrase in ("open a game you already have", "download a game", "find a game to play")
        )
        if lower in {"yes", "yeah", "yep", "sure", "ok", "okay"} and game_context:
            answer = "Sure. Which do you mean: open a game you already have, download a game, or find a game to play?"
        elif game_choice_context and re.search(r"\b(open|launch|start)\b.*\b(game|gta)\b", lower):
            answer = "Got it—you want to open a game you already have. I cannot launch an app on your device from this chat, but I can guide you. Which device are you using: Windows PC, Android phone, PlayStation, or Xbox?"
        elif game_choice_context and re.search(r"\b(download|install|get)\b.*\b(game|gta)\b", lower):
            answer = "Got it—you want to download or install a game. Which device are you using: Windows PC, Android phone, PlayStation, or Xbox?"
        elif game_choice_context and re.search(r"\b(find|recommend|choose)\b.*\bgame\b", lower):
            answer = "I can help you find a game. What kind of games do you enjoy—action, racing, sports, adventure, or something else?"
        elif lower in {"gta", "grand theft auto"} and game_context:
            answer = "GTA usually means Grand Theft Auto. Which one do you mean—GTA V, GTA Online, or another GTA game?"
        elif re.search(r"\bwhat word comes next|predict the next word|next word after\b", lower):
            phrase = text.split("after", 1)[-1].strip(" :") if "after" in lower else ""
            answer = self._predict_next_word(phrase)
        elif re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening)[!. ]*", lower):
            answer = "Hello! I'm Expert 802 AI. We can talk in English. What would you like to discuss?"
        elif re.search(r"\b(who are you|what are you|your name)\b", lower):
            answer = "I'm Expert 802 AI, a project-owned English language assistant. I help with word meanings, spelling, grammar and conversation."
        elif re.search(r"^what is .+? in full(?: words)?[?!.]*$", lower) or re.search(r"^full form of .+[?!.]*$", lower):
            target = re.sub(r"^what is\s+|^full form of\s+", "", lower)
            target = re.sub(r"\s+in full(?: words)?[?!.]*$", "", target).strip(" ?.!'\"" )
            if target in CURATED_DEFINITIONS and CURATED_DEFINITIONS[target][0].startswith("abbreviation"):
                expansions = {"ai": "artificial intelligence", "fbi": "Federal Bureau of Investigation", "gta": "Grand Theft Auto"}
                answer = target.upper() + " stands for " + expansions[target] + "."
            else:
                answer = self._define_target(target)
        elif re.search(r"\b(what does|meaning of|define|definition of)\b", lower):
            answer = self._define(text, words)
        elif re.search(r"\bwhat is\b", lower):
            # Definitions are local; named world facts must not be mistaken for dictionary entries.
            target = self._definition_target(text)
            if target and re.match(r"(?:the )?(?:capital|president|population|weather|price|score|latest)\b", target, re.I):
                answer = "I recognise this as a factual question, but I do not have a connected current-world knowledge source in this version. I should not invent an answer."
            elif target:
                answer = self._define_target(target)
            else:
                answer = "I can explain a word or concept if you tell me which one you mean."
        elif re.search(r"\b(check|correct|grammar|grammatically|is this sentence)\b", lower):
            corrected, notes = correct_common_grammar(text)
            if notes:
                answer = "A possible correction is:\n“" + corrected + "”\n\n" + " ".join(notes)
            elif spelling_changes:
                answer = "The spelling and word spacing look corrected above. I did not find another error covered by my current grammar rules."
            else:
                answer = "I did not detect an error covered by my current grammar rules. That does not prove the sentence is fully correct; my grammar coverage is still growing."
        elif re.search(r"\b(noun|verb|adjective|adverb|pronoun|tense|preposition|conjunction|article|clause|phrase|subject|predicate|object|word order|sentence structure|active voice|passive voice|conditional|modal verb|gerund|infinitive|punctuation|agreement)\b", lower) and re.search(r"\b(explain|grammar|what is|meaning|how does|how do)\b", lower):
            answer = self._explain_grammar(lower)
        elif re.search(r"\b(i need my game right now|i need a game right now)\b", lower):
            answer = "Got it—you need your game right now. Are you trying to open a game you already have, download one, or find a game to play?"
        elif re.search(r"\b(affect and effect|difference between affect and effect)\b", lower):
            answer = "“Affect” is usually a verb meaning to influence: “Rain can affect traffic.” “Effect” is usually a noun meaning a result: “The effect was a delay.”"
        elif lower.strip(" .!?") in CURATED_DEFINITIONS:
            answer = self._define_target(lower.strip(" .!?"))
        elif re.search(r"\b(thank you|thanks)\b", lower):
            answer = "You're welcome. What else would you like to talk about?"
        elif re.search(r"\b(you are wrong|that's wrong|you don't understand|not what i asked)\b", lower):
            answer = "Thanks for correcting me. I may have interpreted your words incorrectly. Tell me what you intended and I will use that correction in this conversation."
        elif re.search(r"\b(where is|when did|how many|capital of|president of|latest|today)\b", lower):
            answer = "I recognise this as a factual question, but I do not have a connected current-world knowledge source in this version. I should not invent an answer."
        elif len(words) == 1:
            answer = self._define_target(words[0]) + "\n\nWould you like an example sentence, pronunciation help, or a grammar explanation?"
        else:
            corrected, notes = correct_common_grammar(text)
            if notes:
                answer = "A possible correction is:\n“" + corrected + "”\n\n" + " ".join(notes) + "\n\nIs that what you meant?"
            elif spelling_changes:
                answer = "I recovered the likely spelling and word spacing. What would you like to know or do with this?"
            else:
                answer = "I can help explain words, grammar and sentence structure. Could you tell me what you want to do with this sentence?"
        return {
            "reply": correction_note + answer,
            "engine": "python-english-v2",
            "language": "English",
            "trained_model": bool(self.model),
            "training_vocabulary": self.model.get("vocabulary_size", 0) if self.model else 0,
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
        key = re.sub(r"\s+", " ", target.lower().strip(" ?.!'“”\""))
        if key in CURATED_DEFINITIONS:
            pos, definition = CURATED_DEFINITIONS[key]
            return f"“{key}” ({pos}): {definition}"
        return self._define(target, tokens(target))

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
        key = re.sub(r"\s+", " ", target.lower().strip(" ?.!'“”\""))
        if key in CURATED_DEFINITIONS:
            pos, definition = CURATED_DEFINITIONS[key]
            return f"“{key}” ({pos}): {definition}"
        # Prefer the actual requested phrase, not the final word in the question.
        if " " in key:
            return f"I do not yet have a reliable entry for the full phrase “{key}”. I can define its individual words if you want."
        definition = dictionary_lookup(key)
        if isinstance(definition, dict):
            definitions = definition.get("definitions", [])
            pos = definition.get("part_of_speech", "word")
            synonyms = definition.get("synonyms", [])[:4]
            if definitions:
                extra = (" Related words: " + ", ".join(synonyms) + ".") if synonyms else ""
                return f"“{key}” is a {pos}. {definitions[0]}{extra}"
        if definition:
            return f"“{key}” means {definition}"
        return f"I do not have a reliable definition for “{key}” in my local dictionary yet."

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
