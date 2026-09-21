"""Local intent layer (full-local, zero cloud).

One local forward pass per utterance via laya-mlx (Apple Silicon MLX):
a speculative fan-out of every question the executor could need.
Laya never generates text; every free-text value (text to type, search
query, domain) is produced as *candidates* in code and Laya selects
the right one ("select instead of generate").

Large option sets (> LAYA_MAX_OPTIONS, e.g. 103 installed apps or 39
shortcuts) are pre-shortlisted in code with fuzzy matching before the
forward pass: Laya's per-option token budget collapses past ~20 options
(Banking77: 0.425 vs Jev 0.870). The shortlist always keeps the fallback
key (`none` / `other`) so "no match" stays expressible.
"""
from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import actions, config

ACTIONS: dict[str, str] = {
    "open_app": "Launch, open, switch to, or bring up an application program on the Mac (for example Chrome, Cursor, Slack, Finder, Terminal, Notes)",
    "open_website": "Go to a website or web page by name or domain, with no search query (for example 'go to youtube', 'open reddit', 'pull up gmail')",
    "web_search": "Search for something on the web or on a specific site: Google it, look it up, find videos of, search YouTube for, search Amazon for",
    "type_text": "Type, write, dictate, or enter some text into whatever is currently focused",
    "new_item": "Create something new inside an app: a new note, document, file, tab, window, message, email, or page (for example 'new note', 'open a new note in the notes app', 'make a new note called groceries', 'new document')",
    "shortcut": "Press a single key or keyboard shortcut: enter, escape, tab, copy, paste, undo, save, select all, new tab, close tab, reload, go back, quit the app, switch app, and similar",
    "scroll": "Scroll the current page or document up or down, to the top or bottom",
    "volume": "Change the system sound volume: louder, quieter, mute, unmute, max",
    "media": "Control music or video playback: play, pause, resume, next track, previous track, skip",
    "screenshot": "Take a screenshot of the screen",
    "open_folder": "Open a folder like Downloads, Desktop, Documents, or the home folder in Finder",
    "system": "System-level action: lock the screen, put the display to sleep, show the desktop, toggle dark mode, empty the trash",
    "stop": "Tell the assistant to stop listening, go to sleep, or exit",
    "none": "Not a command for the computer: conversation, thinking aloud, background chatter, or unintelligible",
}

SHORTCUT_CRITERIA: dict[str, str] = {
    "enter": "press enter / return / submit",
    "escape": "press escape / cancel / dismiss",
    "tab": "press the tab key",
    "space": "press the space bar",
    "backspace": "delete the previous character / backspace",
    "arrow_up": "press the up arrow",
    "arrow_down": "press the down arrow",
    "arrow_left": "press the left arrow",
    "arrow_right": "press the right arrow",
    "copy": "copy the selection",
    "paste": "paste from the clipboard",
    "cut": "cut the selection",
    "undo": "undo the last change",
    "redo": "redo",
    "select_all": "select all / select everything",
    "save": "save the file / document",
    "find": "open find / search within the page or document",
    "new": "create a new file, document, note, or message in the current app",
    "new_tab": "open a new browser tab",
    "close_tab_or_window": "close the current tab or window",
    "reopen_closed_tab": "reopen the last closed tab",
    "quit_app": "quit / exit the current application entirely",
    "minimize_window": "minimize the window",
    "hide_app": "hide the current app",
    "fullscreen": "toggle full screen",
    "next_tab": "switch to the next tab",
    "previous_tab": "switch to the previous tab",
    "browser_back": "go back to the previous page",
    "browser_forward": "go forward",
    "reload": "reload / refresh the page",
    "address_bar": "focus the browser address bar / URL bar",
    "spotlight": "open Spotlight search",
    "switch_app": "switch to the previous / next application (command-tab)",
    "next_window": "switch to the next window of the current app",
    "delete_word": "delete the previous word",
    "delete_line": "delete the current line / everything before the cursor on this line",
    "zoom_in": "zoom in / make text bigger",
    "zoom_out": "zoom out / make text smaller",
    "bold": "make the selection bold",
    "italic": "make the selection italic",
    "send_message": "send the message (command-enter)",
    "emoji_picker": "open the emoji picker",
}

# regexes that peel the payload text off a spoken command
_TEXT_PATTERNS = [
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:type|write|enter|dictate|input|put|insert|say|send|text|paste)(?:\s+in|\s+out|\s+the\s+words?|\s+the\s+text|\s+this|\s+that)?[:,]?\s+(?P<t>.+)$",
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:search|google|look\s*up|find|look\s+for|show\s+me|pull\s+up)(?:\s+(?:on|in)\s+\w+(?:\s+\w+)?)?(?:\s+for)?[:,]?\s+(?P<t>.+)$",
    r"^.*?\b(?:for|about|of|on)\s+(?P<t>.+)$",
    r"[\"“'](?P<t>[^\"”']+)[\"”']",
]
_TITLE = re.compile(r"\b(?:called|titled|named|labeled|that says|saying|with the title)\s+(?P<t>.+)$", re.I)
_TRAILING_IN_APP = re.compile(r"\s+(?:in|into|inside|on)\s+(?:the\s+)?(?:[A-Z][\w.]*|notes|chrome|cursor|safari|slack|mail|messages|terminal|finder)(?:\s+app)?\s*[.!?]?$")
_TRAILING_SUBMIT = re.compile(
    r"[\s,.]*(?:and|then)?\s*(?:hit|press|and)\s+(?:enter|return|send|submit)\s*[.!]?$", re.I
)
_SPLIT_COMPOUND = re.compile(r"\s*(?:,\s*)?\b(?:and then|then|and also|and)\b\s*", re.I)


def _clean(s: str) -> str:
    s = s.strip().strip('"“”\'')
    s = _TRAILING_SUBMIT.sub("", s).strip()
    return s.rstrip(" .").strip()


def text_candidates(utterance: str) -> dict[str, str]:
    """Candidate spans that might be the payload text. Laya picks; code never guesses."""
    cands: list[str] = []

    def add(t: str) -> None:
        t = _clean(t)
        if t and t not in cands:
            cands.append(t)

    m = _TITLE.search(utterance)
    if m:
        add(m.group("t"))
    for pat in _TEXT_PATTERNS:
        m = re.search(pat, utterance, flags=re.I)
        if m:
            add(m.group("t"))
            add(_TRAILING_IN_APP.sub("", m.group("t")))
    whole = _clean(utterance)
    if whole and whole not in cands:
        cands.append(whole)
    if not cands:
        cands.append(utterance.strip() or "(nothing)")
    return {f"c{i}": c for i, c in enumerate(cands[:6])}


_DOMAIN = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.I)
_SITE_WORD = re.compile(
    r"\b(?:go to|open|visit|pull up|bring up|load|navigate to|take me to)\s+(?:the\s+)?(?:website\s+|site\s+)?([a-z0-9][a-z0-9 .-]*?)(?:\s+(?:website|site|dot com|\.com|page|homepage))?\s*[.!?]?$",
    re.I,
)


def domain_guess(utterance: str) -> str | None:
    m = _DOMAIN.search(utterance)
    if m:
        return m.group(1).lower()
    m = _SITE_WORD.search(utterance)
    if m:
        word = m.group(1).strip().lower().replace(" dot ", ".").replace(" ", "")
        if word and word not in ("it", "that", "this"):
            return word if "." in word else f"{word}.com"
    return None


def _shortlist(
    utterance: str,
    criteria: dict[str, Any],
    max_n: int,
    keep: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fuzzy pre-filter for large choice sets (see module docstring)."""
    if len(criteria) <= max_n:
        return criteria
    fallback = {k: criteria[k] for k in keep if k in criteria}
    rest = [(k, v) for k, v in criteria.items() if k not in fallback]
    hay = utterance.lower()
    ranked = sorted(
        rest,
        key=lambda kv: difflib.SequenceMatcher(
            None, hay, f"{kv[0]} {kv[1] or ''}".lower()
        ).ratio(),
        reverse=True,
    )
    return dict(list(fallback.items()) + ranked[: max(0, max_n - len(fallback))])


@dataclass
class Plan:
    utterance: str
    action: str
    confidence: float
    args: dict[str, Any] = field(default_factory=dict)
    answers: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0

    def __str__(self) -> str:
        a = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.action}({a})  conf={self.confidence:.2f}  {self.latency_ms}ms"


class Brain:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.LAYA_MODEL
        self._agent: Any = None

    def _agent_lazy(self) -> Any:
        if self._agent is None:
            import laya_mlx as laya

            self._agent = laya.load(self.model, batch_size=config.LAYA_BATCH_SIZE)
        return self._agent

    # ------------------------------------------------------------ questions

    def _questions(
        self, utterance: str, cands: dict[str, str], apps: list[str]
    ) -> dict[str, Any]:
        max_n = config.LAYA_MAX_OPTIONS
        app_criteria = _shortlist(
            utterance, {**{a: None for a in apps}, "none": "No listed application matches what the user said"},
            max_n, keep=("none",),
        )
        site_criteria = _shortlist(
            utterance, {**{s: None for s in actions.SITES}, "other": "A site not in this list"},
            max_n, keep=("other",),
        )
        q: dict[str, Any] = {
            "action": {
                "type": "choice",
                "instructions": "The user is speaking a voice command to their Mac. `utterance` is the transcript. Which single kind of action are they asking the computer to perform right now?",
                "criteria": ACTIONS,
            },
            "addressed": {
                "type": "noul",
                "instructions": "Is `utterance` an instruction spoken to a voice assistant that controls this computer (open, type, search, scroll, press, play, close, and so on), rather than conversation with another person, a phone call, reading aloud, or thinking out loud?",
                "criteria": {"true": "A direct instruction for the computer to do something now", "false": "Not directed at the computer, or not an instruction"},
            },
            "compound": {
                "type": "noul",
                "instructions": "Does `utterance` ask for two or more separate actions to be performed one after another (for example 'open chrome and go to youtube')? A single action with several words is not compound.",
                "criteria": {"true": "Two or more distinct actions are requested", "false": "Exactly one action is requested"},
            },
            "app": {
                "type": "choice",
                "instructions": "Assume the user wants to open or switch to an application. Which installed application in `apps` do they mean? Match on meaning: 'chrome' means Google Chrome, 'settings' means System Settings, 'browser' means the default browser. Choose `none` if no listed app matches.",
                "criteria": app_criteria,
            },
            "site": {
                "type": "choice",
                "instructions": "Assume the user wants to open a website. Which site do they mean? Choose `other` if it is not one of the listed sites.",
                "criteria": site_criteria,
            },
            "engine": {
                "type": "choice",
                "instructions": "Assume the user wants to search for something. Which site or search engine should the search run on? If the user does not name a site, choose google.",
                "criteria": {e: None for e in actions.SEARCH_ENGINES},
            },
            "text": {
                "type": "choice",
                "instructions": "Assume the user wants some text typed or searched. `candidates` holds possible payloads cut from the utterance. Which candidate is exactly the payload text the user intends, with no command words (like 'type', 'search for', 'on youtube') and no trailing 'and press enter' included?",
                "criteria": {k: v for k, v in cands.items()},
            },
            "in_app": {
                "type": "noul",
                "instructions": "Does the user name a specific application that the action should happen inside of (for example 'in the notes app', 'in chrome', 'in cursor')? Naming an app as the thing to open does not count unless the action is something done inside it.",
                "criteria": {"true": "An application is named as the place where the action happens", "false": "No application is named, or the app is only the thing being opened"},
            },
            "new_kind": {
                "type": "choice",
                "instructions": "Assume the user wants to create something new. What kind of thing?",
                "criteria": {"tab": "a new browser tab", "window": "a new window", "item": "a new note, document, file, message, email, page, or anything else created with the app's New command"},
            },
            "has_title": {
                "type": "noul",
                "instructions": "Assume the user is creating a new note, document, or file. Do they give it a title or initial text (for example 'called groceries', 'titled ideas', 'that says hello')?",
                "criteria": {"true": "A title or initial text is given", "false": "No title or text is given"},
            },
            "submit": {
                "type": "noul",
                "instructions": "After typing the text, does the user also want the enter/return key pressed (they say things like 'and hit enter', 'and send it', 'and search')?",
                "criteria": {"true": "The user explicitly asks to submit, send, or press enter afterwards", "false": "They only want the text typed"},
            },
            "shortcut": {
                "type": "choice",
                "instructions": "Assume the user wants a key or keyboard shortcut pressed. Which one?",
                "criteria": _shortlist(utterance, SHORTCUT_CRITERIA, max_n),
            },
            "scroll_dir": {
                "type": "choice",
                "instructions": "Assume the user wants to scroll. In which direction?",
                "criteria": {"down": "scroll down / further", "up": "scroll up / back up", "top": "jump to the very top", "bottom": "jump to the very bottom"},
            },
            "scroll_amount": {
                "type": "choice",
                "instructions": "Assume the user wants to scroll up or down. How far?",
                "criteria": {"little": "a little / a bit / a few lines", "page": "a normal amount, about one screen; the default when unspecified", "a_lot": "a lot / way down / far"},
            },
            "volume_op": {
                "type": "choice",
                "instructions": "Assume the user wants to change the volume. What change?",
                "criteria": {"up": "louder / turn it up", "down": "quieter / turn it down", "mute": "mute / silence", "unmute": "unmute / sound back on", "max": "maximum / all the way up", "half": "medium / half volume"},
            },
            "media_op": {
                "type": "choice",
                "instructions": "Assume the user wants to control playback. What?",
                "criteria": {"play_pause": "play, pause, resume, or stop the current track or video", "next": "next track / skip this song", "previous": "previous track / go back a song"},
            },
            "folder": {
                "type": "choice",
                "instructions": "Assume the user wants to open a folder in Finder. Which one?",
                "criteria": {f: None for f in actions.FOLDERS},
            },
            "system_op": {
                "type": "choice",
                "instructions": "Assume the user wants a system-level action. Which one?",
                "criteria": {"lock": "lock the screen", "sleep_display": "put the display / screen to sleep", "show_desktop": "show the desktop", "toggle_dark_mode": "switch between dark and light mode", "empty_trash": "empty the trash"},
            },
        }
        return q

    # ------------------------------------------------------------ inference

    def evaluate(self, utterance: str, front: str | None = None) -> Plan:
        apps = actions.installed_apps()
        cands = text_candidates(utterance)
        state = {
            "utterance": utterance,
            "frontmost_app": front if front is not None else actions.frontmost_app(),
            "apps": apps,
            "candidates": cands,
        }
        questions = self._questions(utterance, cands, apps)
        t0 = time.perf_counter()
        data = self._agent_lazy().predict(state, questions)
        ms = int((time.perf_counter() - t0) * 1000)
        return self._to_plan(utterance, data["answers"], cands, ms)

    def _to_plan(self, utterance: str, ans: dict[str, Any], cands: dict[str, str], ms: int) -> Plan:
        act = ans["action"]
        action = act["choice"]
        conf = float(act["confidence"])
        args: dict[str, Any] = {}

        def ch(key: str) -> tuple[str, float]:
            a = ans[key]
            return a["choice"], float(a["confidence"])

        explicit = _explicit_route(utterance, ans, cands)
        if explicit is not None:
            action, args, conf = explicit
        elif action == "open_app":
            app, c = ch("app")
            code_app, code_conf = resolve_app(utterance, actions.installed_apps())
            if _SITE_INTENT.search(utterance) and code_conf < 0.85 and domain_guess(utterance):
                # "go to youtube" misrouted to open_app: reroute to the site Laya picked.
                action, args = "open_website", {}
                site, c = ch("site")
                if site != "other":
                    args["url"] = actions.SITES[site]
                    args["site"] = site
                    conf = min(conf, c)
                else:
                    dom = domain_guess(utterance)
                    if dom:
                        args["url"] = f"https://{dom}"
                        args["site"] = dom
                    else:  # nothing to navigate to: fall back to a search
                        action = "web_search"
                        engine, c1 = ch("engine")
                        tkey, c2 = ch("text")
                        args["engine"] = engine
                        args["query"] = cands.get(tkey, utterance)
                        conf = min(conf, c2)
            elif code_conf >= 0.85 and (app == "none" or c < 0.5):
                args["app"] = code_app
                conf = min(conf, code_conf)
            else:
                args["app"] = app
                conf = min(conf, c)
        elif action == "open_website":
            site, c = ch("site")
            if site != "other":
                args["url"] = actions.SITES[site]
                args["site"] = site
                conf = min(conf, c)
            else:
                dom = domain_guess(utterance)
                if dom:
                    args["url"] = f"https://{dom}"
                    args["site"] = dom
                else:  # nothing to navigate to: fall back to a search
                    action = "web_search"
        elif action == "web_search":
            engine, c1 = ch("engine")
            tkey, c2 = ch("text")
            args["engine"] = engine
            args["query"] = cands.get(tkey, utterance)
            conf = min(conf, c2)
        elif action == "type_text":
            tkey, c = ch("text")
            args["text"] = cands.get(tkey, utterance)
            args["submit"] = float(ans["submit"]["noul"]) > config.YES or bool(
                _TRAILING_SUBMIT.search(utterance)
            )
            conf = min(conf, c)
        elif action == "new_item":
            args["kind"], _ = ch("new_kind")
            if float(ans["has_title"]["noul"]) > config.YES:
                tkey, _ = ch("text")
                args["title"] = cands.get(tkey, "")
        elif action == "shortcut":
            s, c = ch("shortcut")
            args["shortcut"] = s
            conf = min(conf, c)
        elif action == "scroll":
            d, c = ch("scroll_dir")
            a, _ = ch("scroll_amount")
            args["direction"], args["amount"] = d, a
            conf = min(conf, c)
        elif action == "volume":
            op, c = ch("volume_op")
            args["op"] = op
            conf = min(conf, c)
        elif action == "media":
            op, c = ch("media_op")
            args["op"] = op
            conf = min(conf, c)
        elif action == "open_folder":
            f, c = ch("folder")
            args["folder"] = f
            conf = min(conf, c)
        elif action == "system":
            op, c = ch("system_op")
            args["op"] = op
            conf = min(conf, c)

        # Code-first type rescue first: a leading "type ..." claims the
        # utterance (so "type X and hit enter" doesn't become bare Enter).
        # Also boosts Laya's own type_text when it picked code's first
        # candidate but scored it ~0.03.
        if conf < 0.3 and action in ("open_app", "none", "web_search", "type_text"):
            m = _TYPE_VERB.search(utterance)
            rest = utterance[m.end():].strip(" ,.") if m else ""
            if rest:
                first = next(iter(text_candidates(utterance).values()))
                if action == "type_text":
                    try:
                        picked = cands.get(ans["text"]["choice"])
                    except (KeyError, TypeError):
                        picked = None
                    if picked == first:
                        conf = 0.8
                else:
                    args = {
                        "text": first,
                        "submit": float(ans["submit"]["noul"]) > config.YES
                        or bool(_TRAILING_SUBMIT.search(utterance)),
                    }
                    action, conf = "type_text", 0.8
        # Code-first shortcut rescue (type_text excluded: "type copy"
        # means the word, not the key).
        if conf < 0.3 and action in ("open_app", "none", "shortcut", "scroll"):
            ks = keyword_shortcut(utterance)
            if ks:
                action, args = "shortcut", {"shortcut": ks}
                conf = 0.8
        # "in the notes app": focus that app before acting inside it
        if action in ("new_item", "shortcut", "type_text", "scroll") and float(ans["in_app"]["noul"]) > config.YES:
            app, _ = ch("app")
            if app != "none":
                args["in_app"] = app
        args["compound"] = float(ans["compound"]["noul"]) > config.YES
        args["addressed"] = round(float(ans["addressed"]["noul"]), 2)
        return Plan(utterance, action, conf, args, ans, ms)


_SUBMIT_ONLY = re.compile(r"^(?:then\s+)?(?:hit|press|and)?\s*(?:enter|return|send|submit)(?:\s+it)?$", re.I)

APP_ALIASES = {
    "system settings": "System Settings",
    "activity monitor": "Activity Monitor",
    "visual studio code": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "google chrome": "Google Chrome",
    "chrome": "Google Chrome",
    "safari": "Safari",
    "firefox": "Firefox",
    "cursor": "Cursor",
    "terminal": "Terminal",
    "finder": "Finder",
    "notes": "Notes",
    "settings": "System Settings",
    "mail": "Mail",
    "calendar": "Calendar",
    "music": "Music",
    "messages": "Messages",
    "photos": "Photos",
    "calculator": "Calculator",
    "preview": "Preview",
    "chatgpt": "ChatGPT",
    "chat gpt": "ChatGPT",
    "gpt": "ChatGPT",
    "claude": "Claude",
    "mdapp": "MDapp",
    "md app": "MDapp",
    "message": "Messages",
    "whatsapp": "WhatsApp",
    "whats app": "WhatsApp",
}

_SITE_INTENT = re.compile(r"\b(go to|open|visit|pull up|bring up|load|navigate to|take me to)\b", re.I)

_SHORTCUT_RX = [
    ("close_tab_or_window", r"\bclose (this |that |the )?(tab|window)\b"),
    ("reopen_closed_tab", r"\breopen\b"),
    ("new_tab", r"\bnew tab\b"),
    ("next_tab", r"\bnext tab\b"),
    ("previous_tab", r"\bprevious tab\b"),
    ("copy", r"\bcopy\b"),
    ("paste", r"\bpaste\b"),
    ("cut", r"\bcut\b"),
    ("undo", r"\bundo\b"),
    ("redo", r"\bredo\b"),
    ("select_all", r"\bselect all\b"),
    ("save", r"\bsave\b"),
    ("find", r"\bfind\b"),
    ("reload", r"\breload\b|\brefresh (the page)?\b"),
    ("browser_back", r"\bgo back\b"),
    ("browser_forward", r"\bgo forward\b"),
    ("quit_app", r"\bquit( the app)?\b"),
    ("switch_app", r"\bswitch app"),
    ("fullscreen", r"\bfull ?screen\b"),
    ("enter", r"\bpress enter\b|\bhit enter\b"),
    ("escape", r"\bescape\b|\bcancel\b|\bdismiss\b"),
    ("address_bar", r"\baddress bar\b"),
    ("spotlight", r"\bspotlight\b"),
]


_TYPE_VERB = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:type|write|dictate|input|put|insert)\b", re.I
)


def keyword_shortcut(utterance: str) -> str | None:
    """Code-first shortcut rescue: explicit words ("close this tab") map
    deterministically instead of relying on zero-shot confidences ~0.05."""
    low = utterance.lower()
    for key, rx in _SHORTCUT_RX:
        if re.search(rx, low):
            return key
    return None


_LEAD_VERBS = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:open|launch|switch to|go to|close|quit)\b\s*", re.I
)


def _app_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def resolve_app(utterance: str, apps: list[str]) -> tuple[str, float]:
    """Code-first app resolution (deterministic).

    Laya zero-shot cannot pick among 100+ installed apps (confidences
    ~0.05 even when the argmax is right), so code resolves the obvious
    cases: aliases, then substring/fuzzy match. Returns (app, score)
    where score gates whether the caller trusts code over Laya.
    """
    low = utterance.lower()
    app_set = {a.lower(): a for a in apps}
    for alias in sorted(APP_ALIASES, key=len, reverse=True):
        target = APP_ALIASES[alias]
        if target.lower() not in app_set:
            continue
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            return target, 0.95
    low = _LEAD_VERBS.sub("", utterance.lower()).strip()
    app_set = {a.lower(): a for a in apps}
    for alias in sorted(APP_ALIASES, key=len, reverse=True):
        target = APP_ALIASES[alias]
        if target.lower() not in app_set:
            continue
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            return target, 0.95
    best, best_score = "none", 0.0
    for a in apps:
        tokens = _app_tokens(a)
        score = difflib.SequenceMatcher(None, low, a.lower()).ratio()
        for w in re.findall(r"[a-z0-9]+", low):
            if len(w) >= 4 and w in tokens:
                score = max(score, 0.85)
        if score > best_score:
            best, best_score = a, score
    return best, best_score


_FOLDER_WORDS = {
    "downloads": "downloads",
    "desktop": "desktop",
    "documents": "documents",
    "pictures": "pictures",
    "movies": "movies",
    "trash": "trash",
    "applications": "applications",
    "home": "home",
}

_SEARCH_VERB = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:search|google|look\s*up|find|look\s+for)\b", re.I
)

_ENGINE_WORDS = [
    ("youtube", "youtube"),
    ("amazon", "amazon"),
    ("maps", "google_maps"),
    ("wikipedia", "wikipedia"),
    ("wiki", "wikipedia"),
    ("github", "github"),
    ("reddit", "reddit"),
    ("spotify", "spotify"),
    ("perplexity", "perplexity"),
    ("twitter", "twitter_x"),
]


def _explicit_route(
    utterance: str, ans: dict[str, Any], cands: dict[str, str]
) -> tuple[str, dict[str, Any], float] | None:
    """Deterministic routing for unambiguous phrases.

    Returns (action, args, conf) or None. Code owns these patterns outright;
    Laya handles everything else.
    """
    low = utterance.lower()
    if re.search(r"\btake a screenshots?\b", low):
        return "screenshot", {}, 0.9
    m = re.search(
        r"\b(open|show|go to)\b.{0,12}\b(downloads|desktop|documents|pictures|movies|trash|applications|home)( folder)?\b",
        low,
    )
    if m and m.group(2) in _FOLDER_WORDS:
        return "open_folder", {"folder": _FOLDER_WORDS[m.group(2)]}, 0.9
    if re.search(r"\block( the screen)?\b", low):
        return "system", {"op": "lock"}, 0.9
    if re.search(r"\b(sleep|put).{0,12}\bdisplay\b|\bdisplay.{0,8}\bsleep\b", low):
        return "system", {"op": "sleep_display"}, 0.9
    if re.search(r"\bshow (the )?desktop\b", low):
        return "system", {"op": "show_desktop"}, 0.9
    if re.search(r"\bdark mode\b", low):
        return "system", {"op": "toggle_dark_mode"}, 0.9
    if re.search(r"\bempty (the )?trash\b", low):
        return "system", {"op": "empty_trash"}, 0.9
    m = re.search(r"\bvolume\b.{0,8}\b(up|louder|down|quieter|mute|unmute|max|half)\b", low)
    if not m:
        m = re.search(r"\b(turn it up|louder|turn it down|quieter|^mute$|^unmute$|max volume|half volume)\b", low)
    if m:
        word = m.group(1) if m.lastindex else m.group(0)
        op = {"up": "up", "louder": "up", "turn it up": "up", "down": "down", "quieter": "down",
              "turn it down": "down", "mute": "mute", "unmute": "unmute",
              "max volume": "max", "half volume": "half"}.get(word, "up")
        return "volume", {"op": op}, 0.9
    m = re.search(r"\b(pause|play|resume|stop)( the (music|song|video))?\b", low)
    if m and _TYPE_VERB.search(utterance) is None:
        return "media", {"op": "play_pause"}, 0.85
    m = re.search(r"\bnext (track|song)\b|\bskip\b", low)
    if m:
        return "media", {"op": "next"}, 0.85
    m = re.search(r"\bprevious (track|song)\b|\bgo back a song\b", low)
    if m:
        return "media", {"op": "previous"}, 0.85
    m = re.search(r"\bscroll\b.{0,8}\b(up|down)\b", low)
    if m:
        amount = "a_lot" if re.search(r"\ba ?lot\b|\bway\b|\bfar\b", low) else (
            "little" if re.search(r"\blittle\b|\bbit\b", low) else "page")
        return "scroll", {"direction": m.group(1), "amount": amount}, 0.85
    if re.search(r"\bgo to the (top|bottom)\b", low):
        return "scroll", {"direction": "top" if "top" in low else "bottom", "amount": "page"}, 0.85
    if _SEARCH_VERB.search(utterance):
        engine = "google"
        for word, eng in _ENGINE_WORDS:
            if re.search(r"\b" + re.escape(word) + r"\b", low):
                engine = eng
                break
        else:
            try:
                engine = ans["engine"]["choice"]
            except (KeyError, TypeError):
                pass
        try:
            tkey = ans["text"]["choice"]
        except (KeyError, TypeError):
            tkey = next(iter(cands))
        query = cands.get(tkey, utterance)
        words = engine.replace("_", " ").split()
        pat = r"^" + re.escape(words[0]) + (r"(?:\s+" + re.escape(words[1]) + r")?" if len(words) > 1 else "") + r"\s*(?:for\s+)?"
        query = re.sub(pat, "", query, flags=re.I).strip() or query
        return "web_search", {"engine": engine, "query": query}, 0.8
    return None


def split_compound(utterance: str) -> list[str]:
    """Split 'A and then B' into parts. A trailing 'hit enter' is not its own step:
    the type_text plan already carries submit=True."""
    parts = [p.strip(" ,.") for p in _SPLIT_COMPOUND.split(utterance)]
    parts = [p for p in parts if len(p) > 1]
    if len(parts) > 1 and _SUBMIT_ONLY.match(parts[-1]):
        parts = parts[:-1]
        if len(parts) == 1:
            return [utterance]
        parts[-1] = parts[-1] + " and hit enter"
    return parts
