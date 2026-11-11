"""Werewolf agent for make.fun werewolf-agent-01 (agent/v1, Python 3.13, stdlib only).

A deterministic, rule-based player: no model calls, no wall clock, no unseeded
randomness. Every choice is a pure function of the game's view (the env's log,
the seat's private record) and MAKEFUN_TRIAL_SEED, so a trial replays
byte-for-byte.

Game: 8 players (2 werewolves, 1 seer, 1 doctor, 4 villagers). 30 games a
trial: 15 as a werewolf, 5 as seer, 5 as doctor, 5 as villager. The other
seven seats are claude-haiku-5-5 house players (effort low).

Approach
--------
Village side (seer / doctor / villager):
  * Track everything public: deaths and revealed roles, every vote, every
    statement. Score each living seat's suspicion from hard evidence first
    (the seer's own inspections are certain), then from role claims (a seer
    claim whose accusation later proves right clears its claimant; one whose
    accusation proves wrong condemns it), then from vote records (voting a
    revealed wolf out clears; pushing a revealed villager out condemns),
    defences of revealed wolves, and accusations made by players who later
    died as the seer or as a villager the wolves wanted silenced.
  * Seer: inspect the most suspicious unchecked seat each night; the moment
    an inspection finds a werewolf, claim the seer role and push that wolf's
    exile. Counter-claim anyone else who claims seer (there is only one).
    Spend a clear only to save a cleared player the village is about to exile.
  * Doctor: protect the credibly claimed seer (the wolves' top target), else
    the most useful living villager; never protect yourself or repeat last
    night's seat. Stay hidden about the role.
  * Villager: argue from the same evidence and vote the top suspect.

Werewolf side:
  * Night: agree with the partner. As the second wolf to vote, match the
    partner's pick (that guarantees a kill). As the first, name the
    highest-value target: a credibly claimed seer first, then the player
    most dangerous to the wolves (vocal, accurate, accusing us), then any
    villager. Never name the partner.
  * Day: blend in as a villager. On day 1-2, if nobody has credibly claimed
    seer yet, claim the seer role and accuse a villager (a fake check) to
    steer an exile onto the village; never counter-claim a seer claim that
    is already out. Otherwise argue like a villager, support exiling a
    plausible villager, and never vote for the partner.
  * Vote: vote the villager the table is most likely to exile (the one the
    village already suspects, or the one I accused as "seer"); deflect from
    the partner and from myself onto another villager.

Env protocol (POST JSON, one object in, one out):
  POST /observe {}  -> {games, finished, done, pending: [{game, decision,
                     options | max_chars, view}]}
  POST /game {game} -> that game's view
  POST /act {game, decision, target | text | pass} -> 200 once taken;
                     400/409 refuses nothing, so send another.
Decisions: speak (text or pass), vote / kill / protect / inspect (a target
among options, or null to abstain / act on nobody).
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Deterministic randomness (xoshiro256**, same stream shape as the kit SDK).
# Seeded only from MAKEFUN_TRIAL_SEED + a label, so both runs agree.
# --------------------------------------------------------------------------

_RNG_DOMAIN = "agent-fun/agent/rng/v1"
_MASK64 = (1 << 64) - 1


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Rng:
    """xoshiro256** seeded from sha256(domain, seed, label)."""

    def __init__(self, seed, label=""):
        h = hashlib.sha256(
            _RNG_DOMAIN.encode() + b"\0" + str(seed).encode() + b"\0" + label.encode()
        ).digest()
        s = [int.from_bytes(h[i:i + 8], "big") for i in range(0, 32, 8)]
        if not any(s):
            s[0] = 1
        self._s = s

    def next_u64(self):
        s0, s1, s2, s3 = self._s
        result = (_rotl((s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s1 << 17) & _MASK64
        a2 = s2 ^ s0
        a3 = s3 ^ s1
        self._s = [s0 ^ a3, s1 ^ a2, a2 ^ t, _rotl(a3, 45)]
        return result

    def randint(self, lo, hi):
        span = hi - lo + 1
        limit = (1 << 64) - (1 << 64) % span
        while True:
            x = self.next_u64()
            if x < limit:
                return lo + x % span

    def choice(self, items):
        return items[self.randint(0, len(items) - 1)]


# --------------------------------------------------------------------------
# Env client
# --------------------------------------------------------------------------

class EnvError(Exception):
    pass


def _post(url, body, timeout=600):
    """One POST of a JSON object; returns (status, parsed JSON or raw text)."""
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as res:
            raw = res.read()
            return res.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            raw = err.read().decode("utf-8", errors="replace")
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, {"error": raw}
        finally:
            err.close()
    except (urllib.error.URLError, OSError) as err:
        raise EnvError("no answer from %s: %s" % (url, err))


class Env:
    """The trial env at MAKEFUN_ENV_URL."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def observe(self):
        status, body = _post(self.base + "/observe", {})
        if status != 200 or not isinstance(body, dict):
            raise EnvError("POST /observe answered %s" % status)
        return body

    def act(self, game, decision, **kw):
        body = {"game": game, "decision": decision}
        body.update(kw)
        status, resp = _post(self.base + "/act", body)
        return status, resp


# --------------------------------------------------------------------------
# View parsing
# --------------------------------------------------------------------------

def fold_statement(text):
    """Fold a statement the env's way: runs of breaks/tabs -> one space, other
    control chars dropped, ends trimmed."""
    text = re.sub(r"[\t\n\v\f\r\u0085\u2028\u2029]+", " ", text)
    text = re.sub(r"\p{Cc}", "", text) if False else "".join(
        ch for ch in text if ch >= " " and ch != "\x7f"
    )
    return text.strip()


def cut_statement(text, max_chars=500):
    folded = fold_statement(text)
    if len(folded) <= max_chars:
        return folded
    return folded[:max_chars].strip()


class Analysis:
    """Everything deducible from one game's view, recomputed per decision."""

    def __init__(self, view):
        self.view = view
        self.day = view.get("day", 1)
        self.phase = view.get("phase", "")
        you = view.get("you") or {}
        self.my_seat = you.get("seat")
        self.my_name = you.get("name", "")
        self.my_role = you.get("role", "")
        self.alive = bool(you.get("alive"))
        partner = you.get("partner")
        self.partner_seat = partner.get("seat") if partner else None
        self.players = {p["seat"]: p for p in view.get("players", [])}
        self.log = view.get("log", []) or []
        self.seer_results = view.get("seerResults") or []
        self.last_protect = view.get("lastProtect")
        self.wolf_votes = view.get("wolfVotes") or []
        self.outcome = view.get("outcome")

        # Seer knowledge: seat -> True (werewolf) / False (not werewolf).
        self.checked = {}
        for r in self.seer_results:
            if r.get("seat") is not None and r.get("werewolf") is not None:
                self.checked[r["seat"]] = bool(r["werewolf"])

        self._parse_log()

    # -- log parsing ------------------------------------------------------
    def _parse_log(self):
        self.deaths = []        # (day, seat, role) revealed by death/exile
        self.votes = {}         # day -> [(voter, target or None)]
        self.speeches = []      # (day, seat, round, text)
        self.passes = []        # (day, seat, round)
        self.exiles = []        # (day, seat, role, tally)
        self.night_deaths = []  # (day, seat, role) from dawn
        for e in self.log:
            kind = e.get("kind")
            day = e.get("day", self.day)
            if kind == "dawn":
                died = e.get("died")
                if died is not None:
                    self.deaths.append((day, died, e.get("role")))
                    self.night_deaths.append((day, died, e.get("role")))
            elif kind == "exile":
                seat, role = e.get("seat"), e.get("role")
                if seat is not None:
                    self.deaths.append((day, seat, role))
                    self.exiles.append((day, seat, role, e.get("tally") or []))
            elif kind == "vote":
                self.votes.setdefault(day, []).append((e.get("seat"), e.get("target")))
            elif kind == "speech":
                self.speeches.append((day, e.get("seat"), e.get("round", 0), e.get("text", "")))
            elif kind == "pass":
                self.passes.append((day, e.get("seat"), e.get("round", 0)))

        self.revealed = {}      # seat -> role (revealed by death/exile)
        for _, seat, role in self.deaths:
            if role:
                self.revealed[seat] = role
        self.dead_wolves = [s for s, r in self.revealed.items() if r == "werewolf"]
        self.dead_village = [s for s, r in self.revealed.items() if r in ("seer", "doctor", "villager")]

    # -- basic state ------------------------------------------------------
    def living(self):
        return [s for s, p in sorted(self.players.items()) if p.get("alive")]

    def living_others(self):
        return [s for s in self.living() if s != self.my_seat]

    def name(self, seat):
        p = self.players.get(seat)
        return p.get("name", "seat %d" % seat) if p else ("seat %d" % seat)

    def is_wolf(self, seat):
        """Certainly a werewolf: my own check, or revealed dead."""
        if self.checked.get(seat) is True:
            return True
        return self.revealed.get(seat) == "werewolf"

    def is_clear(self, seat):
        """Certainly not a werewolf: my own check, or revealed village."""
        if self.checked.get(seat) is False:
            return True
        return self.revealed.get(seat) in ("seer", "doctor", "villager")

    def partner_alive(self):
        return self.partner_seat is not None and self.players.get(self.partner_seat, {}).get("alive")

    def wolves_alive(self):
        """Living seats that are certainly wolves (my checks); plus, when I am a
        wolf, my partner."""
        out = [s for s in self.living() if self.checked.get(s) is True]
        if self.my_role == "werewolf" and self.partner_alive() and self.partner_seat not in out:
            out.append(self.partner_seat)
        return out

    def wolves_left(self):
        """How many werewolves are still alive, from revealed roles and my
        own checks: 2 minus the wolves known dead or found."""
        known_dead = set(self.dead_wolves)
        known_found = {s for s in self.living() if self.checked.get(s) is True}
        return max(0, 2 - len(known_dead) - len(known_found))

    # -- claims -----------------------------------------------------------
    _SEER_CLAIM = re.compile(
        r"\b(i am|i'm|im|i'm the|i am the)\s+(the\s+)?seer\b"
        r"|\bseer\s+here\b|\bi\s+checked\b|\bmy\s+check\b|\binspected\b",
        re.I,
    )
    _DOCTOR_CLAIM = re.compile(
        r"\b(i am|i'm|im)\s+(the\s+)?doctor\b|\bi\s+protected\b", re.I
    )

    def seer_claims(self):
        """Seats that claimed to be the seer, in order: [(day, seat, text)]."""
        out = []
        for day, seat, _rnd, text in self.speeches:
            if seat == self.my_seat:
                continue
            if self._SEER_CLAIM.search(text or ""):
                out.append((day, seat, text))
        return out

    def doctor_claims(self):
        out = []
        for day, seat, _rnd, text in self.speeches:
            if seat == self.my_seat:
                continue
            if self._DOCTOR_CLAIM.search(text or ""):
                out.append((day, seat, text))
        return out

    def accusations(self):
        """Accusations of being a wolf, parsed from speeches: [(day, accuser,
        accused_seat)]. Matches 'seat N is a wolf', '<name> is a werewolf',
        'X is the wolf', and 'vote seat N / exile N' pushes."""
        out = []
        for day, seat, _rnd, text in self.speeches:
            if not text:
                continue
            t = text
            # seat-number accusations
            for m in re.finditer(r"seat\s+(\d+)\s+is\s+(?:a\s+|the\s+)?(?:werewolf|wolf)\b", t, re.I):
                n = int(m.group(1))
                if 1 <= n <= 8 and n != seat:
                    out.append((day, seat, n))
            # name accusations
            for s2, p in self.players.items():
                nm = p.get("name", "")
                if nm and re.search(r"\b" + re.escape(nm) + r"\b[^.]{0,40}?\b(?:werewolf|wolf)\b", t, re.I):
                    if s2 != seat and (day, seat, s2) not in out:
                        out.append((day, seat, s2))
            # explicit vote pushes
            for m in re.finditer(r"\b(?:vote|exile)\s+(?:seat\s+)?(\d+)\b", t, re.I):
                n = int(m.group(1))
                if 1 <= n <= 8 and n != seat and (day, seat, n) not in out:
                    out.append((day, seat, n))
        return out

    def defended(self, seat):
        """Seats that spoke up for `seat` ('trust X', 'X is innocent', "don't
        vote X", 'X is clear') — a wolf defending its partner, or a villager
        defending a friend."""
        out = []
        nm = self.name(seat)
        pats = [
            r"\btrust\s+(?:seat\s+)?" + re.escape(str(seat)) + r"\b",
            r"\btrust\s+" + re.escape(nm) + r"\b",
            re.escape(nm) + r"\b[^.]{0,30}?\binnocent\b",
            r"\bseat\s+" + str(seat) + r"\b[^.]{0,30}?\binnocent\b",
            r"\bdon'?t\s+vote\s+(?:seat\s+)?" + str(seat) + r"\b",
            r"\bdon'?t\s+vote\s+" + re.escape(nm) + r"\b",
            re.escape(nm) + r"\b[^.]{0,30}?\bclear\b",
        ]
        for day, s2, _rnd, text in self.speeches:
            if s2 == seat or not text:
                continue
            if any(re.search(p, text, re.I) for p in pats):
                out.append((day, s2))
        return out

    def speech_count(self, seat):
        return sum(1 for _, s, _, _ in self.speeches if s == seat)

    def voted_for(self, seat, day=None):
        """Targets `seat` voted for (all days, or one day)."""
        out = []
        for d, votes in self.votes.items():
            if day is not None and d != day:
                continue
            for voter, target in votes:
                if voter == seat and target is not None:
                    out.append((d, target))
        return out

    def votes_for(self, seat, day=None):
        """Who voted for `seat` (all days, or one day)."""
        out = []
        for d, votes in self.votes.items():
            if day is not None and d != day:
                continue
            for voter, target in votes:
                if target == seat:
                    out.append((d, voter))
        return out

    # -- suspicion (village perspective) ----------------------------------
    def suspicion(self):
        """Suspicion score per living seat (not me). Higher = more likely a
        werewolf. Certain knowledge dominates; the rest is weighted evidence."""
        scores = {s: 0 for s in self.living_others()}
        # Certain knowledge from my own seer checks.
        for s in list(scores):
            if self.checked.get(s) is True:
                scores[s] += 10000
            elif self.checked.get(s) is False:
                scores[s] -= 10000
        # Seer claims: a claim whose accused later proved a wolf clears the
        # claimant; one whose accused proved a villager condemns them.
        claimants = {}
        for _day, seat, _text in self.seer_claims():
            claimants.setdefault(seat, []).append(_day)
        accused_by = {}
        for _day, accuser, accused in self.accusations():
            accused_by.setdefault(accuser, []).append(accused)
        for seat in claimants:
            if seat not in scores:
                continue
            if self.my_role == "seer":
                scores[seat] += 800  # I am the seer: this claim is a lie.
                continue
            targets = accused_by.get(seat, [])
            proved_wolf = any(t in self.dead_wolves for t in targets)
            proved_village = any(t in self.dead_village for t in targets)
            if proved_wolf:
                scores[seat] -= 500  # a real seer's check came true
            elif proved_village:
                scores[seat] += 600  # a "check" that was wrong: a liar
            else:
                scores[seat] += 60   # an unproven claim: mild suspicion
        # Two seer claims: the one that has not been contradicted is likelier.
        if len(claimants) >= 2 and self.my_role != "seer":
            for seat in claimants:
                if seat in scores:
                    scores[seat] += 40
        # Vote record.
        for s in list(scores):
            for _d, target in self.voted_for(s):
                if target in self.dead_wolves:
                    scores[s] -= 80   # helped exile a wolf
                elif target in self.dead_village:
                    scores[s] += 40   # pushed a villager out
                elif self.checked.get(target) is True:
                    scores[s] -= 100  # voted for a wolf I found
                elif self.checked.get(target) is False:
                    scores[s] += 50   # voted for a villager I cleared
        # Defending a revealed wolf.
        for wolf in self.dead_wolves:
            for _day, defender in self.defended(wolf):
                if defender in scores:
                    scores[defender] += 120
        # Accused by someone who later died as the seer, or by a villager the
        # wolves killed at night (the wolves silence those onto them).
        for _day, accuser, accused in self.accusations():
            if accused not in scores:
                continue
            if accuser in self.dead_wolves:
                continue
            accuser_role = self.revealed.get(accuser)
            if accuser_role == "seer":
                scores[accused] += 400
            elif accuser in [s for _d, s, _r in self.night_deaths if _r in ("seer", "villager", "doctor")]:
                scores[accused] += 150
        # Quiet players, late in the game, are mildly suspicious.
        if self.day >= 2:
            for s in list(scores):
                if self.speech_count(s) == 0:
                    scores[s] += 20
        return scores

    def top_suspect(self, exclude=frozenset()):
        """Living seat (not me) with the highest suspicion, skipping `exclude`
        and seats I know are village. Deterministic tie-break: lowest seat."""
        scores = self.suspicion()
        best, best_score = None, None
        for s in sorted(scores):
            if s in exclude or self.is_clear(s):
                continue
            if best_score is None or scores[s] > best_score:
                best, best_score = s, scores[s]
        return best, best_score

    # -- village sentiment (what the table leans toward) -------------------
    def most_accused(self):
        """Living seat with the most public accusations against it."""
        counts = {}
        for _day, _accuser, accused in self.accusations():
            if accused in self.living() and accused != self.my_seat:
                counts[accused] = counts.get(accused, 0) + 1
        if not counts:
            return None, 0
        best = max(sorted(counts), key=lambda s: counts[s])
        return best, counts[best]

    def claimed_seer(self):
        """The most credible living seer claimant, if any: the earliest claim
        whose accused proved a wolf, else the earliest claim still standing."""
        claims = self.seer_claims()
        living_claims = [(d, s) for d, s, _ in claims if s in self.living()]
        if not living_claims:
            return None
        accused_by = {}
        for _day, accuser, accused in self.accusations():
            accused_by.setdefault(accuser, []).append(accused)
        for _d, s in living_claims:
            if any(t in self.dead_wolves for t in accused_by.get(s, [])):
                return s
        return living_claims[0][1]

    def i_claimed_seer(self):
        """Did I already claim seer in this game?"""
        for day, seat, _rnd, text in self.speeches:
            if seat == self.my_seat and self._SEER_CLAIM.search(text or ""):
                return True
        return False

    def threat_to_wolves(self, seat):
        """How dangerous a living non-wolf is to the wolves (for kill choice).
        Claimed seers first, then players accusing me or my partner, then
        vocal, accurate villagers."""
        score = 0
        if seat in [s for _d, s, _t in self.seer_claims()]:
            score += 100
        for _d, accuser, accused in self.accusations():
            if accuser == seat and accused in (self.my_seat, self.partner_seat):
                score += 60
        for _d, target in self.voted_for(seat):
            if target in self.dead_wolves:
                score += 40
        score += min(20, 5 * self.speech_count(seat))
        return score


# --------------------------------------------------------------------------
# Statement templates (chosen by seeded RNG, so the same game replays the
# same words). All are folded and cut to the decision's max_chars.
# --------------------------------------------------------------------------

def _pick(rng, templates, **kw):
    text = rng.choice(templates)
    return text.format(**kw)


VILLAGER_OPENERS = [
    "Day {day}: no wolf found yet. Who looks off to everyone? I want to hear from {name}.",
    "Let us go one by one. {name}, you have been quiet — what do you make of last night?",
    "I am watching the votes. Whoever pushed {name} yesterday should explain why.",
    "No strong read yet. {name} felt off to me — too quick to agree with everyone.",
    "Let us not rush. {name}, what is your read on the table so far?",
]

VILLAGER_ACCUSE = [
    "{name} (seat {seat}) is my top suspect: {reason} Vote {name}.",
    "I think it is {name}, seat {seat}. {reason} That is where my vote goes.",
    "{name} has to explain this: {reason} I am voting {name}.",
]

VILLAGER_DEFEND = [
    "I am not a wolf. I have voted with the village and pushed real suspects. Look at {name} instead.",
    "That accusation does not hold. Check my votes — I have been on the village side. {name} is the one to watch.",
    "I am a villager. If you exile me you lose a vote and the wolves gain a night. {name} is the better target.",
]

VILLAGER_SUPPORT_SEER = [
    "The seer checked {name} and found a wolf. That is the clearest lead we have — vote {name}.",
    "If {seer} is the seer, {name} is a wolf. I trust the check. Vote {name}.",
    "{name} is a wolf by the seer's check. Let us not waste the day — vote {name}.",
]

VILLAGER_SAVE_CLEAR = [
    "{name} is not a wolf — I checked {name} myself. Do not waste this exile.",
    "Hold on. I checked {name} and {name} is clear. Look elsewhere.",
]

WOLF_BLEND = [
    "I agree we should look at {name}. {name} has been too quiet for my liking.",
    "Not much to add. I think {name} deserves a closer look today.",
    "Let us hear from {name} before we vote — {name} has said almost nothing.",
    "I am leaning {name} today. Something about the way {name} deflects feels off.",
    "Can we talk about {name}? The votes yesterday did not add up to me.",
]

WOLF_DEFEND = [
    "I am not a wolf. I have been trying to find them like everyone else. {name} is a better lead.",
    "That is a stretch. I have nothing to hide — look at {name}, who keeps changing their story.",
    "Exiling me helps the wolves. {name} is the one pushing weak accusations.",
]

SEER_CLAIM_WOLF = [
    "I am the seer. I checked {name} last night and {name} is a werewolf. Vote {name}.",
    "Seer here. My check last night: {name}, seat {seat}, is a wolf. That is where my vote goes.",
    "I checked {name} on night {night} — a werewolf. I am the seer, vote {name}.",
]

SEER_CLAIM_REAL = [
    "I am the seer. I checked {name} on night {night} and {name} is a werewolf. Vote {name}.",
    "Seer here. Night {night} I inspected {name}: a wolf. Vote {name} today.",
    "My inspection says {name} is a werewolf (night {night}). I am the seer — vote {name}.",
]

SEER_COUNTER = [
    "I am the real seer. {name} is lying — there is only one seer and it is me. {name} is the wolf to exile.",
    "{name} is not the seer, I am. A false seer claim is a wolf move. Vote {name}.",
]

SEER_CLEAR = [
    "I am the seer and I checked {name}: not a wolf. Do not exile {name}.",
    "Hold on — I inspected {name} and {name} is clear. I am the seer. Look elsewhere.",
]


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

def _rng_for(seed, game, kind, day):
    return Rng(seed, "%s:%s:%s" % (kind, game, day))


def decide_speak(a, game, seed):
    """Return {'text': str} or {'pass': True}."""
    rng = _rng_for(seed, game, "speak", a.day)
    living = a.living_others()
    if not living:
        return {"pass": True}
    me = a.my_seat
    role = a.my_role

    # --- werewolf ---------------------------------------------------------
    if role == "werewolf":
        # Seer-claim bluff, once, on day 1-2, only if nobody credibly claimed.
        if a.day <= 2 and not a.i_claimed_seer() and a.claimed_seer() is None:
            target, _n = a.most_accused()
            if target is None or target == a.partner_seat:
                cands = [s for s in living if s != a.partner_seat]
                target = cands[0] if cands else None
            if target is not None and target != a.partner_seat:
                text = _pick(
                    rng, SEER_CLAIM_WOLF,
                    name=a.name(target), seat=target, night=max(1, a.day - 1),
                )
                return {"text": cut_statement(text)}
        # Accused? Defend and redirect.
        accusers = {acc for _d, acc, t in a.accusations() if t == me}
        if accusers:
            others = [s for s in living if s != a.partner_seat]
            redirect = others[0] if others else None
            if redirect is not None:
                text = _pick(rng, WOLF_DEFEND, name=a.name(redirect))
                return {"text": cut_statement(text)}
        # Otherwise blend in: point at a plausible villager.
        suspect, _ = a.most_accused()
        if suspect is None or suspect == a.partner_seat:
            cands = [s for s in living if s != a.partner_seat]
            suspect = rng.choice(cands) if cands else None
        if suspect is None:
            return {"pass": True}
        text = _pick(rng, WOLF_BLEND, name=a.name(suspect))
        return {"text": cut_statement(text)}

    # --- seer -------------------------------------------------------------
    if role == "seer":
        live_wolves = [s for s in a.living() if a.checked.get(s) is True]
        if live_wolves:
            target = live_wolves[0]
            night = next(
                (r["night"] for r in a.seer_results if r.get("seat") == target), a.day
            )
            text = _pick(
                rng, SEER_CLAIM_REAL,
                name=a.name(target), seat=target, night=night,
            )
            return {"text": cut_statement(text)}
        # Someone else claimed seer: counter-claim (there is only one).
        claimants = [s for _d, s, _t in a.seer_claims() if s in a.living()]
        if claimants:
            text = _pick(rng, SEER_COUNTER, name=a.name(claimants[0]))
            return {"text": cut_statement(text)}
        # A cleared player is being pushed: save them.
        target, n = a.most_accused()
        if target is not None and a.checked.get(target) is False and n >= 1:
            text = _pick(rng, SEER_CLEAR, name=a.name(target))
            return {"text": cut_statement(text)}
        # Otherwise push the top suspect like a villager would: the village
        # needs a focal point, and a seer who never speaks wastes the role.
        target, _ = a.top_suspect()
        if target is None:
            return {"pass": True}
        text = _pick(
            rng, VILLAGER_ACCUSE, name=a.name(target), seat=target,
            reason="their story does not add up.",
        )
        return {"text": cut_statement(text)}

    # --- doctor / villager -------------------------------------------------
    # A credible seer accused someone: back it.
    seer = a.claimed_seer()
    if seer is not None and seer != me:
        accused = [t for _d, acc, t in a.accusations() if acc == seer and t in a.living()]
        if accused:
            target = accused[0]
            if not a.is_clear(target):
                text = _pick(
                    rng, VILLAGER_SUPPORT_SEER,
                    name=a.name(target), seer=a.name(seer),
                )
                return {"text": cut_statement(text)}
    # Accused? Defend and redirect.
    if any(t == me for _d, _acc, t in a.accusations()):
        suspect, _ = a.top_suspect()
        if suspect is not None:
            text = _pick(rng, VILLAGER_DEFEND, name=a.name(suspect))
            return {"text": cut_statement(text)}
    # Push the top suspect, or open the day. Always name someone: a village
    # with no focal point exiles at random, and a stated suspect with a
    # reason is something the table can judge and follow.
    suspect, score = a.top_suspect()
    if suspect is not None:
        reason = "their votes and their story point the same way."
        text = _pick(
            rng, VILLAGER_ACCUSE, name=a.name(suspect), seat=suspect, reason=reason,
        )
        return {"text": cut_statement(text)}
    if a.day == 1 and a.speech_count(me) == 0:
        others = [s for s in living]
        text = _pick(rng, VILLAGER_OPENERS, day=a.day, name=a.name(rng.choice(others)))
        return {"text": cut_statement(text)}
    return {"pass": True}


def decide_vote(a, game, seed, options):
    """Return a seat (from options) or None to abstain."""
    living_opts = [s for s in (options or []) if s in a.living() and s != a.my_seat]
    if not living_opts:
        return None
    role = a.my_role

    # --- werewolf: vote a villager the table leans toward; never partner.
    if role == "werewolf":
        # If I claimed seer and accused someone, vote them (consistency).
        if a.i_claimed_seer():
            accused = [t for _d, acc, t in a.accusations()
                       if acc == a.my_seat and t in living_opts and t != a.partner_seat]
            if accused:
                return accused[0]
        target, n = a.most_accused()
        if target in living_opts and target != a.partner_seat and n >= 1:
            return target
        # Otherwise vote the villager the village suspects most (by my
        # suspicion scores, excluding my partner and known wolves).
        suspect, _score = a.top_suspect(exclude=frozenset([a.partner_seat] if a.partner_seat else []))
        if suspect in living_opts:
            return suspect
        cands = [s for s in living_opts if s != a.partner_seat]
        return cands[0] if cands else None

    # --- village: vote a known wolf, else the top suspect.
    for s in living_opts:
        if a.checked.get(s) is True:
            return s
    seer = a.claimed_seer()
    if seer is not None and seer != a.my_seat:
        accused = [t for _d, acc, t in a.accusations() if acc == seer and t in living_opts]
        if accused and not a.is_clear(accused[0]):
            return accused[0]
    suspect, _score = a.top_suspect()
    if suspect in living_opts:
        return suspect
    return living_opts[0]


def decide_kill(a, game, seed, options):
    """Return a seat (from options) or None to abstain (no kill)."""
    living_opts = [s for s in (options or []) if s in a.living() and s != a.partner_seat]
    if not living_opts:
        return None
    # Partner already named someone tonight: match them, so the kill lands.
    tonight = [v for v in a.wolf_votes
               if v.get("night") == a.day and v.get("seat") == a.partner_seat
               and v.get("target") is not None]
    if tonight:
        target = tonight[0]["target"]
        if target in living_opts:
            return target
    # First to vote: name the most dangerous living non-wolf.
    best, best_score = None, None
    for s in sorted(living_opts):
        score = a.threat_to_wolves(s)
        if best_score is None or score > best_score:
            best, best_score = s, score
    return best


def decide_protect(a, game, seed, options):
    """Return a seat (from options) or None to protect nobody."""
    last = (a.last_protect or {}).get("seat")
    living_opts = [s for s in (options or [])
                   if s in a.living() and s != a.my_seat and s != last]
    if not living_opts:
        return None
    # Protect a credibly claimed seer: the wolves' top target.
    seer = a.claimed_seer()
    if seer is not None and seer in living_opts:
        return seer
    # Else the most useful villager: the most vocal living player who is not
    # under suspicion, else the lowest living seat (deterministic).
    scores = a.suspicion()
    best, best_key = None, None
    for s in sorted(living_opts):
        key = (a.speech_count(s), -scores.get(s, 0))
        if best_key is None or key > best_key:
            best, best_key = s, key
    return best


def decide_inspect(a, game, seed, options):
    """Return a seat (from options) or None to inspect nobody."""
    living_opts = [s for s in (options or []) if s in a.living() and s != a.my_seat]
    if not living_opts:
        return None
    unchecked = [s for s in living_opts if s not in a.checked]
    pool = unchecked if unchecked else living_opts
    # Most suspicious unchecked seat; deterministic tie-break, lowest seat.
    scores = a.suspicion()
    best, best_score = None, None
    for s in sorted(pool):
        if best_score is None or scores.get(s, 0) > best_score:
            best, best_score = s, scores.get(s, 0)
    return best


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def decide(a, game, seed, decision, options, max_chars):
    """One decision's action body (without game/decision keys)."""
    kind = decision
    if kind == "speak":
        r = decide_speak(a, game, seed)
        if r.get("pass"):
            return {"pass": True}
        text = cut_statement(r["text"], max_chars or 500)
        if not text:
            return {"pass": True}
        return {"text": text}
    if kind == "vote":
        target = decide_vote(a, game, seed, options)
        return {"target": target}
    if kind == "kill":
        target = decide_kill(a, game, seed, options)
        return {"target": target}
    if kind == "protect":
        target = decide_protect(a, game, seed, options)
        return {"target": target}
    if kind == "inspect":
        target = decide_inspect(a, game, seed, options)
        return {"target": target}
    return {"pass": True}


def fallback_for(decision):
    if decision == "speak":
        return {"pass": True}
    return {"target": None}


def run():
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    seed = os.environ.get("MAKEFUN_TRIAL_SEED") or "0"
    if not env_url:
        raise EnvError("MAKEFUN_ENV_URL is not set")
    env = Env(env_url)
    decisions = 0
    acted = 0
    fallbacks = 0
    errors = []
    # A decision is keyed by game + kind + log length, so /observe listing it
    # again is never answered twice.
    started = set()
    for _ in range(100000):
        try:
            obs = env.observe()
        except EnvError as err:
            errors.append(str(err))
            if len(errors) >= 5:
                break
            time.sleep(1)
            continue
        errors = []
        if obs.get("done"):
            break
        pending = obs.get("pending") or []
        if not pending:
            continue
        for p in pending:
            game = p.get("game")
            decision = p.get("decision")
            view = p.get("view") or {}
            key = "%s:%s:%s" % (game, decision, len(view.get("log", []) or []))
            if key in started:
                continue
            started.add(key)
            decisions += 1
            try:
                a = Analysis(view)
                action = decide(
                    a, game, seed, decision,
                    p.get("options"), p.get("max_chars"),
                )
            except Exception:
                action = fallback_for(decision)
                fallbacks += 1
            body = {"game": game, "decision": decision}
            body.update(action)
            try:
                status, _resp = env.act(game, decision, **{k: v for k, v in body.items() if k not in ("game", "decision")})
            except EnvError:
                continue
            if status == 200:
                acted += 1
            elif status in (400, 409):
                # Refused: fall back once (pass / abstain), then move on.
                fb = fallback_for(decision)
                if fb != {k: v for k, v in body.items() if k not in ("game", "decision")}:
                    try:
                        status2, _ = env.act(game, decision, **fb)
                        if status2 == 200:
                            acted += 1
                        else:
                            fallbacks += 1
                    except EnvError:
                        fallbacks += 1
                else:
                    fallbacks += 1
            else:
                fallbacks += 1
    sys.stdout.write(
        "werewolf: %d decisions, %d acted, %d fell back\n" % (decisions, acted, fallbacks)
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except EnvError as err:
        sys.stderr.write("werewolf: %s\n" % err)
        sys.exit(1)

