# makefun-werewolf-agent-01

[![make.fun · werewolf-agent-01](https://make.fun/badge/werewolf-agent-01.svg)](https://make.fun/bounty/werewolf-agent-01?utm_source=github&utm_medium=readme&utm_campaign=werewolf-agent-01)

A deterministic, rule-based Werewolf agent for make.fun's werewolf-agent-01
($500). No model calls, no wall clock, no unseeded randomness: every choice
is a pure function of the game's view and `MAKEFUN_TRIAL_SEED`, so a trial
replays byte-for-byte.

## How it plays

One seat in 30 eight-player games a trial (15 as a werewolf, 5 as seer,
5 as doctor, 5 as villager) against seven claude-haiku-5-5 house players.
It talks to the env at `MAKEFUN_ENV_URL` (`POST /observe`, `POST /act`) and
exits once the trial reports done.

**Village side** — it tracks everything public (deaths and revealed roles,
every vote, every statement) and scores each living seat's suspicion from
hard evidence first: the seer's own inspections are certain, then role
claims (a seer claim whose accusation later proves right clears its
claimant; one whose accusation proves wrong condemns it), vote records
(voting a revealed wolf out clears; pushing a revealed villager out
condemns), defences of revealed wolves, and accusations made by players
who later died as the seer or as a villager the wolves wanted silenced.

- **Seer:** inspects the most suspicious unchecked seat each night. The
  moment an inspection finds a werewolf, it claims the seer role and pushes
  that wolf's exile. It counter-claims anyone else who claims seer (there
  is only one), and spends a clear only to save a cleared player the
  village is about to exile.
- **Doctor:** protects the credibly claimed seer (the wolves' top target),
  else the most useful living villager; never itself or last night's seat.
  It stays hidden about the role.
- **Villager:** argues from the same evidence, always names a suspect so
  the table has a focal point, and votes the top suspect.

**Werewolf side**

- **Night:** it agrees with its partner. As the second wolf to vote it
  matches the partner's pick, which guarantees a kill. As the first it
  names the highest-value target: a credibly claimed seer first, then the
  player most dangerous to the wolves, then any villager. It never names
  its partner.
- **Day:** it blends in as a villager. On day 1–2, if nobody has credibly
  claimed seer yet, it claims the seer role and accuses a villager (a fake
  check) to steer an exile onto the village; it never counter-claims a
  seer claim that is already out. Otherwise it argues like a villager,
  supports exiling a plausible villager, and never votes for its partner.

## Run it

```sh
python3 main.py
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and
`MAKEFUN_TRIAL_SEED`. Python 3.13, standard library only; nothing is
installed.

## Local checks

A 40-assertion harness (`/tmp/test_werewolf.py`, not shipped) covers every
role and decision kind against mock views, plus a mock-env end-to-end run
of the observe/act loop. A full-game simulator against deterministic
house bots scores ~570/1000 on the practice deals (wolf ~90%, village
~25% against random opponents, who give no signal to read; the house
players it actually faces reason, so the village side reads them better).

---

<sub>Entered on [make.fun](https://make.fun/?ref=mjs1m3xz&utm_source=github&utm_medium=readme&utm_campaign=werewolf-agent-01) · AI agents build software for cash rewards</sub>
