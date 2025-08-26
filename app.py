# === Colab setup ===
!pip -q install ortools pandas rapidfuzz ipywidgets

import math
import pandas as pd
from rapidfuzz import process, fuzz
from ortools.linear_solver import pywraplp
from google.colab import files
import ipywidgets as w
from IPython.display import display, clear_output

# ============ Upload & load your data ============
print("Please upload your CSV (must contain columns: Position, Player, Team, Points, VOR, AAV).")
uploaded = files.upload()
if not uploaded:
    raise ValueError("No file uploaded.")
CSV_PATH = list(uploaded.keys())[0]
print(f"Loaded: {CSV_PATH}")

# Read & clean
df = pd.read_csv(CSV_PATH)
expected_cols = ["Position", "Player", "Team", "Points", "VOR", "AAV"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

for col in ["Points", "VOR", "AAV"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["VOR", "AAV"]).copy()

# Normalize positions
def norm_pos(p):
    p = str(p).strip().upper()
    mapping = {
        "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
        "K": "K", "PK": "K", "DST": "DEF", "D/ST": "DEF", "DEF": "DEF"
    }
    return mapping.get(p, p)

df["Position"] = df["Position"].map(norm_pos)
valid_pos = {"QB", "RB", "WR", "TE", "K", "DEF"}
df = df[df["Position"].isin(valid_pos)].reset_index(drop=True)

# ======== Optimizer ========
SLOTS = ["QB", "WR", "TE", "FLEX", "DEF", "K"]   # keepers are 2 RB + 1 WR; RB only eligible for FLEX here

def eligible_slots_for_position(pos):
    if pos == "QB":  return ["QB"]
    if pos == "WR":  return ["WR", "FLEX"]
    if pos == "TE":  return ["TE", "FLEX"]
    if pos == "RB":  return ["FLEX"]
    if pos == "K":   return ["K"]
    if pos == "DEF": return ["DEF"]
    return []

class DraftOptimizer:
    def __init__(self, df, budget=141):
        self.df_all = df.copy()
        self.budget_start = int(budget)  # full cap for these 6 slots
        self.budget_left = int(budget)   # live wallet
        self.owned = {}                  # name -> price_paid
        self.gone = set()
        self.name_to_rows = {}
        for idx, row in self.df_all.iterrows():
            self.name_to_rows.setdefault(row["Player"], []).append(int(idx))

        # --- Compute "ideal" pre-draft ceiling VOR once (nobody owned/gone) ---
        # This is the denominator for grades (Ideal = 100%)
        ideal_sol = self._solve_maxvor_tiebreak_minspend_internal(
            candidates=self.df_all.copy().reset_index(drop=True),
            force_owned=False,               # ignore any owned at init
            forced_player=None, forced_price=None
        )
        self.ideal_vor = ideal_sol["total_vor"] if ideal_sol["ok"] else None

    # ---- helpers ----
    def _match_name(self, name):
        if name in self.name_to_rows:
            return name
        choices = list(self.name_to_rows.keys())
        match, score, _ = process.extractOne(name, choices, scorer=fuzz.WRatio)
        if score >= 85:
            return match
        raise ValueError(f"Could not confidently match '{name}'. Closest was '{match}' ({score}).")

    def gone_player(self, name):
        n = self._match_name(name)
        if n in self.owned:
            return f"{n} already owned; not marking gone."
        self.gone.add(n)
        return f"Marked gone: {n}"

    # ---------- CORE SOLVERS ----------
    def _candidates(self):
        mask = ~self.df_all["Player"].isin(self.gone)
        c = self.df_all[mask].copy().reset_index(drop=True)
        return c

    def _build_solver(self, candidates):
        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            raise RuntimeError("Failed to create CBC solver.")
        x = {}
        for i, r in candidates.iterrows():
            pos = r["Position"]
            for slot in eligible_slots_for_position(pos):
                x[(i, slot)] = solver.BoolVar(f"x_{i}_{slot}")
        # Each slot exactly 1 player
        for slot in SLOTS:
            solver.Add(sum(x[(i, slot)] for i, r in candidates.iterrows() if (i, slot) in x) == 1)
        # Each player at most once
        for i, r in candidates.iterrows():
            solver.Add(sum(x[(i, slot)] for slot in SLOTS if (i, slot) in x) <= 1)
        return solver, x

    def _force_owned_and_optional_nominee(self, solver, x, candidates, forced_player=None, include_owned=True):
        name_to_idx = {}
        for i, r in candidates.iterrows():
            name_to_idx.setdefault(r["Player"], []).append(i)

        if include_owned:
            for name in self.owned.keys():
                if name in name_to_idx:
                    idxs = name_to_idx[name]
                    solver.Add(sum(x[(i, slot)] for i in idxs for slot in SLOTS if (i, slot) in x) == 1)

        forced_index = None
        if forced_player is not None:
            fname = self._match_name(forced_player)
            if fname not in name_to_idx:
                return None, f"Nominee {fname} not in candidate set."
            forced_index = name_to_idx[fname][0]
            solver.Add(sum(x[(forced_index, slot)] for slot in SLOTS if (forced_index, slot) in x) == 1)
        return forced_index, None

    def _price_used(self, i, row, forced_index, forced_price):
        name = row["Player"]
        if forced_index is not None and i == forced_index and forced_price is not None:
            return float(forced_price)
        if name in self.owned:
            return float(self.owned[name])
        return float(row["AAV"])

    # ----- internal variants so we can compute the "ideal" ignoring owned/gone -----
    def _solve_maxvor_tiebreak_minspend_internal(self, candidates, force_owned=True, forced_player=None, forced_price=None):
        if candidates.empty:
            return {"ok": False, "reason": "No candidates available."}

        solver, x = self._build_solver(candidates)
        forced_index, err = self._force_owned_and_optional_nominee(
            solver, x, candidates, forced_player=forced_player, include_owned=force_owned
        )
        if err:
            return {"ok": False, "reason": err}

        total_spend = solver.Sum(self._price_used(i, candidates.loc[i], forced_index, forced_price) *
                                 x[(i, slot)] for (i, slot) in x)
        solver.Add(total_spend <= float(self.budget_start))  # cap for these 6 slots

        total_vor = solver.Sum(float(candidates.loc[i, "VOR"]) * x[(i, slot)] for (i, slot) in x)
        solver.Maximize(total_vor)
        s1 = solver.Solve()
        if s1 != pywraplp.Solver.OPTIMAL:
            return {"ok": False, "reason": "Infeasible or no optimal solution in phase 1."}

        best_vor = total_vor.solution_value()
        tiny = 1e-6
        solver.Add(total_vor >= best_vor - tiny)
        solver.Add(total_vor <= best_vor + tiny)
        solver.Minimize(total_spend)
        s2 = solver.Solve()
        if s2 != pywraplp.Solver.OPTIMAL:
            return {"ok": False, "reason": "No optimal tie-break solution."}

        chosen, tot_sp, owned_sp = [], 0.0, 0.0
        for (i, slot), var in x.items():
            if var.solution_value() > 0.5:
                row = candidates.loc[i]
                p = self._price_used(i, row, forced_index, forced_price)
                tot_sp += p
                if row["Player"] in self.owned and force_owned:
                    owned_sp += p
                chosen.append({
                    "slot": slot, "name": row["Player"], "pos": row["Position"],
                    "team": row.get("Team", ""), "vor": float(row["VOR"]),
                    "aav": float(row["AAV"]), "price_used": p
                })

        return {
            "ok": True, "total_vor": best_vor, "total_spend": tot_sp,
            "owned_spend": owned_sp, "future_spend_needed": max(0.0, tot_sp - owned_sp),
            "lineup": sorted(chosen, key=lambda d: SLOTS.index(d["slot"]))
        }

    def _solve_min_spend_any_vor_internal(self, candidates, force_owned=True, forced_player=None, forced_price=None):
        if candidates.empty:
            return {"ok": False, "reason": "No candidates available."}

        solver, x = self._build_solver(candidates)
        forced_index, err = self._force_owned_and_optional_nominee(
            solver, x, candidates, forced_player=forced_player, include_owned=force_owned
        )
        if err:
            return {"ok": False, "reason": err}

        total_spend = solver.Sum(self._price_used(i, candidates.loc[i], forced_index, forced_price) *
                                 x[(i, slot)] for (i, slot) in x)
        solver.Add(total_spend <= float(self.budget_start))
        solver.Minimize(total_spend)
        s = solver.Solve()
        if s != pywraplp.Solver.OPTIMAL:
            return {"ok": False, "reason": "No feasible min-spend completion."}

        chosen, tot_sp, owned_sp = [], 0.0, 0.0
        for (i, slot), var in x.items():
            if var.solution_value() > 0.5:
                row = candidates.loc[i]
                p = self._price_used(i, row, forced_index, forced_price)
                tot_sp += p
                if row["Player"] in self.owned and force_owned:
                    owned_sp += p
                chosen.append({
                    "slot": slot, "name": row["Player"], "pos": row["Position"],
                    "team": row.get("Team", ""), "vor": float(row["VOR"]),
                    "aav": float(row["AAV"]), "price_used": p
                })

        return {
            "ok": True, "total_spend": tot_sp, "owned_spend": owned_sp,
            "future_spend_needed": max(0.0, tot_sp - owned_sp),
            "lineup": sorted(chosen, key=lambda d: SLOTS.index(d["slot"]))
        }

    # Public wrappers under current constraints (owned/gone respected)
    def _solve_maxvor_tiebreak_minspend(self, forced_player=None, forced_price=None):
        return self._solve_maxvor_tiebreak_minspend_internal(
            candidates=self._candidates(), force_owned=True,
            forced_player=forced_player, forced_price=forced_price
        )

    def _solve_min_spend_any_vor(self, forced_player=None, forced_price=None):
        return self._solve_min_spend_any_vor_internal(
            candidates=self._candidates(), force_owned=True,
            forced_player=forced_player, forced_price=forced_price
        )

    def best_lineup(self):
        return self._solve_maxvor_tiebreak_minspend()

    # ---- Grading helpers ----
    def _grade(self, vor):
        if self.ideal_vor and self.ideal_vor > 0:
            return 100.0 * float(vor) / float(self.ideal_vor)
        return None

    # VOR-based cap intersected with affordability cap + grades
    def compute_threshold(self, name):
        n = self._match_name(name)

        # VOR-preserving threshold
        base = self._solve_maxvor_tiebreak_minspend()
        if not base["ok"]:
            return {"ok": False, "reason": base.get("reason", "No baseline solution.")}

        base_vor = base["total_vor"]
        trial1 = self._solve_maxvor_tiebreak_minspend(forced_player=n, forced_price=1)
        if not trial1["ok"] or trial1["total_vor"] + 1e-6 < base_vor:
            vor_cap = 0
            with_lineup = None
        else:
            lo, hi = 1, int(self.budget_left)  # never recommend beyond wallet
            vor_cap, with_lineup = lo, trial1
            while lo <= hi:
                mid = (lo + hi) // 2
                t = self._solve_maxvor_tiebreak_minspend(forced_player=n, forced_price=mid)
                if t["ok"] and t["total_vor"] + 1e-6 >= base_vor:
                    vor_cap, with_lineup = mid, t
                    lo = mid + 1
                else:
                    hi = mid - 1

        # Affordability cap
        minfill = self._solve_min_spend_any_vor(forced_player=n, forced_price=1)
        if not minfill["ok"]:
            afford_cap = 0
        else:
            need = math.ceil(minfill["future_spend_needed"])
            afford_cap = max(0, int(self.budget_left - need))

        final_cap = min(vor_cap, afford_cap)

        # --- Grades ---
        baseline_grade = self._grade(base["total_vor"])
        with_grade = self._grade(with_lineup["total_vor"]) if with_lineup else None

        return {
            "ok": True, "player": n,
            "max_bid": final_cap, "vor_cap": vor_cap, "afford_cap": afford_cap,
            "baseline": base, "with_lineup": with_lineup if final_cap > 0 else None,
            "baseline_grade": baseline_grade, "with_grade": with_grade,
            "ideal_vor": self.ideal_vor
        }

    # ---- Buy with affordability guard ----
    def buy(self, name, price):
        n = self._match_name(name)
        price = int(price)

        # Affordability check BEFORE committing
        minfill = self._solve_min_spend_any_vor(forced_player=n, forced_price=1)
        if not minfill["ok"]:
            return f"Cannot buy {n}: no feasible way to complete roster under the cap."
        need = math.ceil(minfill["future_spend_needed"])
        max_affordable = max(0, int(self.budget_left - need))
        if price > max_affordable:
            return (f"Blocked: Buying {n} at ${price} would leave only ${self.budget_left - price}, "
                    f"but you need at least ${need} to fill the remaining slots. "
                    f"Max you can pay: ${max_affordable}.")

        # Commit the purchase
        if n in self.gone:
            self.gone.remove(n)
        prev = self.owned.get(n)
        if prev is not None:
            self.budget_left += prev
        self.owned[n] = price
        self.budget_left -= price
        msg = f"Added {n} for ${price}. Budget left: ${self.budget_left}"
        if self.budget_left < 0:
            msg += "  (Warning: budget negative)"
        return msg

# Helper: table formatting
def sol_to_df(sol):
    if not sol or not sol.get("ok"):
        return pd.DataFrame()
    rows = []
    for r in sol["lineup"]:
        rows.append({
            "Slot": r["slot"],
            "Player": r["name"],
            "Pos": r["pos"],
            "Team": r["team"],
            "VOR": round(r["vor"], 2),
            "PriceUsed": int(round(r["price_used"])),
            "AAV": int(round(r["aav"]))
        })
    order = {s:i for i,s in enumerate(SLOTS)}
    out = pd.DataFrame(rows).sort_values("Slot", key=lambda s: s.map(order))
    out.index = range(1, len(out)+1)
    return out

# ==========================
# Create optimizer
# ==========================
opt = DraftOptimizer(df, budget=141)

# ==========================
# UI (per-position dropdowns)
# ==========================
hide_gone_owned = w.Checkbox(value=True, description="Hide owned/gone")
show_lineups = w.Checkbox(value=False, description="Show details")

SENTINEL = ("— Select —", "")
def options_for_pos(pos):
    d = df[df["Position"] == pos]
    if hide_gone_owned.value:
        d = d[~d["Player"].isin(opt.gone)]
        d = d[~d["Player"].isin(opt.owned.keys())]
    names = sorted(d["Player"].unique().tolist())
    return [SENTINEL] + [(n, n) for n in names]

dq  = w.Dropdown(options=options_for_pos("QB"),  description="QB:")
drb = w.Dropdown(options=options_for_pos("RB"),  description="RB:")
dwr = w.Dropdown(options=options_for_pos("WR"),  description="WR:")
dte = w.Dropdown(options=options_for_pos("TE"),  description="TE:")
dk  = w.Dropdown(options=options_for_pos("K"),   description="K:")
ddef= w.Dropdown(options=options_for_pos("DEF"), description="DEF:")

reco_label = w.HTML("<b>Bid up to:</b> —")
caps_label = w.HTML("")   # shows (VOR cap vs Afford cap) when details on
grade_label = w.HTML("Grades: —")  # NEW: percentage grades vs ideal ceiling
buy_price = w.Text(value="", placeholder="Won @ $", description="Won @ $:")
buy_btn = w.Button(description="Buy", button_style="success")
gone_btn = w.Button(description="Gone", button_style="warning")
output = w.Output()

def budget_html():
    return f"<b>Budget left:</b> ${opt.budget_left} &nbsp;|&nbsp; <b>Total cap (6 slots):</b> ${opt.budget_start}"

budget_label = w.HTML(budget_html())

def refresh_all_dropdowns(_=None):
    dq.options   = options_for_pos("QB")
    drb.options  = options_for_pos("RB")
    dwr.options  = options_for_pos("WR")
    dte.options  = options_for_pos("TE")
    dk.options   = options_for_pos("K")
    ddef.options = options_for_pos("DEF")

hide_gone_owned.observe(refresh_all_dropdowns, names="value")

def get_selected_player():
    for dd in (dq, drb, dwr, dte, dk, ddef):
        if dd.value:
            return dd.value
    return ""

def clear_others(except_dd):
    for dd in (dq, drb, dwr, dte, dk, ddef):
        if dd is not except_dd:
            dd.value = ""  # sentinel

def fmt_pct(x):
    return "—" if x is None else f"{x:.1f}%"

def update_recommendation(*args):
    player = get_selected_player()
    caps_label.value = ""
    if not player:
        reco_label.value = "<b>Bid up to:</b> —"
        grade_label.value = "Grades: —"
        with output:
            clear_output(wait=True)
        return
    res = opt.compute_threshold(player)
    if not res["ok"]:
        reco_label.value = "<b>Bid up to:</b> (no solution)"
        grade_label.value = "Grades: —"
        with output:
            clear_output(wait=True)
            print(res.get("reason", "No solution."))
        return
    reco_label.value = f"<b>Bid up to:</b> ${res['max_bid']}"
    if show_lineups.value:
        caps_label.value = f"(VOR cap: ${res['vor_cap']}, Afford cap: ${res['afford_cap']})"

    # NEW: Grades display
    grade_label.value = (
        f"Grades: Fallback {fmt_pct(res.get('baseline_grade'))} | "
        f"With {fmt_pct(res.get('with_grade'))} | "
        f"Ideal 100%"
    )

    with output:
        clear_output(wait=True)
        if show_lineups.value:
            print("Fallback lineup (without nominee):")
            display(sol_to_df(res["baseline"]))
            if res["with_lineup"]:
                print(f"\nBest lineup WITH {res['player']} at ${res['max_bid']}:")
                display(sol_to_df(res["with_lineup"]))
    budget_label.value = budget_html()

def on_dd_change(change, dd):
    if change["name"] == "value":
        if dd.value:
            clear_others(dd)
        update_recommendation()

for dd in (dq, drb, dwr, dte, dk, ddef):
    dd.observe(lambda ch, d=dd: on_dd_change(ch, d), names="value")

def on_buy_clicked(_):
    player = get_selected_player()
    val = buy_price.value.strip()
    with output:
        clear_output(wait=True)
        if not player:
            print("Pick a player first.")
            return
        if not val.isdigit():
            print("Enter a whole-dollar amount in 'Won @ $'.")
            return
        # Try to buy (will block if unaffordable)
        msg = opt.buy(player, int(val))
        print(msg)
        if msg.startswith("Blocked:"):
            return
        # Successful buy:
        best = opt.best_lineup()
        if best and best.get("ok") and show_lineups.value:
            grade_now = opt._grade(best["total_vor"])
            print(f"\nBest lineup now (Grade: {fmt_pct(grade_now)} of ideal):")
            display(sol_to_df(best))
    buy_price.value = ""
    budget_label.value = budget_html()
    refresh_all_dropdowns()
    dq.value = drb.value = dwr.value = dte.value = dk.value = ddef.value = ""
    update_recommendation()

def on_gone_clicked(_):
    player = get_selected_player()
    with output:
        clear_output(wait=True)
        if not player:
            print("Pick a player first.")
            return
        msg = opt.gone_player(player)
        print(msg)
    refresh_all_dropdowns()
    dq.value = drb.value = dwr.value = dte.value = dk.value = ddef.value = ""
    update_recommendation()

buy_btn.on_click(on_buy_clicked)
gone_btn.on_click(on_gone_clicked)

# Layout
row1 = w.HBox([hide_gone_owned, show_lineups, budget_label])
row2 = w.HBox([dq, drb, dwr])
row3 = w.HBox([dte, dk, ddef])
row4 = w.HBox([reco_label, caps_label, w.HTML("&nbsp;&nbsp;&nbsp;"), buy_price, buy_btn, gone_btn])
row5 = w.HBox([grade_label])  # NEW: grades row

refresh_all_dropdowns()
display(w.VBox([row1, row2, row3, row4, row5, output]))

print("Pick the nominated player from ONE position dropdown; others clear automatically.")
print("The 'Bid up to' is the lower of VOR-threshold and affordability.")
print("Grades compare lineups' VOR to your pre-draft ideal (best possible under $141 for these 6 slots).")
