import math
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from ortools.linear_solver import pywraplp

st.set_page_config(page_title="Auction Draft Optimizer", layout="wide")

# -------------------- Roster / Rules --------------------
SLOTS = ["QB", "WR", "TE", "FLEX", "DEF", "K"]  # RB only eligible for FLEX (you already keep 2 RB + 1 WR)
DEFAULT_BUDGET = 141

def eligible_slots_for_position(pos: str):
    pos = str(pos).upper()
    if pos == "QB":  return ["QB"]
    if pos == "WR":  return ["WR", "FLEX"]
    if pos == "TE":  return ["TE", "FLEX"]
    if pos == "RB":  return ["FLEX"]
    if pos == "K":   return ["K"]
    if pos == "DEF": return ["DEF"]
    return []

def norm_pos(p):
    p = str(p).strip().upper()
    mapping = {"QB":"QB","RB":"RB","WR":"WR","TE":"TE","K":"K","PK":"K","DST":"DEF","D/ST":"DEF","DEF":"DEF"}
    return mapping.get(p, p)

# -------------------- Optimizer --------------------
class DraftOptimizer:
    def __init__(self, df: pd.DataFrame, budget=DEFAULT_BUDGET):
        self.df_all = df.copy()
        self.budget_start = int(budget)
        self.budget_left = int(budget)
        self.owned = {}      # name -> price
        self.gone = set()
        self.name_to_rows = {}
        for idx, row in self.df_all.iterrows():
            self.name_to_rows.setdefault(row["Player"], []).append(int(idx))

        # Precompute ideal (max VOR under full 6-slot cap, no owned/gone)
        ideal = self._solve_maxvor_tiebreak_minspend_internal(
            candidates=self.df_all.copy().reset_index(drop=True),
            force_owned=False, forced_player=None, forced_price=None
        )
        self.ideal_vor = ideal["total_vor"] if ideal.get("ok") else None

    # ---- Helpers ----
    def _match_name(self, name: str) -> str:
        if name in self.name_to_rows:
            return name
        choices = list(self.name_to_rows.keys())
        match, score, _ = process.extractOne(name, choices, scorer=fuzz.WRatio)
        if score >= 85:
            return match
        raise ValueError(f"Could not confidently match '{name}'. Closest was '{match}' ({score}).")

    def gone_player(self, name: str):
        n = self._match_name(name)
        if n in self.owned:
            return f"{n} already owned; not marking gone."
        self.gone.add(n)
        return f"Marked gone: {n}"

    def _candidates(self):
        return self.df_all[~self.df_all["Player"].isin(self.gone)].copy().reset_index(drop=True)

    def _build_solver(self, candidates):
        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            raise RuntimeError("Failed to create CBC solver.")
        x = {}
        for i, r in candidates.iterrows():
            for slot in eligible_slots_for_position(r["Position"]):
                x[(i, slot)] = solver.BoolVar(f"x_{i}_{slot}")
        # slot fill
        for slot in SLOTS:
            solver.Add(sum(x[(i, slot)] for i, r in candidates.iterrows() if (i, slot) in x) == 1)
        # player at most once
        for i, r in candidates.iterrows():
            solver.Add(sum(x[(i, slot)] for slot in SLOTS if (i, slot) in x) <= 1)
        return solver, x

    def _force_owned_and_nominee(self, solver, x, candidates, forced_player=None, include_owned=True):
        name_to_idx = {}
        for i, r in candidates.iterrows():
            name_to_idx.setdefault(r["Player"], []).append(i)
        # force owned
        if include_owned:
            for name in self.owned.keys():
                if name in name_to_idx:
                    idxs = name_to_idx[name]
                    solver.Add(sum(x[(i, s)] for i in idxs for s in SLOTS if (i, s) in x) == 1)
        # force nominee
        forced_index = None
        if forced_player is not None:
            fname = self._match_name(forced_player)
            if fname not in name_to_idx:
                return None, f"{fname} not in candidates."
            forced_index = name_to_idx[fname][0]
            solver.Add(sum(x[(forced_index, s)] for s in SLOTS if (forced_index, s) in x) == 1)
        return forced_index, None

    def _price_used(self, i, row, forced_index, forced_price):
        n = row["Player"]
        if forced_index is not None and i == forced_index and forced_price is not None:
            return float(forced_price)
        if n in self.owned:
            return float(self.owned[n])
        return float(row["AAV"])

    def _solve_maxvor_tiebreak_minspend_internal(self, candidates, force_owned=True, forced_player=None, forced_price=None):
        if candidates.empty: return {"ok": False, "reason": "No candidates"}
        solver, x = self._build_solver(candidates)
        forced_index, err = self._force_owned_and_nominee(solver, x, candidates, forced_player, include_owned=force_owned)
        if err: return {"ok": False, "reason": err}

        total_spend = solver.Sum(self._price_used(i, candidates.loc[i], forced_index, forced_price) * x[(i, s)]
                                 for (i, s) in x)
        solver.Add(total_spend <= float(self.budget_start))
        total_vor = solver.Sum(float(candidates.loc[i, "VOR"]) * x[(i, s)] for (i, s) in x)

        # Phase 1: max VOR
        solver.Maximize(total_vor)
        if solver.Solve() != pywraplp.Solver.OPTIMAL:
            return {"ok": False, "reason": "No optimal VOR solution"}
        best_vor = total_vor.solution_value()

        # Phase 2: min spend among max-VOR
        tiny = 1e-6
        solver.Add(total_vor >= best_vor - tiny)
        solver.Add(total_vor <= best_vor + tiny)
        solver.Minimize(total_spend)
        if solver.Solve() != pywraplp.Solver.OPTIMAL:
            return {"ok": False, "reason": "No min-spend tie-break"}

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

    def _solve_maxvor_tiebreak_minspend(self, forced_player=None, forced_price=None):
        return self._solve_maxvor_tiebreak_minspend_internal(
            candidates=self._candidates(), force_owned=True,
            forced_player=forced_player, forced_price=forced_price
        )

    def _solve_min_spend_any_vor(self, forced_player=None, forced_price=None):
        c = self._candidates()
        if c.empty: return {"ok": False}
        solver, x = self._build_solver(c)
        forced_index, err = self._force_owned_and_nominee(solver, x, c, forced_player, include_owned=True)
        if err: return {"ok": False, "reason": err}
        total_spend = solver.Sum(self._price_used(i, c.loc[i], forced_index, forced_price) * x[(i, s)]
                                 for (i, s) in x)
        solver.Add(total_spend <= float(self.budget_start))
        solver.Minimize(total_spend)
        if solver.Solve() != pywraplp.Solver.OPTIMAL:
            return {"ok": False}
        chosen, tot_sp, owned_sp = [], 0.0, 0.0
        for (i, slot), var in x.items():
            if var.solution_value() > 0.5:
                row = c.loc[i]
                p = self._price_used(i, row, forced_index, forced_price)
                tot_sp += p
                if row["Player"] in self.owned:
                    owned_sp += p
                chosen.append({
                    "slot": slot, "name": row["Player"], "pos": row["Position"],
                    "team": row.get("Team", ""), "vor": float(row["VOR"]),
                    "aav": float(row["AAV"]), "price_used": p
                })
        return {"ok": True, "total_spend": tot_sp, "owned_spend": owned_sp,
                "future_spend_needed": max(0.0, tot_sp - owned_sp), "lineup": chosen}

    def _grade(self, vor):
        if self.ideal_vor and self.ideal_vor > 0:
            return 100.0 * float(vor) / float(self.ideal_vor)
        return None

    def compute_threshold(self, name):
        n = self._match_name(name)
        base = self._solve_maxvor_tiebreak_minspend()
        if not base.get("ok"): return {"ok": False, "reason": base.get("reason")}
        base_vor = base["total_vor"]

        trial = self._solve_maxvor_tiebreak_minspend(forced_player=n, forced_price=1)
        if (not trial.get("ok")) or (trial["total_vor"] + 1e-6 < base_vor):
            vor_cap, with_lineup = 0, None
        else:
            lo, hi = 1, int(self.budget_left)
            vor_cap, with_lineup = lo, trial
            while lo <= hi:
                mid = (lo + hi) // 2
                t = self._solve_maxvor_tiebreak_minspend(forced_player=n, forced_price=mid)
                if t.get("ok") and t["total_vor"] + 1e-6 >= base_vor:
                    vor_cap, with_lineup = mid, t
                    lo = mid + 1
                else:
                    hi = mid - 1

        minfill = self._solve_min_spend_any_vor(forced_player=n, forced_price=1)
        if not minfill.get("ok"):
            afford_cap = 0
        else:
            need = math.ceil(minfill["future_spend_needed"])
            afford_cap = max(0, int(self.budget_left - need))

        final_cap = min(vor_cap, afford_cap)
        return {
            "ok": True, "player": n, "max_bid": final_cap,
            "vor_cap": vor_cap, "afford_cap": afford_cap,
            "baseline": base, "with_lineup": with_lineup,
            "baseline_grade": self._grade(base["total_vor"]),
            "with_grade": self._grade(with_lineup["total_vor"]) if with_lineup else None,
            "ideal_vor": self.ideal_vor
        }

    def zero_chance_players(self):
        base = self._solve_maxvor_tiebreak_minspend()
        if not base.get("ok"): return []
        base_vor = base["total_vor"]
        tiny = 1e-6
        cands = self._candidates()
        names = sorted(cands["Player"].unique().tolist())
        zero = []
        for n in names:
            t = self._solve_maxvor_tiebreak_minspend(forced_player=n, forced_price=1)
            if (not t.get("ok")) or (t["total_vor"] + tiny < base_vor):
                zero.append(n)
        return zero

    def buy(self, name, price):
        n = self._match_name(name)
        price = int(price)
        minfill = self._solve_min_spend_any_vor(forced_player=n, forced_price=1)
        if not minfill.get("ok"):
            return f"Cannot buy {n}: no feasible completion."
        need = math.ceil(minfill["future_spend_needed"])
        max_aff = max(0, int(self.budget_left - need))
        if price > max_aff:
            return (f"Blocked: ${price} leaves ${self.budget_left - price}, "
                    f"need at least ${need} to fill remaining slots. Max: ${max_aff}.")
        if n in self.gone: self.gone.remove(n)
        prev = self.owned.get(n)
        if prev is not None: self.budget_left += prev
        self.owned[n] = price
        self.budget_left -= price
        return f"Added {n} for ${price}. Budget left: ${self.budget_left}"

# -------------------- UI --------------------
st.title("Auction Draft Optimizer (VOR)")

st.sidebar.header("1) Upload CSV")
uploaded = st.sidebar.file_uploader("Columns: Position, Player, Team, Points, VOR, AAV", type=["csv"])
budget = st.sidebar.number_input("Budget for these 6 slots", min_value=1, value=DEFAULT_BUDGET, step=1)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    # Basic validation/clean
    for c in ["Points","VOR","AAV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["VOR","AAV"]).copy()
    df["Position"] = df["Position"].map(norm_pos)
    df = df[df["Position"].isin({"QB","RB","WR","TE","K","DEF"})].reset_index(drop=True)

    # Create optimizer once per new file/budget
    if "opt" not in st.session_state or st.session_state.get("last_budget") != budget or st.session_state.get("last_file") != uploaded.name:
        st.session_state.opt = DraftOptimizer(df, budget=budget)
        st.session_state.last_budget = budget
        st.session_state.last_file = uploaded.name

    opt: DraftOptimizer = st.session_state.opt

    st.sidebar.header("2) Controls")
    hide_gone_owned = st.sidebar.checkbox("Hide owned/gone in pickers", value=True)
    show_details = st.sidebar.checkbox("Show lineups & grades", value=False)

    # Per-position pickers (with auto-clear)
    def options_for_pos(pos):
        d = df[df["Position"] == pos]
        if hide_gone_owned:
            d = d[~d["Player"].isin(opt.gone)]
            d = d[~d["Player"].isin(opt.owned.keys())]
        names = ["— Select —"] + sorted(d["Player"].unique().tolist())
        return names

    def clear_others(keep_key):
        for key in ["sel_QB","sel_RB","sel_WR","sel_TE","sel_K","sel_DEF"]:
            if key != keep_key:
                st.session_state[key] = "— Select —"

    cols1 = st.columns(3)
    cols2 = st.columns(3)

    sel_QB = cols1[0].selectbox("QB", options_for_pos("QB"), key="sel_QB", index=0,
                                on_change=lambda: clear_others("sel_QB"))
    sel_RB = cols1[1].selectbox("RB (FLEX)", options_for_pos("RB"), key="sel_RB", index=0,
                                on_change=lambda: clear_others("sel_RB"))
    sel_WR = cols1[2].selectbox("WR", options_for_pos("WR"), key="sel_WR", index=0,
                                on_change=lambda: clear_others("sel_WR"))
    sel_TE = cols2[0].selectbox("TE", options_for_pos("TE"), key="sel_TE", index=0,
                                on_change=lambda: clear_others("sel_TE"))
    sel_K  = cols2[1].selectbox("K",  options_for_pos("K"),  key="sel_K", index=0,
                                on_change=lambda: clear_others("sel_K"))
    sel_D  = cols2[2].selectbox("DEF", options_for_pos("DEF"), key="sel_DEF", index=0,
                                on_change=lambda: clear_others("sel_DEF"))

    def get_selected_player():
        for key in ["sel_QB","sel_RB","sel_WR","sel_TE","sel_K","sel_DEF"]:
            val = st.session_state.get(key, "— Select —")
            if val and val != "— Select —":
                return val
        return None

    st.markdown(f"**Budget left:** ${opt.budget_left} &nbsp;&nbsp;|&nbsp;&nbsp; **Total cap (6 slots):** ${opt.budget_start}")

    nominee = get_selected_player()
    reco_col, caps_col, grade_col = st.columns([1,1,2])

    if nominee:
        res = opt.compute_threshold(nominee)
        if not res.get("ok"):
            reco_col.error("No solution")
        else:
            reco_col.metric("Bid up to", f"${res['max_bid']}")
            if show_details:
                caps_col.write(f"VOR cap: **${res['vor_cap']}**  |  Afford cap: **${res['afford_cap']}**")
                def fmt(x): return "—" if x is None else f"{x:.1f}%"
                grade_col.write(f"**Grades:** Fallback {fmt(res['baseline_grade'])} | With {fmt(res['with_grade'])} | Ideal 100%")

            if show_details:
                st.subheader("Fallback lineup (without nominee)")
                fb = pd.DataFrame(res["baseline"]["lineup"])
                st.dataframe(fb[["slot","name","pos","team","vor","price_used","aav"]].rename(columns={
                    "slot":"Slot","name":"Player","pos":"Pos","team":"Team","vor":"VOR","price_used":"PriceUsed","aav":"AAV"
                }), use_container_width=True)

                if res["with_lineup"]:
                    st.subheader(f"Best lineup WITH {res['player']} at ${res['max_bid']}")
                    wl = pd.DataFrame(res["with_lineup"]["lineup"])
                    st.dataframe(wl[["slot","name","pos","team","vor","price_used","aav"]].rename(columns={
                        "slot":"Slot","name":"Player","pos":"Pos","team":"Team","vor":"VOR","price_used":"PriceUsed","aav":"AAV"
                    }), use_container_width=True)

    st.divider()
    st.subheader("Buy / Gone")
    c1, c2, c3 = st.columns([1,1,2])
    win_price = c1.number_input("Won @ $", min_value=1, step=1, value=1)
    if c2.button("Buy"):
        if not nominee:
            st.warning("Pick a player first.")
        else:
            msg = opt.buy(nominee, int(win_price))
            st.write(msg)
    if c3.button("Gone"):
        if not nominee:
            st.warning("Pick a player first.")
        else:
            st.write(opt.gone_player(nominee))

    if show_details:
        st.subheader("Best lineup now")
        best = opt._solve_maxvor_tiebreak_minspend()
        if best.get("ok"):
            now = pd.DataFrame(best["lineup"])
            st.dataframe(now[["slot","name","pos","team","vor","price_used","aav"]].rename(columns={
                "slot":"Slot","name":"Player","pos":"Pos","team":"Team","vor":"VOR","price_used":"PriceUsed","aav":"AAV"
            }), use_container_width=True)
            grade_now = opt._grade(best["total_vor"])
            st.caption(f"Grade now: {('—' if grade_now is None else f'{grade_now:.1f}%')} of ideal")

    st.divider()
    st.subheader("Zero-chance players (cannot be in any max-VOR roster even at $1)")
    if st.button("Recompute zero-chance"):
        pass  # no-op; forces rerun

    zero = opt.zero_chance_players()
    zero_df = (df[df["Player"].isin(zero)][["Player","Position","Team","VOR","AAV"]]
               .sort_values(["Position","VOR"], ascending=[True, False]).reset_index(drop=True))
    st.write(f"Count: **{len(zero_df)}**")
    st.dataframe(zero_df, use_container_width=True)

else:
    st.info("Upload your CSV to begin.")
