# Judge demo — 4 minutes

Open `http://localhost:8000`, full screen, dark room. Have
`http://localhost:8000/field` open on a phone or a second tab.

**For the live-data version**, start with `./run.sh --live --https` instead and
open `https://<your-ip>:8443/track` on the phone (accept the certificate
warning). Real weather + real GPS. If the venue wifi is bad, plain `./run.sh`
gives you the identical demo with simulated weather — nothing on stage depends
on the network.

**Do not narrate the tech stack.** Every beat below leads with the problem.

---

## 0:00 — The hook (20 s)

> "Eight states. Forty-five million people. And almost all of it depends on a
> handful of roads. When one slope fails in Meghalaya, Mizoram runs out of
> medicine — not metaphorically, it has happened.
>
> This is a live risk surface for that network."

Point at the map. Colour = predicted risk right now. Red dashed = predicted
blocked. The KPI strip reads how much of the network is cut.

---

## 0:20 — It explains itself (50 s)  ← **the strongest beat, do not rush it**

**Click any orange or red segment.** Right panel opens.

> "It says this road has a 62% chance of disruption. Fine — but nobody diverts a
> convoy because a model said 0.62.
>
> So it tells you *why*, and this is not a feature-importance chart. It is this
> specific prediction, decomposed exactly."

Read one driver out loud, verbatim:

> *"Ground is already saturated from 214 mm of antecedent rainfall — slopes fail
> on the next moderate burst, not the first one."*

Then point at the additivity line:

> "Baseline plus these contributions reconstructs the model's output to zero
> absolute error. It's verified in the response — you can check the explanation
> isn't decorative."

**If you say one more thing, say this:**

> "And the strongest driver the model found across the whole dataset — we didn't
> tell it this — is seven-day antecedent rainfall. It recovered the actual
> physics of Himalayan slope failure on its own: slopes don't fail because of
> today's rain, they fail because today's rain lands on ground already
> saturated."

---

## 1:10 — Routing that understands the trade (45 s)

Origin **Guwahati** → Destination **Aizawl**, profile **Emergency**, cargo
**Whole blood**. Click **Plan route**.

> "Three ranked alternates. It isn't picking the shortest — option 3 is shorter
> and it's ranked last, because its peak segment risk is nearly double."

Then the cargo box:

> "Blood has an eight-hour viability window. Predicted transit is far past it.
> So the system doesn't route the blood — it says road transport isn't viable
> and escalates to air lift. Knowing when *not* to dispatch is the useful answer."

Click **Compare profiles** for a pair like Guwahati → Dibrugarh:

> "Same origin and destination, four objectives. Safest accepts 124 km and two
> extra hours to cut peak risk by 30%. The trade is quantified, not implied."

---

## 1:55 — The thing nobody else built (55 s)

Tab: **Choke points**.

> "Every road-risk tool routes around today's problem. None of them tell you
> which roads the region cannot survive losing.
>
> We delete each segment from the graph, one at a time, and re-solve the whole
> network from Guwahati."

Point at row 1:

> "Number one is the Siliguri Corridor — the Chicken's Neck. If it fails,
> eleven districts and 1.2 million people lose their only road link. Number two
> is Kolasib to Aizawl: that's all of Mizoram, on one road. Number eight is
> NH-10 into Sikkim.
>
> We did not hard-code any of that. It fell out of the topology."

Toggle **Choke points** in the Layers box — pink markers appear on the map.

> "And this second ranking is criticality times *today's* failure probability.
> That's the watch-list a control room opens the morning with."

Tab: **Access** — one sentence:

> "Same engine, turned into a planning instrument: an accessibility index per
> district, every sub-score published. Mizoram scores worst in the region at 34
> out of 100, then Tripura, then Sikkim — the three states each hanging off a
> single arterial road. That's the number that argues for a bypass or a tunnel."

---

## 2:50 — Break it live (45 s)

Tab: **What-if** → click **Preset: Meghalaya cloudburst**.

Watch the map. Say nothing for three seconds. Then:

> "210 mm of rain, saturated ground, across Meghalaya and Assam.
>
> Blocked segments went from 3 to 35. Road length cut went from 236 km to 2,636.
> Bhalukpong drops from grade A to grade D.
>
> Nothing there is scripted. The models re-scored all 98 segments, the routes
> re-planned, the accessibility index recomputed. You can drag those sliders to
> anything you like."

---

## 3:25 — Live data (30 s)  ← **add this if you ran with `--live` / `--https`**

Hold up your phone showing `/track`.

> "Two things here are not simulated at all.
>
> The weather is live — real observations and forecasts for all 85 nodes,
> refreshed every 30 minutes. And this phone is a tracked vehicle: every GPS fix
> is snapped to the nearest highway and scored by the same model you've been
> looking at."

Walk a few steps. Point at the control-room map — the hexagon moves.

> "It's not on a separate tracking map. It's tracked *against the risk surface*.
> If I drive onto a segment the model calls blocked, I get the alert before the
> control room has to call me — and it warns about what's ahead, not just what's
> under the wheels."

---

## 3:35 — Close the loop (25 s)

Show the phone at `/field`.

> "The other half of the problem is that the data has to come from people
> standing on the road — and there's no signal exactly where roads fail."

Turn off wifi. Submit a report.

> "Saved to the device. It queues in IndexedDB and replays when a signal
> returns, with a client ID so a retry can't duplicate it."

Turn wifi on, hit **Sync now**.

> "It snaps to the nearest segment and feeds the incident-memory features on the
> next inference. A slope that has failed is more likely to fail again — and the
> model already knows that, because it's the second-strongest driver it found."

---

## 4:00 — One line to finish

> "Real network, honest models, and it runs with the wifi off — because so does
> the North East."

---

# Questions you will get

**"Is this real data?"**
> The network is — 85 real towns, real coordinates, real National Highway
> alignments, 6,858 km. You can check it against any map. Weather and failure
> labels are simulated from real NER climatology and a physical hazard model the
> ML never sees. The machine learning is real: 44,000 observations, held-out
> temporal split. Swapping in IMD and Bhuvan feeds is a table swap, not a
> rewrite.

**"82% accuracy isn't very high."**
> That's the *temporal* split — trained on the past, tested on the future, which
> is how it would actually run. On a random split it's 89.6%. Most projects quote
> the random number. The one that matters operationally is 87% recall on
> `blocked`, because a missed closure sends a convoy into a landslide and a false
> alarm costs a phone call. We weight the classes to protect that.

**"Why not XGBoost / a neural net?"**
> We trained a gradient-boosting challenger; it's about 3 points better and
> it's in the metrics screen. We ship the RandomForest deliberately, because it
> supports exact additive attribution and the boosted model doesn't. For a tool
> a district officer signs off on, that trade is worth more than a point. A
> neural net on 44,000 tabular rows would be worse *and* opaque.

**"Is that really SHAP?"**
> It's the Saabas decision-path method — the same family, exact and additive,
> differing from TreeSHAP in how credit is split when features interact. We say
> so in the code and the README, and we return the additivity residual so you can
> verify it. `shap` wasn't installable in our environment, so we implemented the
> attribution directly rather than shipping a fake bar chart.

**"Why is route delay not just the sum of segment delays?"**
> Because queues cascade, a blocked segment forces a detour and a re-plan, every
> inter-state checkpost adds fixed dwell, and hill sections effectively close
> after dark. The UI shows the naive sum next to the model's answer so you can
> see the gap. Training that second model correctly is also where we hit our
> nastiest bug — ask me about the cascade.

**"What was the hardest bug?"** (a good one to volunteer)
> Train/serve skew in the cascade. The route model was trained on *true* hazard
> but served the classifier's *predicted* risk, and predicted probability
> saturates where true hazard doesn't. A route whose segments summed to 3 hours
> was being predicted at 27. The fix was to retrain stage 2 on stage 1's own
> outputs, so the training distribution matches production.

**"All four profiles gave me the same road."**
> That's the finding, not a bug — and the UI says so. There is no alternative
> corridor to that destination, so safety cannot be traded against time. That is
> the actual condition of most of the North East, and it's what the choke-point
> and accessibility screens exist to quantify.

**"Those local-language alerts — are they correct?"**
> English, Hindi, Assamese, Bengali and Nepali are reliable. Mizo, Khasi and
> Meitei are marked `review_pending` in the API and flagged in the UI, because
> they were written without a native speaker. Shipping unreviewed text in a
> safety-critical alert is a real harm. The architecture is the deliverable;
> verified translation is a field-partner task and we'd rather flag it than fake
> it.

**"Is the weather real?"**
> It is if we launched with `--live` — Open-Meteo, no API key, real observations
> and forecasts for all 85 nodes, refreshed every 30 minutes. The hard part
> wasn't the HTTP call, it was feature parity: the 7-day antecedent index has to
> be computed with the same decay constant the models trained on, and there's a
> test asserting the two implementations agree. We also take the max of the
> reported weather code and the 24-hour accumulation, because Open-Meteo will
> happily label a segment "light drizzle" when it has taken 120 mm that day.

**"What happens if the weather API goes down mid-demo?"**
> Nothing visible. The refresh logs a failure, returns `ok: false`, and the last
> stored weather keeps serving. We made that explicit because a dashboard that
> dies when a third-party API hiccups is not a dashboard anyone would deploy.

**"Is that really GPS, or a simulation?"**
> Real. That's my phone's `watchPosition` posting to `/api/track`. Watch — [walk
> a few steps, the hexagon moves]. It snaps to the nearest highway segment to
> within a few metres and gets scored by the live model. The endpoint takes
> plain JSON, so an AIS-140 or VTS unit pushes into the same path with no code
> change; the phone is just the cheapest tracker.

**"What would you do with another week?"**
> Wire the real feeds — IMD nowcast, Bhuvan landslide inventory, state PWD
> closure notices, VAHAN for real fleet positions — and retrain on actual closure
> records. The pipeline doesn't change. Then push the alerts over SMS and IVR,
> because a control-room dashboard doesn't help a driver at Sela Pass.

---

# If something breaks on stage

- **Map is blank** → hard-refresh (Ctrl-Shift-R). The SVG renders client-side;
  no tiles to fail.
- **"Backend not ready"** → `python3 scripts/pipeline.py`, then `./run.sh`.
- **Fleet not moving** → the Fleet tab has a Pause/Resume button; the SSE stream
  reconnects on its own.
- **What-if left the map red** → What-if tab → **Reset to live**.
- **Everything is slow** → the departure sweep re-scores the network 24 times;
  it takes ~4 s and shows a spinner. That's the honest cost of the answer.
