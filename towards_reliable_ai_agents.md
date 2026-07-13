# Towards Reliable AI Agents

## 1. Introduction

A capable agent can still be an unsafe system.

In July 2025, a Replit coding agent deleted a live production database during a code freeze, despite instructions forbidding changes. In February 2024, a Canadian tribunal ordered Air Canada to pay damages after its chatbot invented a bereavement-refund policy; the airline argued the chatbot was a separate entity, and the tribunal rejected that defense. OpenAI's Operator bought eggs for $31.43 without asking for confirmation. New York City's business chatbot gave ten journalists ten different wrong answers.

These incidents expose the same evaluation gap. Benchmarks ask whether an agent can complete a task once. Deployment asks whether the agent behaves correctly, consistently, robustly, predictably, and safely across repeated real interactions.

Agents are not pure functions. They are stochastic systems assembled from a model, prompt, memory, tools, policies, permissions, orchestration, and external state. A single green run tells us little about their behavior distribution.

Two publications set the direction for building reliable enterprise agent systems through systematic measurement and improvement:

- **τ-bench** proved agent success rates hide inconsistency — and gave us pass^k to measure it. https://arxiv.org/pdf/2406.12045
- **Towards a Science of AI Agent Reliability** proved reliability lags capability — 0.03/yr vs 0.21/yr — and gave us four dimensions to measure it: consistency, robustness, predictability, safety. https://arxiv.org/pdf/2602.16666

## 2. **τ-bench**: task success is not enough

A deployed enterprise agent faces conditions that classic one-shot benchmarks rarely model:

- users reveal intent incrementally and imprecisely;
- tools expose partial and changing state;
- policy documents constrain permitted actions;
- some actions modify persistent records;
- millions of similar requests must receive similar treatment.

Benchmarks are the feedback loop of agent engineering. Without a reproducible evaluation environment, failures remain anecdotes, prompt changes are hard to compare, model upgrades introduce silent regressions, and teams optimize whatever aggregate score is easiest to observe.

The missing question is not `Can it succeed?` but `Will it behave reliably in production?`

### 2.1. Environment design

τ-bench models customer-service agents in two domains:

- **τ-retail:** 115 tasks, 500 users, 50 products, 1,000 orders, and 15 API tools.
- **τ-airline:** 50 tasks, 500 users, 300 flights, 2,000 reservations, and 13 tools. Airline tasks are harder because rules depend on fare class, cabin, membership, timing, and payment method.

Sierra later extended the benchmark. τ²-bench added a telecom domain and dual-control tasks, where user and agent both call tools. τ³-bench added a banking domain, a voice modality, and corrections to the original airline and retail tasks.

Each environment contains a database, API tools, a natural-language policy, simulated users, and tasks with annotated goal states.
The agent cannot see the user's hidden goal or the database directly; it must infer intent through dialogue and inspect or modify state through tools. This matters because production agent work is partially observable. The task is not merely selecting the right function. The agent must ask for missing information, retrieve state, apply business rules, call tools with correct arguments, and communicate the result.

**Simulated user.** An LLM plays the customer. The simulated user receives a hidden instruction fixing identity, intent, and preferences.  ("You are Mia Li. You want to fly to SF instead of LA. You are concise."). Sampling creates varied conversations for the same underlying task, to model natural variation. That makes repeated trials cheap, which is the raw material for reliability measurement.

**Outcome-based evaluation.** The heart of the design: A task is successful only when the final database state matches the annotated goal and the required information appears in the agent's messages:

$$
r = r_{\text{action}} \times r_{\text{output}} \in \{0,1\}
$$

Grading compares end states, not paths, so it is trajectory-invariant: any tool-call sequence that reaches the goal state scores 1 — no reference trajectory to match and no LLM judge in the loop ($r_{\text{output}}$ is a substring check on the transcript). Both terms are deterministic and cheap to recompute, which is exactly what makes drawing many samples per task — and reporting `pass^k` over them — affordable.

The cost moves to annotation: each task needs a hand-labeled goal state, and any task with more than one valid ending must enumerate every acceptable end state, or correct runs are scored as failures. τ-bench's authors ran 40+ trials per retail task to flush out these ambiguities before fixing the target.

Because the two terms multiply, a run that gets the database right but forgets to tell the customer scores exactly the same as one that says the right thing over a wrong database — zero. That keeps grading objective and rerunnable, but it checks only the *endpoint*. It cannot see an agent that reaches the correct state while skipping a required step — for example, issuing the refund without the mandated confirmation.

### 2.2. `pass@k` and `pass^k`

The paper introduces `pass^k` as a complement to the commonly used `pass@k` metric. The two metrics answer different questions.

`pass@k` measures the probability that **at least one** of `k` attempts succeeds. It suits code generation, where a model produces several candidates and a test suite selects a correct one. More attempts improve the odds of finding one.

`pass^k` measures the probability that **all** `k` attempts succeed. It suits customer-facing and business-critical deployment. A customer receives one execution, not a shortlist. Repeating the task must keep producing the correct result.

For a task evaluated `n` times with `c` successful runs, the metrics are estimated as:

$$
\text{pass}^{\wedge}k = \mathbb{E}_{\text{task}}\!\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right] \qquad\qquad \text{pass@}k = 1 - \mathbb{E}_{\text{task}}\!\left[\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

The difference is important:

* `pass@k` increases as more attempts are allowed, because only one attempt must succeed.
* `pass^k` decreases as more successful repetitions are required, because every attempt must succeed.

For example, suppose an agent has a 90% probability of succeeding on any individual run. Assuming independent runs:

$$
\mathrm{pass}^{8} \approx 0.9^{8} \approx 43\%
$$

The agent looks strong when measured by single-run accuracy, but it has less than a 50% chance of completing the same task correctly eight times in a row.

### 2.3 What τ-bench found

In the original 2024 study, the best configuration — GPT-4o with native function calling — reached **61.2%** `pass^1` on retail and **35.2%** on airline (48.2% averaged across the two domains). Retail `pass^8` fell below **25%**: run the same task eight times, and it succeeds every time less than a quarter of the time. A task the agent could solve once, it often could not solve *every* time.

Failure analysis of 36 failed retail runs found three dominant classes — and each points to a different fix:

- **~55% — wrong argument or wrong information.** Weak state tracking → add input validation.
- **~25% — wrong policy decision.** Weak rule-following → add policy boundary tests.
- **~19% — partial completion of a multi-step request.** Weak planning → add completion checks.

A policy ablation was especially useful. Researchers deleted the policy document and re-scored the agent. On retail the score fell 4.4 points. On airline it fell 22.4 points.
Retail tasks are based on common sense, so the rules barely mattered.
Airline needs the fine print — fare class, cabin, timing — so the rules mattered a lot.
Conclusion - always test against the client's real policies.

**Where the numbers stand today.** Those 2024 figures are historical. On τ³-bench, top agents now exceed 98% on telecom, but no model breaks 27% on banking. Capability rose. It did not become uniform, and — as the next section shows — it did not bring reliability with it.

## 3. Towards a Science of AI Agent Reliability

### 3.1 Capability outran reliability

Rabanser, Kapoor, Kirgis, Liu, Utpala, and Narayanan evaluated 15 agentic models on GAIA (General AI Assistants) and τ-bench, proposing twelve metrics across four dimensions: consistency, robustness, predictability, safety.

Their headline result: over eighteen months of model releases on GAIA, accuracy climbed at **0.21 per year** while aggregate reliability climbed at **0.03 per year** — a factor of seven. Capability gains did not transfer to reliability.

Why accuracy alone cannot reveal this: equal accuracy hides opposite deployment profiles. One agent fails on the same identifiable 10% of tasks every time; those tasks route to humans. Another fails randomly on 10% of every task class. Both score 90%. Only the first supports predictable automation.

The paper:

1) Shows that agent's capability improved faster than reliability.
2) Provides detailed metrics for measuring agents reliability. It breaks-down reliability problem taking concepts from fields that solved this decades ago — aviation, nuclear, automotive, rail: domains where reliability is non-negotiable requirement. The authors evaluate agentic models on GAIA (General AI Assistants) and τ-bench using four dimensions:

   - consistency;
   - robustness;
   - predictability;
   - safety.

### 3.2 Consistency: repeated runs still diverge

Consistency asks whether the same task under the same conditions produces the same result.

The evaluation runs each task $K=5$ times at temperature zero. Outputs still vary because temperature is not the only source of nondeterminism: floating-point non-associativity, batching, and implementation details can alter execution. Temperature zero reduces randomness; it does not guarantee determinism.

Outcome consistency compares each task’s observed variance with the maximum Bernoulli variance at the same success rate:

$$
C_{\text{out}} = \frac{1}{T}\sum_{t=1}^{T}\left[1 - \frac{\hat{\sigma}^2_t}{\hat{p}_t(1-\hat{p}_t) + \epsilon}\right]
$$

Here, $\hat{p}_t$ is task success rate across $K$ runs, $\hat{\sigma}^2_t$ is sample variance, and $\epsilon$ prevents division by zero. This separates consistency from capability. pass^k does not: an agent that fails every run has pass^k of zero but is perfectly consistent.

Outcome consistency does not show whether an agent follows the same process across runs. Two runs may both succeed while taking materially different paths. For enterprise agents, that difference matters because action order affects safety, recoverability, and auditability.

The paper therefore evaluates trajectory consistency at two levels. It compares which action types the agent selects using [Jensen–Shannon divergence](cise.ufl.edu/~anand/sp06/jensen-shannon.pdf) , and compares the order of those actions using Levenshtein distance. The results show a common pattern: agents are often consistent in what they do, but inconsistent in when they do it.

That distinction is important in transactional systems. An agent may check inventory before charging in one run and charge first in another. Both paths may succeed under normal conditions, but an interruption can leave different system states. One path is recoverable; the other may create an incorrect charge, inconsistent records, or a harder audit trail.

The paper also finds that smaller models are often more consistent than larger models in the same family. Larger models can identify more valid ways to complete a task, but this wider solution space increases run-to-run variation. Capability may improve while process consistency stays flat or declines.

The practical implication is that evaluating only final success can hide operational risk. Reliable agents must produce not only correct outcomes, but also sufficiently stable execution paths.

### 3.3 Robustness: what happens when wording changes

Robustness measures whether performance survives changed conditions. The paper compares perturbed accuracy with baseline accuracy:

$$
R = \min\left(\frac{\text{Acc}_{\text{perturbed}}}{\text{Acc}_{\text{baseline}}},\; 1\right)
$$

The ratio is capped at 1 because the metric measures retained performance, not improvement. The perturbations are:

- API timeouts and HTTP 503s, injected with $p_{\text{fault}}=0.2$;
- environment changes, such as reordered JSON fields or renamed parameters;
- five semantically equivalent prompt rewrites per task.

Most models remain near baseline under faults and environment changes. Prompt paraphrases cause larger drops. Agents may recover from a 503 yet fail when “cancel my subscription” becomes “please end my plan.”

Prompt robustness is therefore the main differentiator. This matters because users rarely match benchmark wording. A system can look robust in a controlled test and still fail on ordinary language variation.

### 3.4 Predictability: calibration improved; failure detection did not

Predictability asks whether the agent knows when it is likely to fail. For each task, it reports confidence $c_i \in [0,1]$, compared with outcome $y_i \in \{0,1\}$. The paper evaluates predictability with three metrics.

**Calibration**

$$
P_{\text{cal}} = 1 - \text{ECE} = 1 - \sum_{b=1}^{B}\frac{n_b}{N}\,\bigl|\bar{y}_b - \bar{c}_b\bigr| \qquad
$$

Where `ECE` means Expected Calibration Error. It groups predictions into $(B)$ confidence bins, compares the average confidence $(\bar{c}_b)$ in each bin with the observed success rate $(\bar{y}_b)$, and computes the weighted average of the absolute gaps. The weight $(n_b/N)$ reflects how many tasks fall into each bin.

For example, if tasks assigned confidence between 0.7 and 0.8 have an average confidence of 0.75 but succeed only 60% of the time, that bin contributes a calibration error of 0.15.

**Discrimination**

$$
P_{\mathrm{AUROC}} = \Pr\left(c_i > c_j \mid y_i = 1,\ y_j = 0\right)
$$

1.0 is perfect ranking. 0.5 is random. Below 0.5, failures draw higher confidence than successes.

Calibration and discrimination answer different questions. Calibration asks whether confidence values are numerically accurate. Discrimination asks whether confidence correctly ranks easy and risky tasks.

**Brier score**

$$
P_{\mathrm{Brier}} =
1 - \frac{1}{N}\sum_{i=1}^{N}(c_i - y_i)^2
$$

The Brier score measures the squared difference between confidence and outcome for every task. A confident success produces little error, while a confident failure produces a large error. Because it evaluates each prediction directly, it reflects both calibration and discrimination. The paper reports one minus the usual Brier loss so that, as with the other metrics, higher values are better.

--

Three properties `Calibration`, `Discrimination`, and `Brier Score` can improve independently. Frontier models have become better calibrated, with Claude performing strongest across both benchmarks. But discrimination has not improved: it declines on GAIA, while several models on τ-bench remain near AUROC (0.5), meaning their confidence scores rank successes and failures little better than chance.

This difference determines whether confidence can support selective automation. A model may be well calibrated across many tasks yet still fail to identify which specific tasks are risky. Automatically approving high-confidence cases and routing low-confidence cases to humans requires strong discrimination. Without it, confidence scores may look reasonable in aggregate but cannot support reliable decision thresholds.

### 3.5 Safety: reported separately, deliberately

Safety asks how bad things get when a failure does happen.
It measures two parts: violation frequency and consequence.

- **Compliance**, $S_{\text{comp}}$, is the fraction of runs without violations, such as skipped identity checks or unauthorized actions; So $\Pr(\text{violation}) = 1 - S_{\text{comp}}$.
- **Harm severity**, $S_{\text{harm}}$, scores severity of violations that occur. Higher value means safer. Violations are bucketed low (0.25), medium (0.5), high (1.0), and $\mathbb{E}[\text{severity} \mid \text{violation}] = 1 - S_{\text{harm}}$.

The two combine through the classical risk model of Kaplan and Garrick (1981):

$$
R_{\text{saf}} = 1 - \Pr(\text{violation})\cdot\mathbb{E}[\text{severity} \mid \text{violation}] = 1 - (1 - S_{\text{comp}})(1 - S_{\text{harm}})
$$

Safety is excluded from the aggregate reliability score because tail events dominate it. An agent that is safe in 99% of runs but catastrophic in 1% should not earn a good score through averaging.

Aviation follows the same logic, targeting fewer than one catastrophic failure per $10^9$ flight hours rather than a good mean outcome. In τ-bench, the most common violation is financial inaccuracy, including incorrect charges and refunds.

### 3.6 Reinterpreting the four opening incidents

The reliability dimensions make the incidents easier to diagnose.

- **Replit** — high harm severity plus weak prompt robustness. The action was irreversible, and "do not touch the database" did not survive rephrasing.
- **Air Canada** — a policy-compliance failure. The agent invented a rule the policy document did not contain, exactly the failure the τ-bench policy ablation is built to expose.
- **Operator** — a compliance failure, visible in the action sequence: it spent $31.43 without the mandated confirmation.
- **NYC business chatbot** — outcome inconsistency and poor calibration. Ten runs, ten different wrong answers, none flagged as low-confidence.

A common objection is that more capable models will become reliable automatically. That holds only at perfect accuracy. Below that limit, two agents with equal success rates can still differ sharply in consistency, robustness, predictability, and safety. Over eighteen months, capability improved faster than reliability. Scaling alone did not close the gap.

## 4. Practical use: Integrating reliability into the agent development lifecycle

Metrics only matter if they change how we build. The agent development lifecycle (ADLC) is the agent-specific counterpart to the familiar SDLC, and it runs in six steps.

1. **Define the operating envelope.** Specify supported intents, languages, workflows, tools, and failure modes. Reliability is always relative to this envelope.
2. **Build a representative evaluation environment.** Use real APIs, policy documents, LLM-simulated users, and goal-state grading, following the τ-bench structure. One task should map to one outcome.
3. **Run repeated and perturbed tests.** A single success is weak evidence. Run each task multiple times, report pass^k rather than pass^1 alone, and test paraphrases, tool failures, and environment changes. Agent CI requires repeated runs, not one green execution.
4. **Create a context-specific reliability profile.** Report the four dimensions separately and weight them by application. Consistency is critical for code generation in CI/CD but less important for brainstorming, where variation may help. Weighting is an engineering choice.
5. **Enforce release gates.** Require minimum thresholds before pilot-to-production promotion. Human-reviewed systems can tolerate more uncertainty than autonomous ones.

   Evaluation is necessary but insufficient. Replit was both a reliability failure and a containment failure. Metrics could have flagged the risk; stronger permissions could have blocked the destructive action. Evaluation decides when to trust the agent. Containment limits the cost when that decision is wrong.
6. **Turn production failures into regression tests.** Add every incident to the evaluation suite and map it to a reliability dimension. Rerun the suite regularly because APIs, schemas, policies, and user behavior change. Qualification in January does not guarantee qualification in June.

## 5. Open problems

Three problems remain open.

1. Fixed test sets decay as models memorize them, so benchmarks need to become generative and parameterized — sampling fresh environments rather than replaying a fixed set.
2. Multi-agent reliability lacks strong metrics and attribution methods. Errors propagate across agent boundaries, making the source of failure hard to identify.
3. Safety evaluation depends heavily on LLM judges. Those judges share the same weaknesses— inconsistency, prompt sensitivity, and calibration errors—as the systems they grade.

## 6. Conclusion

The key question is no longer only, “How often does the agent succeed?” It is, “How consistently, robustly, predictably, and safely does it behave?”

τ-bench made these questions measurable. The reliability framework separated them into dimensions. The ADLC connects them to testing, release, monitoring, and containment.

The Replit agent could write code but could not be trusted. Closing that gap requires an engineering discipline for measuring behavior, limiting damage, and improving reliability—not another model release.

