# Master's Thesis Defense: Presentation Plan

## Scope

- Title: **Machine Learning for Satellite Link Fade Nowcasting**
- Speaker: Giuseppe Gabriele Russo
- Supervisor: Davide Bacciu
- Co-supervisor: Giovanni Scognamiglio
- Language: English for both slides and oral presentation.
- Audience assumption: a computer science committee with general machine-learning knowledge, but no specialist satellite-communications background.
- Format: 17 main slides including the closing screen, approximately 19 minutes 20 seconds including transitions and time to explain figures. Reserve the remaining 40 seconds for pauses. Ten optional backup slides are outside this allocation.
- Status: implemented as `defense.tex`, a 16:9 LaTeX Beamer deck. Speaker notes below accompany the slides; compilation instructions are in the local `README.md`.

**Central message:** The most accurate predictor under a task-specific metric is not necessarily the most useful switch policy. Target definition, output conversion, and decision coverage must be evaluated together.

## Running Order

| Slide | Title | Time | Cumulative |
| --- | --- | --- | --- |
| 1 | Machine Learning for Satellite Link Fade Nowcasting | 0:20 | 0:20 |
| 2 | Industrial Context and Operational Problem | 1:20 | 1:40 |
| 3 | Research Objective and Contribution | 1:00 | 2:40 |
| 4 | Data and Experimental Setting | 1:10 | 3:50 |
| 5 | Four Predictive Formulations | 1:20 | 5:10 |
| 6 | Perfect Switch and Evaluation | 0:55 | 6:05 |
| 7 | Shared Switch Post-Processing | 1:05 | 7:10 |
| 8 | Autoregressive Forecasting | 1:40 | 8:50 |
| 9 | Current-Level Persistence | 1:25 | 10:15 |
| 10 | Long-Fade Detection | 1:20 | 11:35 |
| 11 | Survival Persistence | 1:35 | 13:10 |
| 12 | Cross-Task Switch Performance | 1:20 | 14:30 |
| 13 | Active Duration and Switching Events | 1:40 | 16:10 |
| 14 | An Event-Level Comparison | 1:20 | 17:30 |
| 15 | Limitations and Next Steps | 0:55 | 18:25 |
| 16 | Conclusions | 0:50 | 19:15 |
| 17 | Thank You / Questions | 0:05 | 19:20 |

## Slide 1. Machine Learning for Satellite Link Fade Nowcasting

**On screen:** thesis title, candidate, supervisors, University of Pisa, Master's Degree in Computer Science, Curriculum: Artificial Intelligence. Add the short subtitle: "Experimental thesis carried out at M.B.I. S.r.l."

**Visual:** University logo from `../figures/cherubino_pant541.png` and company logo from `mbi.png`. Subsequent slides show only the small UniPi logo in the footer.

**Speaker notes:**

Good morning. Today I will present my experimental master's thesis, carried out at M.B.I. S.r.l. The work investigates how machine learning can use satellite signal measurements to support decisions about switching to a backup communication link during a fade.

**Transition:** I will begin with the application context and the operational problem behind the experiments.

## Slide 2. Industrial Context and Operational Problem

**On screen:**

- Industrial setting: satellite-signal analysis at M.B.I. S.r.l.
- Operational need: distinguish brief fades from persistent link degradation.
- Experimental objective: assess when signal history supports a backup-link decision.

**Visual:** a simple link/backup diagram and one signal trace with the current time and the operational threshold. Show the future as unavailable at decision time. Use a short spike and a sustained fade to explain the distinction. Derive the trace from an existing test event, rather than inventing experimental data.

**Speaker notes:**

The application context is satellite communications. The study uses real fade and beacon recordings acquired at the Fucino ground station and examines a practical decision: when the satellite link degrades, is the fade likely to persist long enough to justify activating a backup link?

Switching too late can leave the link exposed to degradation. Reacting to every short fluctuation can instead cause unnecessary backup activation. The goal is therefore to anticipate persistence, rather than simply detect that a threshold has been crossed.

This is an experimental study of that decision, using historical data and an offline reference, not a deployed controller. The evaluation uses a ten-decibel threshold and a five-minute persistence horizon. At each decision time, the predictive models receive recent signal history, not the future observations used to assess their decisions.

**Transition:** This motivates a comparison of different ways to formulate the prediction problem.

## Slide 3. Research Objective and Contribution

**On screen:**

> Which predictive formulation supports useful switch decisions from signal history alone?

- Four task formulations with different targets.
- Established models evaluated against a shared reference.
- Prediction quality and operational behavior assessed separately.

**Visual:** a compact view of the common input and final switch decision, with the alternative predictive outputs between them. Avoid an architecture catalogue here.

**Speaker notes:**

The contribution is a decision-level comparison of four formulations. We can predict a future trajectory, estimate a duration, classify an event, or estimate the probability that the remaining event duration exceeds a given horizon.

Each formulation has its own natural training objective. However, all of them eventually need to support the same operational action. I therefore compare both their native predictive performance and the switch sequences obtained from their outputs. An important part of the framework is making the evaluation domain explicit: the tasks do not necessarily produce decisions at the same timestamps. Without accounting for that difference, their scores would not be directly comparable.

**Transition:** The comparison starts from a common collection of signal measurements.

## Slide 4. Data and Experimental Setting

**On screen:**

- Satellite fade and beacon recordings from Fucino.
- Signal-only inputs, 30-second sampling, 30-observation context.
- Sparse fades amid long near-baseline stretches motivate event-focused sampling instead of uniformly sampling the full recordings.
- Forecasting/current-level: candidate-event windows extending 90 minutes before and after onset. Long-fade/survival: timestamps inside task-defined events.
- External test source: `fc-uplink-fade.csv`.
- Chronological development splits and validation-based model selection.

**Visual:** two columns: rarity and task-specific sample selection on the left, development and external-test splits on the right. Shared sampling settings occupy one short line. Detailed gap handling remains in backup B5.

**Speaker notes:**

The recordings mostly contain quiet, near-baseline conditions; fades are sparse. Uniformly sampling the full series would risk making training focus on reproducing these long plateaus rather than the changes relevant to switching. We therefore select samples around fade episodes.

Forecasting and current-level persistence use windows extending ninety minutes before and after candidate onset. These include nearby background and recovery, not only above-threshold samples. Long-fade and survival instead sample inside their task-specific events.

Inputs use signal only, normally thirty observations at thirty-second sampling. One complete source is held out; development splits are chronological and event-aware, with model selection on validation. The resulting evaluation is event-focused, not a test of continuous monitoring over all raw recordings.

**Methodological clarification for questions:** plateau dominance is the sampling rationale, not a demonstrated full-series versus event-window ablation. Cropping selects eligible prediction timestamps; it does not join distant observations or allow contexts to cross acquisition gaps. Current-level duration labels are searched on the full continuous signal, not truncated at the candidate-window boundary. Timestamp validation and conservative small-gap handling still apply (B5/B6). Chronos is zero-shot; the survival switch threshold has the explicitly stated exploratory caveat.

**Transition:** The inputs are related, but the four targets answer different questions.

## Slide 5. Four Predictive Formulations

**On screen:**

| Task | Prediction |
| --- | --- |
| Autoregressive forecasting | The next ten signal values |
| Current-level persistence | Time until the signal falls below its current level minus delta |
| Long-fade detection | Probability that the entire grouped event lasts at least five minutes |
| Survival persistence | Probability that the remaining fade duration exceeds a future horizon |

**Visual:** four small output illustrations with the same past-window convention. Explicitly distinguish total event duration from remaining duration. Use a shared color per task throughout the deck.

**Speaker notes:**

Autoregressive forecasting predicts the next ten observations, covering five minutes. Current-level persistence predicts the time until the signal falls below a level defined relative to its current value, with delta equal to half a decibel.

Long-fade detection asks whether the entire grouped event lasts at least five minutes. This is an event-classification question, not a prediction of how much time is left. Survival persistence addresses that remaining-time question directly by estimating a probability distribution.

These distinctions matter. A decline below the current level does not necessarily mean that a fade has ended. Similarly, knowing that an event is long does not tell us exactly when to turn the backup link off. The different targets therefore imply different decision rules.

**Transition:** To assess those rules, we need an explicit reference and a common evaluation procedure.

## Slide 6. Perfect Switch and Evaluation

**On screen:**

- Perfect Switch: an offline reference based on the observed signal.
- Precision and recall: unnecessary activation versus missed reference activity.
- Active duration and switching intervals: time spent on backup and fragmentation.

**Visual:** signal and threshold above a Perfect Switch row and a final model-switch row. Highlight one overlap, one false-positive interval, and one missed interval. Label this as an illustrative example. Keep intermediate post-processing rows for slide 7.

**Speaker notes:**

Perfect Switch is an offline reference constructed using the true signal evolution. It identifies persistent threshold intervals and is not available to a real-time predictor. The models are evaluated according to how closely their decisions agree with this reference.

Precision tells us how much predicted switch activity agrees with the reference. Recall tells us how much reference activity is recovered. We also measure total active time and the number of separate switch intervals. These capture operational differences that a single F1 score can hide.

**Transition:** Before measuring these quantities, we apply the same post-processing to the model-derived switch decisions.

## Slide 7. Shared Switch Post-Processing

**On screen:**

- Raw switch decision: a binary request derived from the model output.
- Minimum-duration extension: extend short active runs to ten samples.
- Hold during fade: keep an active switch on while the current signal is at least 10 dB.

**Visual:** four vertically aligned rows: observed signal with its threshold, raw binary decision, decision after minimum-island extension, and final held decision. Reveal the binary rows in order. Use distinct annotations for samples added by extension and samples added by holding. This modifies decisions, not the measured signal or model predictions.

**Illustrative data for the future figure:** use 15 consecutive samples at 30-second spacing. The signal is 9 dB at samples 1-2 and 15, and 11 dB at samples 3-14. The raw switch is one only at samples 3-4. Minimum-island processing extends it through sample 12, giving ten active samples. Holding adds samples 13-14. At sample 15, the final switch returns to zero because the signal is below threshold and the extended decision is zero. These are schematic values, not experimental measurements.

**Speaker notes:**

This post-processing acts on the binary switch decision, not on the measured signal. First, each task converts its prediction into an immediate switch request. The conversion differs between tasks, but the following steps are shared.

In this example, the initial request lasts only two samples. Minimum-duration processing extends that run to ten samples. It extends short activations rather than removing them.

The holding rule then checks the current observed signal. If the switch was active at the previous sample and the signal is still at or above ten decibels, it remains active even when the immediate request is zero. Once the minimum extension is complete, recovery below the threshold allows it to turn off. All subsequent switch results refer to this final sequence.

**Transition:** I will now explain how each predictive formulation produces the initial switch request.

## Slide 8. Autoregressive Forecasting

**On screen:**

- Models: GRU S2V, GRU encoder-decoder, PatchTST, XGBoost, Chronos zero-shot.
- Main setup: 30 past values to ten future values; raw and context-standardized trainable variants, raw zero-shot Chronos.
- Immediate switch: all ten predicted signal values are at least 10 dB.
- Lowest test RMSE: GRU S2V raw, 2.852 dB.
- Highest native switch F1: PatchTST raw, 0.808.

**Visual:** two columns: input/output size, representation variants and initial switch rule on the left; a two-model RMSE/F1 comparison on the right. Architecture names occupy one short line. Exact normalization and the additional Chronos L120 experiment remain in backups B3 and B5.

**Speaker notes:**

The forecasting models receive a past window and predict ten future signal values. I evaluate recurrent models, PatchTST, ten separate XGBoost regressors, and Chronos in zero-shot mode. The trained model families include raw and context-standardized variants. Standardization uses the past window, and forecasts are converted back to the original scale before evaluation.

The immediate switch rule requires every predicted point to remain at or above the threshold. This is evaluated at the current decision time, rather than by mixing predictions issued at different times.

The raw GRU sequence-to-vector model has the lowest test RMSE. However, raw PatchTST achieves the highest switch F1 on the native autoregressive grid. The trajectory is useful, but average forecast error does not specifically weight mistakes that change the switch decision. This formulation is effective but indirect: we predict an entire trajectory to answer a persistence question.

**Transition:** The next formulation removes the trajectory and predicts a duration directly.

## Slide 9. Current-Level Persistence

**On screen:**

- Target: time until the signal falls below S(t) - 0.5 dB.
- Models: XGBoost and shapelet models with MLP, convolutional, or Transformer heads.
- Switch: current signal at least 10 dB and predicted duration at least 300 s.
- Input: relative-to-current history plus scalar signal features.
- Native test XGBoost: duration MAE 586.9 s, median absolute error 27.2 s, switch F1 0.902.
- Current-level recovery is not necessarily fade recovery; all three shapelet variants have no final switch activations on this test set.
- Useful switches can coexist with large duration errors.

**Visual:** the same two-column layout as slide 8, with three metric rows. The level/fade mismatch and shapelet outcome remain visible as short statements; B6 and B7 provide the detailed explanation.

**Additional native-test values for Q&A:** XGBoost switch precision 1.000 and recall 0.822. These remain off the main slide.

**Speaker notes:**

This target measures time until the signal falls below its current level minus half a decibel. That is not the same as fade recovery: the signal could drop and still remain above ten decibels.

XGBoost has a median absolute error of twenty-seven seconds but a mean error of almost ten minutes. Useful switching does not contradict these large errors. For example, predicting ten minutes instead of one hour is a very large duration error, but both exceed the five-minute decision threshold. This example illustrates the distinction; it is not a measured decomposition of our errors.

The current-signal gate and shared post-processing also contribute to the final policy. XGBoost's switches are conservative on this holdout, but this does not establish reliable duration estimation. The tested shapelet variants never activate because their predictions fail the five-minute gate when the signal is above threshold.

**Transition:** Classification offers another shortcut, but it changes the meaning of the prediction.

## Slide 10. Long-Fade Detection

**On screen:**

- Target: total duration of the grouped event is at least 300 s.
- The same label applies throughout the grouped event, even near its end.
- Models: XGBoost, TCN, shapelet convolution classifier.
- Switch: predicted class probability at least 0.5.
- Nearly perfect native classification, but excessive switch activity.

**Visual:** three rows: classification AUPRC approximately 1.000, switch precision 0.540 and recall 1.000. These reveal excessive activation more directly than F1 alone. Keep the event-conditioned, highly imbalanced evaluation caveat visible. B6 contains the full grouping definition.

**Additional native-test values for Q&A:** TCN switch F1 0.702 and active duration 445 min. These are native-grid values, not the common-grid results shown later.

**Speaker notes:**

Long-fade detection predicts whether a grouped event is long. Nearby threshold crossings can belong to the same group, and all eligible timestamps within that event share its label. The dataset therefore focuses on event regions rather than continuous background monitoring.

XGBoost, the TCN, and the shapelet classifier obtain very high classification scores. These scores are conditional on this event-based dataset and should not be interpreted as near-perfect online fade detection.

When the probabilities become instantaneous switch decisions, the mismatch becomes clear. The TCN recovers every reference-positive sample on its native grid, but precision is only about fifty-four percent. A long event retains its positive label even near its end and across temporary recoveries in the group. Recognizing a long event therefore does not tell us how much of it remains. The formulation is useful for event characterization, but its total-duration label is poorly aligned with precise switch timing.

**Transition:** Survival modeling focuses instead on how much of the current fade remains.

## Slide 11. Survival Persistence

**On screen:**

- Output: a survival curve, S(u | X) = P(R > u | X), for remaining fade duration.
- Models: XGBoost-AFT predicts a time distribution; TCN predicts discrete hazards.
- Right-censored samples are retained.
- Switch: S(300 s | X) at least 0.65 and current signal at least 10 dB.

**Visible caveat:** 0.65 is an exploratory test-selected threshold, not a validation-calibrated operating point.

**Visual:** three actual test TCN curves, selected at the 5th, 50th and 95th percentiles of predicted S(300), with the five-minute horizon marked. One short equation defines the survival output. No metric comparison is shown on this slide. Detailed censoring, model equations and the IBS/F1 comparison are in B6, B9 and B10.

**Speaker notes:**

Survival modeling estimates a distribution over the remaining event duration. Of the four formulations, this directly addresses the question of whether the current fade will persist for another five minutes. If acquisition stops before recovery, we know only that the event continued for at least the observed period. These right-censored samples are retained.

XGBoost-AFT models the duration distribution through an accelerated failure-time formulation. The TCN predicts conditional event probabilities over time bins, from which we obtain a survival curve. Both use the same underlying continuous-time labels.

If useful in the spoken explanation: the TCN has the lower integrated Brier score, while XGBoost-AFT produces the better switch sequence at the evaluated operating point. The numbers are in B10: IBS 0.103 versus 0.117, and switch F1 0.784 versus 0.863, respectively. Keep this numerical detour for questions if time is tight.

Direct target alignment does not guarantee the best empirical policy. Probabilities need calibration assessment, and the operating threshold must be chosen independently of the test set. Here 0.65 was explored on the test set, so the reported switches remain exploratory rather than confirmatory evidence of superiority.

**Transition:** To compare the tasks themselves, we must also align their decision timestamps.

## Slide 12. Cross-Task Switch Performance

**On screen:** selected representatives on the common reference grid, after shared post-processing.

| Task / representative | Native coverage | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Current-level / XGBoost | 89.0% | 1.000 | 0.808 | 0.894 |
| Survival / XGBoost-AFT | 9.2% | 0.812 | 0.938 | 0.870 |
| Autoregressive / PatchTST raw | 89.2% | 0.779 | 0.821 | 0.800 |
| Long-fade / TCN | 28.7% | 0.557 | 1.000 | 0.715 |

**Visible caveats:** missing native decisions map to zero. Survival uses the exploratory 0.65 threshold. Representatives summarize observed test behavior, not a new validation-selected cross-task winner.

**Visual:** use either this compact table or a four-method precision-recall plot with coverage labels. Do not put the full all-model chart next to another dense table.

**Speaker notes:**

The tasks make predictions on different domains. In the main comparison, I map their decisions onto a shared reference grid and fill missing native decisions with zero. Coverage therefore becomes part of the result, rather than being ignored.

Current-level XGBoost achieves the highest observed F1 among these representatives and has no false-positive switch samples on this grid, but misses some reference activity. Survival XGBoost-AFT recovers more of that activity, with some false positives. Raw PatchTST is the strongest autoregressive representative, while long-fade detection prioritizes recall.

Survival's low overall timestamp coverage does not imply equally low recall: its native domain concentrates on active fades. It still needs fallback behavior outside that domain. These results identify different trade-offs, rather than a universally superior controller.

**Transition:** Those trade-offs also affect how long the backup remains active and how often it is activated.

## Slide 13. Active Duration and Switching Events

**On screen:** common-grid total active duration, in minutes.

| Reference / method | Active minutes |
| --- | ---: |
| Perfect Switch | 234.5 |
| Current-level XGBoost | 189.5 |
| PatchTST raw | 247.0 |
| Survival XGBoost-AFT | 271.0 |
| Long-fade TCN | 421.0 |

**Visual:** horizontal duration bars with a Perfect reference line. Add a small conceptual example of two binary sequences with equal active duration but different numbers of contiguous intervals. Do not claim the conceptual example is experimental data.

**Speaker notes:**

On the common grid, Perfect Switch is active for about two hundred and thirty-five minutes. Current-level XGBoost stays below that duration, reflecting its conservative behavior. PatchTST is close in total duration, while survival XGBoost-AFT stays active somewhat longer. Long-fade TCN reaches more than four hundred minutes.

Duration alone is still insufficient. Two policies can be active for the same total time but divide it into very different numbers of intervals. Fragmentation matters because each activation represents another transition between links. A switch interval is a contiguous run of positive decisions, not necessarily a separate physical fade event.

For example, within the native autoregressive evaluation, PatchTST produces twenty switch intervals compared with eighteen for Perfect. That is a within-task diagnostic. Counts on different native grids must not be combined as if they shared the same reference domain.

**Transition:** A single event makes these timing and fragmentation differences easier to see.

## Slide 14. An Event-Level Comparison

**On screen:** one signal trace and five aligned rows: Perfect, current-level XGBoost, PatchTST raw, long-fade TCN, survival XGBoost-AFT.

**Visual:** reduced-row common-grid `dataset_007_event_00007` timeline, regenerated from the same underlying data as `../figures/results/cross_task_event00007_all_methods_switches.png`. Show final decisions only, with one common Perfect reference; do not assemble task-native plots with different reference domains.

**Speaker notes:**

This event contains a short early fade and a longer interval later on, followed by additional activity. Every row uses the same time axis and reference. The step curves show final binary decisions after shared post-processing: the lower level is zero and the upper level is one.

The early reference interval is recovered by survival XGBoost-AFT and the long-fade classifier, but not by current-level XGBoost or raw PatchTST. During the sustained central fade, the representative methods agree more closely. Differences become visible again around recovery and the later intervals.

This example illustrates why neither total duration nor a single aggregate score tells the whole story. It also shows that post-processing contributes substantially to the final policy. The event is an illustrative case, not independent evidence of general superiority.

**Transition:** These findings are useful, but the experimental scope sets limits on what we can conclude.

## Slide 15. Limitations and Next Steps

**On screen:**

| Limitation | Next step |
| --- | --- |
| One external test source | More terminals and acquisition periods |
| Exploratory survival threshold | Validation-only, cost-aware calibration |
| Offline evaluation and partial decision coverage | Streaming prototype and explicit fallback policy |

**Speaker notes:**

The results come from one external test source, so they do not establish robustness across all terminals or weather conditions. More independent holdouts are needed. The survival operating point also needs independent calibration on validation data.

Finally, this is an offline experimental framework, not a deployed controller. A streaming implementation must audit data availability during preprocessing, maintain switch state, measure latency, and define what happens when a task cannot produce a decision. Comparing against an operational engineering controller would then be an important additional step.

**Transition:** Within these limits, the experiments support three main conclusions.

## Slide 16. Conclusions

**On screen:**

- Autoregressive: useful trajectory-based decisions, but forecast errors are not switch-specific.
- Current-level: useful conservative XGBoost switches despite unreliable duration estimates; level recovery differs from fade recovery.
- Long-fade: strong event classification, but total duration is not residual duration.
- Survival: directly aligned with residual persistence, with calibration still required.
- No universal winner: one holdout and an exploratory survival operating threshold.

**Closing statement:** Evaluate target alignment, prediction quality and the final switch policy together.

**Speaker notes:**

The four formulations have distinct strengths and limitations. Forecasting supports useful decisions, but average forecast error is not a switch-specific objective. Current-level XGBoost can switch conservatively despite large duration errors; its target is only a proxy for fade persistence. Long-fade classification recognizes long events but does not estimate how much time remains. Survival addresses that residual-duration question directly, although calibration and independent evaluation are still needed.

We therefore have different operating trade-offs, not a universal winner. The central conclusion is to evaluate target alignment, prediction quality and the final switch policy together.

## Slide 17. Thank You / Questions

Leave the closing screen visible during questions. Thank you for your attention. I am happy to take your questions.

The small Backup material button links to the first backup. Each backup has a Questions button to return here.

## Backup Slides

Implemented in `backup_slides.tex`, after the closing screen. These are not part of the timed talk, the main numeric total, or the Berlin navigation markers.

| Backup | Question it supports | Key point |
| --- | --- | --- |
| B1 | How does each model activate a switch? | Four explicit raw-decision rules, followed by shared post-processing. |
| B2 | What exactly is Perfect? Does minimum-island remove short events? | Eleven reference observations span 300 s; short decision islands are extended, then held above threshold. |
| B3 | Which architectures and input variants were evaluated? | Full family matrix, including Chronos L120 and selected survival representations. |
| B4 | How extensive was model selection? | Archived trial counts and validation-only selection criteria. |
| B5 | How are inputs normalized and gaps handled? | Context-only normalization, train-fitted scalar scaling, and offline interpolation availability caveat. |
| B6 | Why are event definitions and censored samples different? | Current-level recovery, grouped events and survival recovery are distinct targets; acquisition gaps bound observation. |
| B7 | How can large duration error coexist with useful switches? | Illustrative duration-threshold counterexamples; not empirical samples or a claim of perfect target alignment. |
| B8 | Is cross-task comparison fair? | Native, common-reference and strict-intersection domains answer different questions. |
| B9 | How can duration labels produce a survival probability? | AFT time distribution versus TCN conditional hazards; both account for censoring. |
| B10 | Is the survival ranking statistically meaningful? | Event-bootstrap C-index interval includes zero; probabilistic accuracy and switch F1 are different. |

For exact winning configurations, use thesis Appendix B and saved run metadata rather than adding dense hyperparameter tables to the talk. B4 counts describe archived searches; B10 uses the refreshed external-test metrics reported in Chapter 6. Do not interpret the C-index uncertainty calculation as a significance test for IBS or switch F1.

## Evidence and Figure Preparation Notes

All paths below are relative to this file unless explicitly described as repository-relative.

- Narrative and native results: `../chapters/06_results.tex`.
- Experimental protocol: `../chapters/05_models_and_experimental_setup.tex`.
- Conclusions and limitations: `../chapters/07_conclusions.tex`.
- Main cross-task evidence: `../../results/comparisons/cross_task_switch/externalHoldout_test_fc_uplink_fade/crossTask_switch_externalHoldout_test_fc_uplink_fade_referenceGrid/tables/cross_task_switch_metrics_reference_grid.csv`.
- Slide 7's illustrative extension and holding sequence follows `../../src/switching/conversion.py`. Ten active samples describe the configured extension length; do not confuse this with the Perfect reference's eleven threshold samples spanning ten sampling intervals. Keep that implementation detail in backup material.
- Slide 13 common-grid durations are the saved `active_duration_seconds` divided by 60. The Perfect duration is 469 positive samples times 30 seconds, divided by 60. These are not the differing native-grid durations in the thesis's individual task sections.
- Slide 8 native results use the canonical L30 setup. Its PatchTST F1 of 0.808 differs legitimately from the common-grid value of 0.800 on slide 12.
- Slide 11 figure source: `../figures/results/survival_tcn_selected_survival_curves.png`.
- Slide 14 source image was inspected when preparing this draft. Its early and sustained intervals support the proposed explanation. Preserve the common reference when reducing the number of rows.
- Reuse the thesis bibliography for model attributions in slide footers or notes. Do not introduce an unverified project affiliation or external performance claim.
- The company context follows `../chapters/01_introduction.tex`. That chapter also mentions the broader NEFOCAST context, but direct attribution of this thesis to that project was previously uncertain. Confirm the relationship with the company before adding a NEFOCAST project claim to the spoken introduction. The present opening states the company and application context without making that claim.
- The shared code extends short islands. Do not carry over the phrase "island-removal" that still appears in the thesis cross-task prose.
- Do not claim that a low overall duration median error guarantees accurate five-minute decisions, that a C-index point-estimate gap is statistically significant, or that offline results establish superiority over a deployed controller.
- Regenerate presentation-sized figures from their sources where necessary. Do not shrink the all-model thesis mosaics until their labels become unreadable.
- Keep selected representatives visibly identified as such. The full model comparison belongs in backup slides.

## Rehearsal

The time allocations include explaining and pointing to figures, not just reading the notes. After the first rehearsal, shorten the four task explanations if needed rather than rushing the cross-task results. Keep the threshold caveat and the native/common-grid distinction even in a shorter version.

If time is running short, omit the native island-count example on slide 13 and shorten the event walkthrough. Preserve the final conclusions. This document is a starting script, not text to memorize word for word.
