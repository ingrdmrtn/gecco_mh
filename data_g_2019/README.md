# data_g_2019

Self-report questionnaire data and two-step task behavioral data from Gillan et al. (2016, 2019).

## Files

### `twostep_data_study1_individual_csv/`
- **649 files**, one CSV per participant, named by subject ID
- Raw two-step task data from Study 1 (Gillan et al., 2016)
- *Notes from Claire Gillan (November 2019):*
  - Task data starts the row after the one that lists `twostep_instruct_9` in column C
  - Subjects can have a varying number of rows before the task starts due to repeated instructions (multiple `instructionLoop` instances)

**Column reference (lettered columns in raw files):**

| Column | Name | Description |
|---|---|---|
| A | `trial_num` | Trial number |
| B | `drift_1` | Reward probability after stage 2, option 1 |
| C | `drift_2` | Reward probability after stage 2, option 2 |
| D | `drift_3` | Reward probability after stage 2, option 3 |
| E | `drift_4` | Reward probability after stage 2, option 4 |
| F | `stage_1_response` | Left or right |
| G | `stage_1_selected_stimulus` | 1 or 2 (redundant with response — stage 1 options do not switch locations) |
| H | `stage_1_rt` | Stage 1 reaction time |
| I | `transition` | Common = TRUE; rare = FALSE |
| J | `stage_2_response` | Left or right |
| K | `stage_2_selected_stimulus` | 1 or 2 (redundant with response — stage 2 options also do not switch locations) |
| L | `stage_2_state` | Identity of second stage reached (2 or 3) |
| M | `stage_2_rt` | Stage 2 reaction time |
| N | `reward` | 1 = rewarded; 0 = not rewarded |
| O | `redundant` | Always 1; redundant task variable |



### `self_report_study1.csv`
- **N = 548 participants**
- Online MTurk sample from Gillan et al. (2016)
- Columns: `subj.x`, `age`, `iq`, `gender`, `sds_total`, `stai_total`, `oci_total`

### `self_report_study2.csv`
- **N = 1,413 participants**
- Larger online sample from Gillan et al. (2019) with expanded transdiagnostic measure set
- Columns: `subj`, `age`, `iq`, `gender`, `lsas_total`, `bis_total`, `sds_total`, `scz_total`, `aes_total`, `stai_total`, `eat_total`, `audit_total`, `oci_total`, `Factor1`, `Factor2`, `Factor3`

### `individual_items_study2.csv`
- **N = 1,413 participants**, 209 questionnaire items
- Raw item-level responses for all Study 2 measures
- Covers: SCZ (43 items), OCI (18), EAT (26), AES (18), AUDIT (10), SDS (20), STAI (20), BIS (30), LSAS (24)

### `weights_study2.csv`
- **209 rows** (one per questionnaire item)
- Factor loadings for each item onto 3 transdiagnostic factors: `AD`, `CIT`, `SW`

## Measures

| Abbreviation | Full Name |
|---|---|
| OCI | Obsessive-Compulsive Inventory |
| STAI | State-Trait Anxiety Inventory |
| SDS | Self-rating Depression Scale |
| LSAS | Liebowitz Social Anxiety Scale |
| BIS | Barratt Impulsiveness Scale |
| SCZ | Schizotypy Questionnaire |
| AES | Apathy Evaluation Scale |
| EAT | Eating Attitudes Test |
| AUDIT | Alcohol Use Disorders Identification Test |

## Transdiagnostic Factors (Study 2)

Three symptom dimensions derived from factor analysis across all 209 items:
- **AD** — Anxious-Depression
- **CIT** — Compulsive/Intrusive Thoughts
- **SW** — Social Withdrawal

## References

- Gillan, C. M., et al. (2016). Characterizing a psychiatric symptom dimension related to deficits in goal-directed control. *eLife*, 5, e11305.
- Gillan, C. M., et al. (2019). Oxford study of transdiagnostic symptoms.
