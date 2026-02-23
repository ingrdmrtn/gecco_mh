# data_g_2019

Self-report questionnaire data from Gillan et al. (2016, 2019). No behavioral task data is included here — see `../data/` for two-step task data.

## Files

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
