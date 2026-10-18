
# AfriHealth AI: Data Governance and Research Readiness

## 1. Purpose

This document defines the minimum governance requirements for
using datasets and publishing findings from AfriHealth AI.

## 2. Dataset Provenance

For each dataset, record:

- Dataset title and source.
- Original publication or repository.
- Version or release date.
- Date accessed.
- Licence or data-use terms.
- Permission for the intended research use.
- Target definition and label provenance.
- Data processing and exclusion decisions.
- Any restrictions on redistribution.

Do not redistribute data unless the applicable terms permit it.

## 3. Patient Privacy

Do not collect or publish unnecessary personal information.

Remove direct identifiers before analysis where appropriate.
Recognise that removing names alone may not eliminate re-identification
risk, especially in small or unusual patient groups.

Do not commit identifiable clinical records, credentials, access
tokens, or confidential datasets to a public repository.

## 4. Ethical Review

Before using hospital records, patient data, or other restricted
health information, determine the applicable institutional and
regulatory requirements.

Obtain ethics approval, permissions, and any required consent or
lawful basis before data collection or analysis.

Do not assume that public availability of a dataset removes every
ethical, legal, or licensing obligation.

## 5. Model and Code Versioning

For each release, record:

- Git commit or release identifier.
- Model artifact version and checksum.
- Metadata file version.
- Dataset version.
- Python and package versions.
- Feature schema and target definition.
- Preprocessing configuration.
- Training and evaluation procedures.
- Evaluation results and uncertainty estimates.

Never silently replace the saved model while retaining old metrics
as though they describe the new model.

## 6. Access and Security

Limit access to sensitive data to authorised people.

Use appropriate storage permissions and secure transfer methods.
Do not place secrets in source code or public documentation.

Review any cloud hosting and third-party services before handling
restricted research data.

## 7. Responsible Reporting

Report all prespecified outcomes, including disappointing results.

Distinguish internal evaluation from external validation.
Distinguish statistical performance from clinical usefulness.
Do not claim African population validity without appropriate evidence.

Document model limitations and potential sources of bias.

## 8. Research Release Checklist

- [ ] Data source and licence verified.
- [ ] Permissions and ethics requirements reviewed.
- [ ] Personal identifiers excluded or appropriately protected.
- [ ] Dataset schema and target labels documented.
- [ ] Model artifact and dependencies versioned.
- [ ] Analysis procedure recorded.
- [ ] Results independently checked.
- [ ] Confidence intervals and limitations reported.
- [ ] Claims supported by completed analyses.
- [ ] Code and data sharing terms reviewed.

## 9. Intended Use

AfriHealth AI remains a research and educational prototype.
It is not established as a diagnostic tool and must not be used
to make treatment decisions.
