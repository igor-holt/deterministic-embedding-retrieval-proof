# Corpus v2 — deduplication note

## What changed

Corpus v1 (699 chunks) contained 6 byte-identical chunk pairs: 6 chunks in PDF 07
(`doe-rfi-cryogenic-quantum-testbeds/response-draft.pdf`) whose text is identical to
6 chunks in PDF 01 (`doe-rfi-cryogenic-quantum-testbeds/response-draft-v2-merged.pdf`).
PDF 07 is an earlier version of the same DOE response; PDF 01 is the merged v2.

## Rule

Remove the **second occurrence in corpus order** of any byte-identical chunk text; keep
the first occurrence. Applied once, to the 6 pairs below. No scoring rule, tie-break
rule, chunking rule, or model changed as part of the dedup itself. (The retrieval rule
change — top-1 as a *set* — is a separate, separately documented change; see REPORT.md.)

## Removed chunks (6)

| Removed (PDF 07) | Kept twin (PDF 01) | Verified |
|---|---|---|
| 07-000 | 01-000 | text byte-identical |
| 07-012 | 01-016 | text byte-identical |
| 07-030 | 01-034 | text byte-identical |
| 07-051 | 01-057 | text byte-identical |
| 07-058 | 01-063 | text byte-identical |
| 07-068 | 01-076 | text byte-identical |

Twin-text equality was verified programmatically (string equality on the normalized
chunk text) before removal. These were the ONLY identical-text duplicate pairs in the
corpus: 6 extra chunks beyond first copies, and all 6 are in this table.

## Result

- Corpus v2 = **693 chunks** (699 − 6). Per-PDF counts: 00:54, 01:79, 02:46, 03:11,
  04:15, 05:20, 06:94, 07:66, 08:42, 09:266.
- Chunk IDs are stable: the remaining PDF-07 chunks keep their original IDs
  (07-001, 07-002, …), so IDs 07-000, 07-012, 07-030, 07-051, 07-058, 07-068 are
  absent from v2 by design. Corpus order is unchanged apart from the removals.
- Files: `chunk_table_v2.json`, `chunk_table_v2.csv`. The v1 tables
  (`chunk_table.json`, `chunk_table.csv`) are preserved unchanged for the audit trail.
