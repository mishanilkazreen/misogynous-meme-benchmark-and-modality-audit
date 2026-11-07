<!-- cspell:ignore Pramanick pramanick capalign IIITD Tiqv gdown harmeme -->
# HarMeme (`Harm-C`) Dataset Setup (`DIS-01`)

This directory stores the COVID-19 harmful meme dataset (`Harm-C`) introduced by Pramanick et al.
(*Findings of EMNLP 2021*, `pramanick2021momenta`) and evaluated in `CapAlign` (*WWW 2024*, `capalign2024www`, Table 5).

## Splits and Binary Label Mapping

The official `Harm-C` JSONL split annotations come from the `LCS2-IIITD/MOMENTA` repository
(`HarMeme_V1/Annotations/Harm-C/{train,val,test}.jsonl`):

| Split | File | Total Memes | Harmless (`0`) | Harmful (`1`) |
| --- | --- | ---: | ---: | ---: |
| Train | `annotations/train.jsonl` | 3,013 | 1,949 | 1,064 |
| Validation | `annotations/validation.jsonl` | 177 | 116 | 61 |
| Test | `annotations/test.jsonl` | 354 | 230 | 124 |
| **Total** | | **3,544** | **2,295** | **1,249** |

Following Pramanick et al. (2021) and `CapAlign` (2024), the 3-way intensity labels in `labels` are collapsed to
binary harmfulness:

- `"not harmful"` -> `0` (harmless)
- `"somewhat harmful"` or `"very harmful"` -> `1` (harmful)

## Automated Download and Verification

Run the download subcommand on a machine with internet access (e.g., the SCIAMA login node before submitting `S6`):

```bash
uv run python -m scripts.revision.harmeme_eval download
```

This command:

1. Downloads `train.jsonl` (`3,013`), `validation.jsonl` (`177`), and `test.jsonl` (`354`) from
   `https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-C/`.
2. Downloads `HarMeme_Images.zip` (Google Drive file ID `1aMvOHACrG5SgMl4tm9BP1VTiqv8zFn9b`, linked from
   `LCS2-IIITD/MOMENTA`) and extracts all `3,544` PNG images into `data/harmeme/images/` (`covid_memes_*.png`).
3. Verifies exact split counts and confirms that all `3,544` images exist on disk.

If Google Drive rate-limits automated downloads on the cluster login node, download `HarMeme_Images.zip` locally from
`https://drive.google.com/file/d/1aMvOHACrG5SgMl4tm9BP1VTiqv8zFn9b/view`, copy it to `data/harmeme/HarMeme_Images.zip`
on SCIAMA via `scp`, and re-run `uv run python -m scripts.revision.harmeme_eval download`.
