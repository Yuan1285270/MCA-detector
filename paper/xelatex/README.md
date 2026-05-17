# XeLaTeX Paper Files

This folder note describes how to compile the XeLaTeX paper drafts in `paper/`.
The layout follows the same conference-paper style as the reference files on the Desktop, but the content is rewritten for the current MCA Detector pipeline.

## Files

```text
paper/regular_session_zh_xelatex.tex
paper/regular_session_en_xelatex.tex
```

The Chinese version uses `ctex`, `IEEEtran`, and PingFang TC.
The English version uses `IEEEtran`.

## Compile

From the repository root:

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/regular_session_zh_xelatex.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/regular_session_zh_xelatex.tex
```

English version:

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/regular_session_en_xelatex.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/regular_session_en_xelatex.tex
```

Run twice so references and table numbers settle.

## Overleaf

Upload the `.tex` file to Overleaf and set the compiler to **XeLaTeX**.

The paper intentionally avoids IEEE branding in the template because the regular session instruction says IEEE-related names or descriptions should be removed from final submissions.

If Overleaf does not have `PingFang TC`, replace the Chinese font lines with an available CJK font such as `Noto Serif CJK TC` or remove the explicit `\setCJK...` lines and let `ctex` choose defaults.
