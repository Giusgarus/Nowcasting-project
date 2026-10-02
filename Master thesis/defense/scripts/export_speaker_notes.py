"""Extract printable spoken notes from the maintained presentation plan."""

from pathlib import Path
import re


DEFENSE = Path(__file__).resolve().parents[1]


def latex_escape(value: str) -> str:
    """Escape literal text without interpreting Markdown as LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def main() -> None:
    """Write a generated content file; keep the source plan unchanged."""
    plan = (DEFENSE / "presentation_plan.md").read_text(encoding="utf-8")
    timing = {}
    for line in plan.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].isdigit():
            timing[int(cells[0])] = cells[2:]

    slides = re.findall(
        r"^## Slide (\d+)\. ([^\n]+)\n(.*?)(?=^## |\Z)", plan,
        flags=re.MULTILINE | re.DOTALL,
    )
    if [int(number) for number, _, _ in slides] != list(range(1, 18)):
        raise ValueError("Expected the current 17-slide presentation plan.")

    output = [r"% Generated from presentation_plan.md; do not edit directly."]
    extras = []
    for number_text, title, body in slides:
        number = int(number_text)
        duration, cumulative = timing[number]
        if number == 17:
            speech = "Thank you for your attention. I am happy to take your questions."
        else:
            match = re.search(
                r"\*\*Speaker notes:\*\*\s*(.*?)(?=\n\*\*|\Z)", body, re.DOTALL
            )
            if match is None:
                raise ValueError(f"Missing spoken notes for slide {number}.")
            speech = match.group(1).strip()

        paragraphs = []
        for paragraph in speech.split("\n\n"):
            if paragraph.startswith("If useful in the spoken explanation:"):
                extras.append((f"Slide {number}: optional numerical comparison", paragraph))
            else:
                paragraphs.append(latex_escape(paragraph.strip()))
        if number > 1:
            output.append(r"\par\addvspace{6mm}")
        output.append(
            r"\noteslide{" + number_text + "}{" + latex_escape(title) + "}{"
            + latex_escape(duration) + "}{" + latex_escape(cumulative) + "}"
        )
        output.append("\n\n".join(paragraphs))
        transition = re.search(r"\*\*Transition:\*\* (.+)", body)
        if transition:
            text = transition.group(1)
            if number == 15:
                text = "Within these limits, the experiments support the following conclusions."
            output.append(r"\transition{" + latex_escape(text) + "}")
        for label in ("Additional native-test values for Q&A", "Methodological clarification for questions"):
            match = re.search(r"\*\*" + re.escape(label) + r":\*\* (.+)", body)
            if match:
                extras.append((f"Slide {number}: {label}", match.group(1)))

    output.extend([
        r"\par\medskip",
        r"\textit{End of the main talk. The following pages are optional Q\&A notes.}",
        r"\clearpage\section*{Questions: backup-slide guide}",
        r"\textit{Not part of the timed speech. Backup slides follow the closing screen in the deck.}",
    ])
    backup_rows = []
    for line in plan.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"B\d+", cells[0]):
            backup_rows.append(cells)
    if not backup_rows or [row[0] for row in backup_rows] != [
        f"B{index + 1}" for index in range(len(backup_rows))
    ]:
        raise ValueError("Expected consecutively numbered backup descriptions.")
    for index, (identifier, question, answer) in enumerate(backup_rows):
        if index > 0 and index % 5 == 0:
            output.append(r"\clearpage\section*{Questions: backup-slide guide (continued)}")
        output.append(r"\subsection*{" + latex_escape(identifier + " | " + question) + "}")
        output.append(latex_escape(answer))
    output.append(r"\clearpage\section*{Questions: additional details}")
    for heading, detail in extras:
        output.append(r"\subsection*{" + latex_escape(heading) + "}")
        output.append(latex_escape(detail))

    destination = DEFENSE / "build" / "speaker_notes_content.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(output) + "\n", encoding="utf-8")
    print(f"Exported {len(slides)} main-slide notes and {len(backup_rows)} backup entries.")
    print(destination)


if __name__ == "__main__":
    main()
