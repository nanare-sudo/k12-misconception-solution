"""Cleaning of the transcriptions — with invented examples, never with competition data."""
import transcription as tr


def test_step_lines_keep_only_prefixed_lines():
    ts = "L1 | 3 + 4 = 7\nL2 | 7 × 2 = 14\n[note without prefix]\nL3a | = 14 cm"
    assert tr.step_lines(ts) == [(1, "3 + 4 = 7"), (2, "7 × 2 = 14"), (3, "= 14 cm")]
    assert tr.step_lines("L1 | a\r\nL2 | b\rL3 | c") == [(1, "a"), (2, "b"), (3, "c")]


def test_verdict_line_without_prefix_is_removed():
    ts = "L1 | 12/5 × 5/13\nL2 | = 12/13\n[pupil solved it correctly]"
    assert tr.has_leak(ts)
    assert tr.strip_leaks(ts) == "L1 | 12/5 × 5/13\nL2 | = 12/13"
    assert not tr.has_leak(tr.strip_leaks(ts))


def test_verdict_bracket_inside_step_line_is_removed_but_descriptions_stay():
    ts = "L1 | 2x = 8 [crossed out: 2x = 6]\nL2 | x = 4 [answer is correct]\nL3 | [sketch at the right]"
    out = tr.strip_leaks(ts)
    assert out == "L1 | 2x = 8 [crossed out: 2x = 6]\nL2 | x = 4\nL3 | [sketch at the right]"
    # "right" as a direction is not a verdict; "incorrectly"/"wrong" are
    assert tr.strip_leaks("L1 | 5 [put down incorrectly as 6]\nL2 | 7 [wrong sign]") == "L1 | 5\nL2 | 7"


def test_real_verdicts_in_brackets_are_leaks_in_every_word_form():
    for note in ("[answer is correct]", "[marked as correctly done]", "[INCORRECT]", "[copied incorrectly]", "[wrong unit]", "[sign wrongly kept]"):
        ts = f"L1 | 5 {note}\nL2 | = 5"
        assert tr.has_leak(ts), note
        assert tr.strip_leaks(ts) == "L1 | 5\nL2 | = 5", note
        assert not tr.has_leak(tr.strip_leaks(ts)), note


def test_descriptions_of_visible_self_corrections_are_not_verdicts():
    # The word forms for "corrected / correction" describe what can be seen on the sheet.
    notes = (
        "[the 7 is struck through and corrected to 9]",
        "[pupil self-corrected the sum]",
        "[self-correction above the line]",
        "[small correction mark next to the 4]",
        "[corrective overwrite on the last digit]",
        "[tries to self-correct the remainder]",
    )
    for note in notes:
        ts = f"L1 | 48 : 6 = 9 {note}\nL2 | = 8"
        assert not tr.has_leak(ts), note
        assert tr.strip_leaks(ts) == ts, note
    # student_text still removes every bracketed annotation
    assert tr.student_text(f"L1 | 48 : 6 = 9 {notes[0]}") == "L1 | 48 : 6 = 9"


def test_verdict_in_round_aside_removes_only_the_aside():
    ts = "L1 | [second row: 4 below 12, 6 below 18 (this entry is wrong), then 9 below 27]\nL2 | KPK = 36"
    assert tr.has_leak(ts)
    assert tr.strip_leaks(ts) == "L1 | [second row: 4 below 12, 6 below 18, then 9 below 27]\nL2 | KPK = 36"
    assert not tr.has_leak(tr.strip_leaks(ts))
    # Verdict outside a round addition: the whole annotation is dropped, including its descriptive part.
    assert tr.strip_leaks("L1 | 3 [first digit hard to read; the total is correct anyway]") == "L1 | 3"
    # If the annotation consists only of the verdict, no empty bracket remains.
    assert tr.strip_leaks("L1 | 3 [(correct)]\nL2 | [(wrong)]\nL3 | 4") == "L1 | 3\nL3 | 4"
    # A round addition without a verdict stays untouched.
    assert tr.strip_leaks("L1 | 3 [boxed (twice)]") == "L1 | 3 [boxed (twice)]"


def test_letterless_lines_after_a_step_line_are_continuations_not_leaks():
    ts = "L1 | 24\n× 3\n-----\n72\nL2 | jadi 72"
    assert not tr.has_leak(ts)
    assert tr.strip_leaks(ts) == "L1 | 24 | × 3 | ----- | 72\nL2 | jadi 72"
    assert tr.student_text(ts) == "L1 | 24 | × 3 | ----- | 72\nL2 | jadi 72"
    assert tr.strip_leaks(tr.strip_leaks(ts)) == tr.strip_leaks(ts)
    # step_lines keeps the prefix rule: lines without a prefix are still skipped there.
    assert tr.step_lines(ts) == [(1, "24"), (2, "jadi 72")]
    # A check mark is a verdict, a line with words is annotator text — both remain a leak.
    for leak_line in ("✓", "= 4 ✗", "the result can be found in the first line"):
        ts = f"L1 | 2 + 2 = 4\n{leak_line}"
        assert tr.has_leak(ts), leak_line
        assert tr.strip_leaks(ts) == "L1 | 2 + 2 = 4", leak_line
    # Before the first student line there is nothing to continue: counts as annotator text and is dropped.
    assert tr.has_leak("12 + 3\nL1 | = 15")
    assert tr.strip_leaks("12 + 3\nL1 | = 15") == "L1 | = 15"


def test_format_stage_squeezes_spaces_and_is_not_a_leak():
    ts = '"\nL1 |   12   +    3\nL2a | =  15'
    assert not tr.has_leak(ts)
    assert tr.strip_leaks(ts) == "L1 | 12 + 3\nL2 | = 15"
    assert tr.squeeze_spaces("  7    ×  8 ") == "7 × 8"


def test_annotation_markup_block_is_removed_even_if_misspelled_or_unclosed():
    ts = 'L1 | 60 : 15 = 5\n<misconcepton category="ABC-01" severity="major" step="L1">\nexplanation text\n</misconcepton>\nL2 | = 5 menit'
    assert tr.has_leak(ts)
    assert tr.strip_leaks(ts) == "L1 | 60 : 15 = 5\nL2 | = 5 menit"
    unclosed = 'L1 | 1 + 1 = 3\n<miskonception category="ABC-01" step="L1">\nstill open'
    assert tr.has_leak(unclosed)
    assert tr.strip_leaks(unclosed) == "L1 | 1 + 1 = 3"


def test_clean_transcription_is_unchanged_and_idempotent():
    ts = "L1 | KPK = 2³ × 3 = 24\nL2 | [factor tree: 91 → 7, 13]\nL3 | 2 April + 24 hari"
    assert not tr.has_leak(ts)
    assert tr.strip_leaks(ts) == ts
    assert tr.strip_leaks(tr.strip_leaks(ts)) == tr.strip_leaks(ts)


def test_student_text_drops_all_bracket_notes():
    ts = "L1 | 2/5 = 16/40 [arrow above]\nL2 | [table]\nL3 | n = 9/40"
    assert tr.student_text(ts) == "L1 | 2/5 = 16/40\nL3 | n = 9/40"


def test_empty_and_non_string_input():
    assert tr.strip_leaks("") == "" and tr.step_lines("") == []
    assert tr.strip_leaks(float("nan")) == ""
    assert not tr.has_leak("")
