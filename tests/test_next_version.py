import next_version as nv


def c(subject, body=""):
    return (subject, body)


def test_first_release_starts_at_1_0_0():
    assert nv.next_version(None, [c("feat: x")]) == (1, 0, 0)


def test_patch_for_fixes_and_housekeeping():
    assert nv.next_version("v1.2.3", [c("fix: a"), c("chore: b"), c("ci: c")]) == (1, 2, 4)
    assert nv.next_version("v1.2.3", [c("some message without a type")]) == (1, 2, 4)


def test_minor_for_a_feature():
    assert nv.next_version("v1.2.3", [c("fix: a"), c("feat(api): b")]) == (1, 3, 0)


def test_major_for_breaking_changes():
    assert nv.next_version("v1.2.3", [c("feat!: drop v1")]) == (2, 0, 0)
    assert nv.next_version("v1.2.3", [c("fix(api)!: x")]) == (2, 0, 0)
    assert nv.next_version("v1.2.3", [c("fix: x", "BREAKING CHANGE: gone")]) == (2, 0, 0)


def test_an_already_tagged_head_keeps_its_version():
    assert nv.next_version("v1.4.0", []) == (1, 4, 0)


def test_a_feature_in_a_longer_word_is_not_a_feature():
    assert nv.next_version("v1.0.0", [c("feature: nope")]) == (1, 0, 1)


def test_notes_group_by_type_and_flag_breaking_changes():
    text = nv.notes([c("feat(api): add x"), c("fix: repair y"), c("docs: z"), c("feat!: big")])
    assert "## Features" in text and "- **api:** add x" in text
    assert "- big **(breaking)**" in text
    assert "## Fixes" in text and "- repair y" in text
    assert "## Other changes" in text and "- z" in text
    assert nv.notes([]) == "No notable changes."
