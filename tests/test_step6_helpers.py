"""Helpers added in Step 6: parity OLS, wing buckets, the README audit tolerance and the Step 5 gate. Synthetic inputs only."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_readme_numbers as audit
import step5_market as m5
import step6_diagnostics as s6


def test_ols_recovers_line_and_standard_errors():
    rng = np.random.default_rng(3)
    K = np.linspace(90, 110, 9)
    y = 4.0 - 0.99 * K + rng.normal(0, 0.05, K.size)
    b, a, cov = s6.ols(K, y)
    assert b == pytest.approx(-0.99, abs=0.01) and a == pytest.approx(4.0, abs=1.0)
    # textbook formula: var(b) = s^2 / sum((K - mean K)^2)
    resid = y - (a + b * K)
    s2 = resid @ resid / (K.size - 2)
    assert cov[0, 0] == pytest.approx(s2 / np.sum((K - K.mean()) ** 2), rel=1e-9)
    assert cov[0, 1] == pytest.approx(-K.mean() * cov[0, 0], rel=1e-9)  # slope and intercept are negatively correlated


def test_ols_exact_line_has_zero_error():
    K = np.array([95.0, 100.0, 105.0])
    b, a, cov = s6.ols(K, 3.0 - K)
    assert (b, a) == (pytest.approx(-1.0), pytest.approx(3.0)) and np.allclose(cov, 0.0, atol=1e-18)


@pytest.mark.parametrize("x, wing", [(0.80, "K/S < 0.93"), (0.9299, "K/S < 0.93"), (0.93, "0.93 <= K/S <= 1.03"), (1.0, "0.93 <= K/S <= 1.03"),
                                     (1.03, "0.93 <= K/S <= 1.03"), (1.0301, "K/S > 1.03")])
def test_wing_boundaries(x, wing):
    assert s6.wing_of(x) == wing


def test_audit_tolerance_is_displayed_precision():
    assert audit.tolerance("0.19") == pytest.approx(0.005) and audit.tolerance("1.5e-3") == pytest.approx(0.5 * 0.1 * 1e-3)
    assert audit.close("0.19", "0.1949") and not audit.close("0.19", "0.196")
    assert not audit.cell_mismatch("-6.14e-3", "-6.1449e-3") and audit.cell_mismatch("-6.14e-3", "-6.2e-3")
    assert audit.cell_mismatch("n/a", "0.5") and not audit.cell_mismatch("1,000", "1000")


def test_audit_detects_hand_edited_table(tmp_path, monkeypatch):
    import readme_tables as rt
    header, rows = rt.TABLES["bermudan_nex"]()
    good = "<!-- table:bermudan_nex -->\n" + rt.render(header, rows) + "\n<!-- /table:bermudan_nex -->\n"
    assert not [e for e in audit.audit_tables(good) if "bermudan_nex" in e]
    bad = good.replace(rows[0][1], "9.9999")
    assert any("bermudan_nex" in e for e in audit.audit_tables(bad))
    stray = good + "\n| a | b |\n|---|---|\n| 1.5 | 2 |\n"
    assert any("unmanaged" in e for e in audit.audit_tables(stray))


def test_gate_requires_snapshot_fields_and_rejects_unconfirmed_status(tmp_path):
    block = "snapshot_date: 2025-05-15\nticker_nondividend: AMZN\nticker_dividend: SPY\nmin_days: 14\nmax_days: 64\nexpiry_targets_days: 14, 28\ncash_distr_types: 1\n"
    ok = tmp_path / "ok.md"
    ok.write_text(block)
    assert m5.parse_hypotheses(ok)["cash_distr_types"] == "1"  # no status line: the committed file records confirmation
    draft = tmp_path / "draft.md"
    draft.write_text("status: DRAFT\n" + block)
    with pytest.raises(SystemExit):
        m5.parse_hypotheses(draft)

    tbd = tmp_path / "tbd.md"
    tbd.write_text(block.replace("cash_distr_types: 1", "cash_distr_types: TBD"))
    with pytest.raises(SystemExit):
        m5.parse_hypotheses(tbd)


def test_figure_manifest_records_provenance_and_audit_detects_drift(tmp_path, monkeypatch):
    import figstyle
    import matplotlib.pyplot as plt
    import readme_tables as rt
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    figstyle.save(fig, tmp_path / "demo_fig.png", "source: synthetic; N=7 contracts; snapshot 2000-01-01")
    fig, ax = plt.subplots()
    figstyle.save(fig, tmp_path / "demo_fig.png", "source: synthetic; N=8 contracts; snapshot 2000-01-01")  # upsert, not append
    lines = (tmp_path / "figure_manifest.csv").read_text().splitlines()
    assert len(lines) == 2 and "N=8" in lines[1]
    monkeypatch.setattr(rt, "ROOT", tmp_path.parent)
    (tmp_path.parent / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (tmp_path.parent / "results" / "figures" / "figure_manifest.csv").write_text((tmp_path / "figure_manifest.csv").read_text())
    good = rt.figure_provenance("demo_fig")
    assert "N=8 contracts" in good
    monkeypatch.setattr(rt, "CAPTIONS", ["demo_fig"])
    assert not [e for e in audit.audit_tables(f"<!-- caption:demo_fig -->\n{good}\n<!-- /caption:demo_fig -->\n") if e.startswith("caption")]
    errs = [e for e in audit.audit_tables(f"<!-- caption:demo_fig -->\n{good.replace('N=8', 'N=9')}\n<!-- /caption:demo_fig -->\n") if e.startswith("caption")]
    assert len(errs) == 1 and "differs from the figure manifest" in errs[0]