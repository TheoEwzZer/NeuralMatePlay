"""Test results tracking and reporting."""

from collections import defaultdict
from typing import Any

from .colors import Colors, subheader, dim


class TestResults:
    """Track test results for final scoring and LLM analysis."""

    def __init__(self) -> None:
        self.tests: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}
        self.diagnostics: dict[str, dict[str, Any]] = {}
        self.issues: list[dict[str, str | None]] = []
        self.recommendations: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        passed: bool,
        score: float | None = None,
        max_score: float = 1.0,
    ) -> None:
        """Add a test result."""
        if score is None:
            score = 1.0 if passed else 0.0
        self.tests.append(
            {"name": name, "passed": passed, "score": score, "max_score": max_score}
        )

    def add_timing(self, name: str, duration: float) -> None:
        """Add a timing measurement."""
        self.timings[name] = duration

    def add_diagnostic(self, category: str, key: str, value: Any) -> None:
        """Add diagnostic data for LLM analysis."""
        if category not in self.diagnostics:
            self.diagnostics[category] = {}
        self.diagnostics[category][key] = value

    def add_issue(
        self, severity: str, category: str, description: str, details: str | None = None
    ) -> None:
        """Add an identified issue."""
        self.issues.append(
            {
                "severity": severity,  # CRITICAL, HIGH, MEDIUM, LOW
                "category": category,
                "description": description,
                "details": details,
            }
        )

    def add_recommendation(self, priority: int, action: str, reason: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(
            {"priority": priority, "action": action, "reason": reason}
        )

    def total_score(self) -> tuple[float, float]:
        """Calculate total score."""
        earned: float = sum(t["score"] for t in self.tests)
        maximum: float = sum(t["max_score"] for t in self.tests)
        return earned, maximum

    def print_summary(self) -> float:
        """Print final summary with detailed LLM-friendly output."""
        from .colors import header

        print(header("FINAL SUMMARY"))

        earned: float
        maximum: float
        earned, maximum = self.total_score()
        percentage: float = (earned / maximum * 100) if maximum > 0 else 0

        color: str
        status: str
        if percentage >= 80:
            color = Colors.GREEN
            status = "HEALTHY"
        elif percentage >= 50:
            color = Colors.YELLOW
            status = "NEEDS IMPROVEMENT"
        else:
            color = Colors.RED
            status = "CRITICAL ISSUES"

        print(subheader("Test Results"))
        print(f"{'Test':<35} {'Score':>10} {'Status':>10}")
        print("-" * 60)

        for t in self.tests:
            status_str: str = (
                f"{Colors.GREEN}PASS{Colors.ENDC}"
                if t["passed"]
                else f"{Colors.RED}FAIL{Colors.ENDC}"
            )
            score_str: str = f"{t['score']:.2f}/{t['max_score']:.1f}"
            pct: float = (
                (t["score"] / t["max_score"] * 100) if t["max_score"] > 0 else 0
            )
            bar_len: int = int(pct / 10)
            bar: str = "#" * bar_len + "-" * (10 - bar_len)
            print(f"  {t['name']:<33} {score_str:>10} [{bar}] {status_str}")

        print("-" * 60)
        print(
            f"\n{Colors.BOLD}Overall Score: {color}{earned:.1f}/{maximum:.1f} ({percentage:.0f}%){Colors.ENDC}"
        )

        grade: str
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        print(f"{Colors.BOLD}Grade: {color}{grade}{Colors.ENDC} - {status}")

        if self.timings:
            print(subheader("Performance Metrics"))
            for name, duration in self.timings.items():
                time_str: str
                if duration < 0.01:
                    time_str = f"{duration*1000000:.1f}μs"
                elif duration < 1:
                    time_str = f"{duration*1000:.1f}ms"
                else:
                    time_str = f"{duration:.2f}s"
                print(f"  {name:<30} {time_str:>15}")

        if self.issues:
            print(subheader("Identified Issues"))

            by_severity: defaultdict[str, list[dict[str, str | None]]] = defaultdict(
                list
            )
            for issue in self.issues:
                by_severity[issue["severity"]].append(issue)

            severity_order: list[str] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            severity_colors: dict[str, str] = {
                "CRITICAL": Colors.RED,
                "HIGH": Colors.RED,
                "MEDIUM": Colors.YELLOW,
                "LOW": Colors.BLUE,
            }

            for severity in severity_order:
                if severity in by_severity:
                    color = severity_colors[severity]
                    print(f"\n  {color}{Colors.BOLD}[{severity}]{Colors.ENDC}")
                    for issue in by_severity[severity]:
                        print(f"    • {issue['description']}")
                        if issue["details"]:
                            print(f"      {dim(issue['details'])}")

        if self.recommendations:
            print(subheader("Recommendations"))
            sorted_recs: list[dict[str, Any]] = sorted(
                self.recommendations, key=lambda x: x["priority"]
            )
            for i, rec in enumerate(sorted_recs, 1):
                print(f"  {i}. {Colors.BOLD}{rec['action']}{Colors.ENDC}")
                print(f"     {dim(rec['reason'])}")

        return percentage
