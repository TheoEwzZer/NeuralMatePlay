"""Test results tracking and reporting."""

from collections import defaultdict
from .colors import Colors, subheader, dim


class TestResults:
    """Track test results for final scoring and LLM analysis."""

    def __init__(self):
        self.tests = []
        self.timings = {}
        self.diagnostics = {}  # Detailed diagnostic data for LLM
        self.issues = []  # List of identified issues
        self.recommendations = []  # List of recommendations

    def add(self, name: str, passed: bool, score: float = None, max_score: float = 1.0):
        """Add a test result."""
        if score is None:
            score = 1.0 if passed else 0.0
        self.tests.append(
            {"name": name, "passed": passed, "score": score, "max_score": max_score}
        )

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement."""
        self.timings[name] = duration

    def add_diagnostic(self, category: str, key: str, value):
        """Add diagnostic data for LLM analysis."""
        if category not in self.diagnostics:
            self.diagnostics[category] = {}
        self.diagnostics[category][key] = value

    def add_issue(
        self, severity: str, category: str, description: str, details: str = None
    ):
        """Add an identified issue."""
        self.issues.append(
            {
                "severity": severity,  # CRITICAL, HIGH, MEDIUM, LOW
                "category": category,
                "description": description,
                "details": details,
            }
        )

    def add_recommendation(self, priority: int, action: str, reason: str):
        """Add a recommendation."""
        self.recommendations.append(
            {"priority": priority, "action": action, "reason": reason}
        )

    def total_score(self) -> tuple:
        """Calculate total score."""
        earned = sum(t["score"] for t in self.tests)
        maximum = sum(t["max_score"] for t in self.tests)
        return earned, maximum

    def print_summary(self):
        """Print final summary with detailed LLM-friendly output."""
        from .colors import header

        print(header("FINAL SUMMARY"))

        earned, maximum = self.total_score()
        percentage = (earned / maximum * 100) if maximum > 0 else 0

        # Color based on score
        if percentage >= 80:
            color = Colors.GREEN
            status = "HEALTHY"
        elif percentage >= 50:
            color = Colors.YELLOW
            status = "NEEDS IMPROVEMENT"
        else:
            color = Colors.RED
            status = "CRITICAL ISSUES"

        # Test Results Table
        print(subheader("Test Results"))
        print(f"{'Test':<35} {'Score':>10} {'Status':>10}")
        print("-" * 60)

        for t in self.tests:
            status_str = (
                f"{Colors.GREEN}PASS{Colors.ENDC}"
                if t["passed"]
                else f"{Colors.RED}FAIL{Colors.ENDC}"
            )
            score_str = f"{t['score']:.2f}/{t['max_score']:.1f}"
            pct = (t["score"] / t["max_score"] * 100) if t["max_score"] > 0 else 0
            bar_len = int(pct / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            print(f"  {t['name']:<33} {score_str:>10} [{bar}] {status_str}")

        print("-" * 60)
        print(
            f"\n{Colors.BOLD}Overall Score: {color}{earned:.1f}/{maximum:.1f} ({percentage:.0f}%){Colors.ENDC}"
        )

        # Grade
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

        # Performance Metrics
        if self.timings:
            print(subheader("Performance Metrics"))
            for name, duration in self.timings.items():
                if duration < 0.01:
                    time_str = f"{duration*1000000:.1f}μs"
                elif duration < 1:
                    time_str = f"{duration*1000:.1f}ms"
                else:
                    time_str = f"{duration:.2f}s"
                print(f"  {name:<30} {time_str:>15}")

        # Issues Summary
        if self.issues:
            print(subheader("Identified Issues"))

            # Group by severity
            by_severity = defaultdict(list)
            for issue in self.issues:
                by_severity[issue["severity"]].append(issue)

            severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            severity_colors = {
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

        # Recommendations
        if self.recommendations:
            print(subheader("Recommendations"))
            sorted_recs = sorted(self.recommendations, key=lambda x: x["priority"])
            for i, rec in enumerate(sorted_recs, 1):
                print(f"  {i}. {Colors.BOLD}{rec['action']}{Colors.ENDC}")
                print(f"     {dim(rec['reason'])}")

        return percentage
