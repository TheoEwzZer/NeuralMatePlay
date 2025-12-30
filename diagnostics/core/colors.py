"""Terminal colors and formatting utilities."""


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY outputs."""
        cls.HEADER = ""
        cls.BLUE = ""
        cls.CYAN = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.ENDC = ""
        cls.BOLD = ""
        cls.DIM = ""


def ok(msg):
    return f"{Colors.GREEN}[OK]{Colors.ENDC} {msg}"


def fail(msg):
    return f"{Colors.RED}[FAIL]{Colors.ENDC} {msg}"


def warn(msg):
    return f"{Colors.YELLOW}[WARN]{Colors.ENDC} {msg}"


def info(msg):
    return f"{Colors.BLUE}[INFO]{Colors.ENDC} {msg}"


def header(title):
    line = "=" * 70
    return f"\n{Colors.BOLD}{Colors.CYAN}{line}\n{title}\n{line}{Colors.ENDC}"


def subheader(title):
    return f"\n{Colors.BOLD}{title}{Colors.ENDC}\n" + "-" * 50


def dim(msg):
    return f"{Colors.DIM}{msg}{Colors.ENDC}"
