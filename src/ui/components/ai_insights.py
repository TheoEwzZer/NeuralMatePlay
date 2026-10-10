"""
Beginner-friendly AI analysis widgets.

Turns raw MCTS numbers (visits, Q-values, priors, search tree) into
plain-language panels:

- WinChancePanel:      "Who is winning?" with a White / Draw / Black bar
- CandidateMovesPanel: moves the AI considered, how much it studied each
                       one and the resulting winning chances
- AIPlanPanel:         the line the AI expects ("it plays X, expects Y")
- AdvantageGraph:      White's winning chances over the whole game
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

import chess

try:
    from ..styles import (
        COLORS,
        FONTS,
        FRENCH_PIECE_NAMES,
        Card,
        Tooltip,
        san_to_words,
    )
except ImportError:
    from src.ui.styles import (
        COLORS,
        FONTS,
        FRENCH_PIECE_NAMES,
        Card,
        Tooltip,
        san_to_words,
    )


def expected_score(wdl: Any) -> float:
    """Expected score in [0, 1] from a [win, draw, loss] triple (draw = half)."""
    return float(wdl[0]) + 0.5 * float(wdl[1])


def chance_color(score: float) -> str:
    """Green when good, red when bad, grey when balanced."""
    if score >= 0.56:
        return COLORS["success"]
    if score <= 0.44:
        return COLORS["error"]
    return COLORS["text_secondary"]


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """
    Describe a move in plain French, e.g. "Cavalier en f3", "Petit roque".

    Args:
        board: Position before the move.
        move: The move to describe.
    """
    try:
        if board.is_kingside_castling(move):
            return "Petit roque (met le roi à l'abri)"
        if board.is_queenside_castling(move):
            return "Grand roque (met le roi à l'abri)"

        piece = board.piece_at(move.from_square)
        name = FRENCH_PIECE_NAMES.get(piece.piece_type, "Pièce") if piece else "Pièce"
        target = chess.square_name(move.to_square)

        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured is None:
                text = f"{name} prend le pion en passant"
            else:
                victim = FRENCH_PIECE_NAMES.get(captured.piece_type, "pièce").lower()
                text = f"{name} prend {'le' if victim != 'dame' else 'la'} {victim} en {target}"
        else:
            text = f"{name} en {target}"

        if move.promotion:
            promo = FRENCH_PIECE_NAMES.get(move.promotion, "Dame").lower()
            text += f", promotion en {promo}"
        if board.gives_check(move):
            text += " — échec !"
        return text
    except Exception:
        return ""


class WinChancePanel(Card):
    """Card answering "who is winning?" with a stacked W/D/L bar."""

    BAR_HEIGHT = 30

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(
            parent,
            title="Qui gagne ?",
            help_text=(
                "L'estimation de l'IA sur l'issue de la partie si les deux camps "
                "jouent bien à partir d'ici.\n\nBlanc = chances des Blancs, gris = "
                "chances de nulle, foncé = chances des Noirs.\n\nLa courbe montre "
                "l'évolution au fil de la partie : en haut, les Blancs mènent ; en "
                "bas, les Noirs mènent. Survolez-la pour voir le détail."
            ),
            **kwargs,
        )
        self._white_name = "les Blancs"
        self._black_name = "les Noirs"
        self._wdl: tuple[float, float, float] | None = None
        self._mate_for: chess.Color | None = None
        self._result_text: str | None = None

        body = self.body
        bg = self.body.cget("bg")

        self._headline = tk.Label(
            body,
            text="Position équilibrée",
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=bg,
            anchor="w",
        )
        self._headline.pack(fill=tk.X)

        self._canvas = tk.Canvas(
            body, height=self.BAR_HEIGHT, bg=bg, highlightthickness=0
        )
        self._canvas.pack(fill=tk.X, pady=(8, 6))
        self._canvas.bind("<Configure>", lambda e: self._draw_bar())

        self._subtitle = tk.Label(
            body,
            text="L'IA donnera son avis après son premier coup.",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=bg,
            anchor="w",
            justify=tk.LEFT,
            wraplength=540,
        )
        self._subtitle.pack(fill=tk.X)

    def set_names(self, white_name: str, black_name: str) -> None:
        """Set how each side is called in sentences ("vous", "l'IA", ...)."""
        self._white_name = white_name
        self._black_name = black_name
        self._update_headline()

    def update_wdl(
        self,
        white_win: float,
        draw: float,
        black_win: float,
        subtitle: str = "",
        mate_for: chess.Color | None = None,
    ) -> None:
        """
        Show new winning chances (from White's point of view).

        Args:
            white_win: Probability that White wins
            draw: Probability of a draw
            black_win: Probability that Black wins
            subtitle: Small explanatory line under the bar
            mate_for: Side that has a forced mate, if any
        """
        total = max(1e-6, white_win + draw + black_win)
        self._wdl = (white_win / total, draw / total, black_win / total)
        self._mate_for = mate_for
        self._result_text = None
        self._subtitle.configure(text=subtitle)
        self._update_headline()
        self._draw_bar()

    def set_result(
        self, winner: chess.Color | None, headline: str, detail: str
    ) -> None:
        """
        Show the final result of a finished game.

        Args:
            winner: Winning color, or None for a draw
            headline: Big result line
            detail: Small explanation under the bar
        """
        if winner is None:
            self._wdl = (0.0, 1.0, 0.0)
        else:
            self._wdl = (1.0, 0.0, 0.0) if winner == chess.WHITE else (0.0, 0.0, 1.0)
        self._mate_for = None
        self._result_text = headline
        self._subtitle.configure(text=detail)
        self._update_headline()
        self._draw_bar()

    def _update_headline(self) -> None:
        """Compute the plain-language headline."""
        if self._result_text:
            self._headline.configure(text=self._result_text, fg=COLORS["warning"])
            return

        if self._wdl is None:
            self._headline.configure(
                text="Position équilibrée", fg=COLORS["text_primary"]
            )
            return

        if self._mate_for is not None:
            who = (
                self._white_name if self._mate_for == chess.WHITE else self._black_name
            )
            self._headline.configure(
                text=f"Mat forcé pour {who} !", fg=COLORS["warning"]
            )
            return

        score = expected_score(self._wdl)
        gap = abs(score - 0.5)
        who = self._white_name if score > 0.5 else self._black_name
        if gap < 0.06:
            text = "Position équilibrée"
        elif gap < 0.15:
            text = f"Léger avantage pour {who}"
        elif gap < 0.30:
            text = f"Net avantage pour {who}"
        elif gap < 0.42:
            text = f"Gros avantage pour {who}"
        else:
            text = f"Victoire quasi certaine pour {who}"
        self._headline.configure(
            text=text[0].upper() + text[1:], fg=COLORS["text_primary"]
        )

    def _draw_bar(self) -> None:
        """Draw the stacked White / Draw / Black bar."""
        c = self._canvas
        c.delete("all")
        width = max(10, c.winfo_width())
        h = self.BAR_HEIGHT

        if self._wdl is None:
            c.create_rectangle(0, 0, width, h, fill=COLORS["bg_tertiary"], outline="")
            c.create_text(
                width // 2,
                h // 2,
                text="En attente de l'analyse…",
                font=FONTS["caption"],
                fill=COLORS["text_muted"],
            )
            return

        segments = [
            ("Blancs", self._wdl[0], COLORS["white_side"], "#1b1e24"),
            ("Nulle", self._wdl[1], COLORS["draw_side"], "#ffffff"),
            ("Noirs", self._wdl[2], COLORS["black_side"], "#eceef2"),
        ]
        x = 0.0
        for name, frac, fill, text_color in segments:
            seg_w = frac * width
            if seg_w <= 0:
                continue
            c.create_rectangle(x, 0, x + seg_w, h, fill=fill, outline="")
            pct = f"{round(frac * 100)} %"
            # Full label when it fits, percentage only otherwise
            text = f"{name} {pct}" if seg_w > 100 else (pct if seg_w > 40 else "")
            if text:
                c.create_text(
                    x + seg_w / 2,
                    h / 2,
                    text=text,
                    font=FONTS["small_bold"],
                    fill=text_color,
                )
            x += seg_w

    def clear(self) -> None:
        """Reset to the initial state."""
        self._wdl = None
        self._mate_for = None
        self._result_text = None
        self._subtitle.configure(text="L'IA donnera son avis après son premier coup.")
        self._update_headline()
        self._draw_bar()


class CandidateMovesPanel(Card):
    """Card listing the moves the AI considered, in plain words."""

    MAX_MOVES = 4

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(
            parent,
            title="Coups envisagés par l'IA",
            help_text=(
                "Avant de jouer, l'IA imagine des milliers de suites possibles.\n\n"
                "• La barre bleue montre où elle a passé le plus de temps : "
                "plus elle est longue, plus l'IA trouve le coup intéressant.\n"
                "• Le pourcentage à droite indique ses chances de gagner si elle "
                "joue ce coup (une nulle compte pour moitié)."
            ),
            **kwargs,
        )
        self._rows_frame = tk.Frame(self.body, bg=self.body.cget("bg"))
        self._rows_frame.pack(fill=tk.BOTH, expand=True)
        self.clear()

    def clear(self) -> None:
        """Show the empty state."""
        for w in self._rows_frame.winfo_children():
            w.destroy()
        tk.Label(
            self._rows_frame,
            text="Les idées de l'IA apparaîtront ici dès qu'elle aura joué.",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=self._rows_frame.cget("bg"),
            wraplength=360,
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, pady=6)

    def update_moves(
        self,
        board: chess.Board,
        stats: list[dict[str, Any]],
        mover_name: str = "l'IA",
    ) -> None:
        """
        Show the top candidate moves.

        Args:
            board: Position the AI searched (before its move)
            stats: MCTS root statistics, sorted by visits (best first)
            mover_name: How to call the side that searched
        """
        for w in self._rows_frame.winfo_children():
            w.destroy()

        moves = [s for s in stats if s.get("visits", 0) > 0][: self.MAX_MOVES]
        if not moves:
            self.clear()
            return

        total = sum(s.get("visits", 0) for s in stats) or 1
        top_share = moves[0]["visits"] / total
        for i, stat in enumerate(moves):
            self._create_row(
                board, stat, i, stat["visits"] / total, top_share, mover_name
            )

    def _create_row(
        self,
        board: chess.Board,
        stat: dict[str, Any],
        index: int,
        share: float,
        top_share: float,
        mover_name: str,
    ) -> None:
        """Create one candidate move row."""
        bg = COLORS["bg_tertiary"] if index == 0 else self._rows_frame.cget("bg")
        row = tk.Frame(self._rows_frame, bg=bg)
        row.pack(fill=tk.X, pady=(0, 3))

        inner = tk.Frame(row, bg=bg)
        inner.pack(fill=tk.X, padx=8, pady=5)

        move: chess.Move = stat["move"]
        try:
            san = san_to_words(board.san(move))
        except Exception:
            san = move.uci()

        top = tk.Frame(inner, bg=bg)
        top.pack(fill=tk.X)

        move_label = tk.Label(
            top,
            text=san,
            font=("Segoe UI Semibold", 12),
            fg=COLORS["text_primary"],
            bg=bg,
            anchor="w",
        )
        move_label.pack(side=tk.LEFT, padx=(0, 10))

        # Full sentence (captured piece, castling purpose...) on hover
        desc = describe_move(board, move)
        if desc:
            Tooltip(move_label, desc, delay=200)

        if index == 0:
            tk.Label(
                top,
                text=" choisi ",
                font=FONTS["caption_bold"],
                fg="#ffffff",
                bg=COLORS["accent"],
            ).pack(side=tk.LEFT, padx=(0, 6))

        # Special situations, in words
        tag_text, tag_color = "", ""
        if stat.get("our_mate_in") is not None:
            n = stat["our_mate_in"]
            tag_text = "échec et mat !" if n <= 1 else f"mat en {n}"
            tag_color = COLORS["success"]
        elif stat.get("opponent_mate_in") is not None:
            tag_text = "laisse un mat"
            tag_color = COLORS["error"]
        elif stat.get("leads_to_draw_repetition"):
            tag_text = "nulle par répétition"
            tag_color = COLORS["warning"]
        if tag_text:
            tk.Label(
                top,
                text=f" {tag_text} ",
                font=FONTS["caption_bold"],
                fg=tag_color,
                bg=COLORS["bg_primary"],
            ).pack(side=tk.LEFT, padx=(0, 6))

        # Winning chances for the side that would play it
        wdl = stat.get("wdl")
        if wdl is not None:
            score = expected_score(wdl)
            chance = tk.Label(
                top,
                text=f"{round(score * 100)} %",
                font=FONTS["body_bold"],
                fg=chance_color(score),
                bg=bg,
            )
            chance.pack(side=tk.RIGHT)
            Tooltip(
                chance,
                f"Chances de gagner pour {mover_name} avec ce coup : "
                f"victoire {round(float(wdl[0]) * 100)} %, "
                f"nulle {round(float(wdl[1]) * 100)} %, "
                f"défaite {round(float(wdl[2]) * 100)} %.",
                delay=200,
            )

        # "Attention" bar: share of the search spent on this move
        bar_row = tk.Frame(inner, bg=bg)
        bar_row.pack(fill=tk.X, pady=(3, 0))
        bar = tk.Canvas(bar_row, height=6, bg=bg, highlightthickness=0)
        bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            bar_row,
            text=f"{round(share * 100)} % de la réflexion",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=bg,
            width=18,
            anchor="e",
        ).pack(side=tk.RIGHT)

        rel = share / top_share if top_share > 0 else 0

        def draw(event: tk.Event, c: tk.Canvas = bar, r: float = rel) -> None:
            c.delete("all")
            c.create_rectangle(
                0, 0, event.width, 6, fill=COLORS["bg_hover"], outline=""
            )
            c.create_rectangle(
                0, 0, max(3, event.width * r), 6, fill=COLORS["visit_bar"], outline=""
            )

        bar.bind("<Configure>", draw)


class AIPlanPanel(Card):
    """Card explaining the line the AI expects, without N/Q/P jargon."""

    MAX_PLIES = 6

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(
            parent,
            title="Ce que l'IA prévoit",
            help_text=(
                "La suite de coups que l'IA juge la plus probable : son coup, la "
                "réponse qu'elle attend de l'adversaire, puis sa propre réponse, etc.\n\n"
                "Survolez un coup pour le voir décrit en mots."
            ),
            **kwargs,
        )
        bg = self.body.cget("bg")
        self._bg = bg

        self._mate_banner = tk.Label(
            self.body,
            text="",
            font=FONTS["small_bold"],
            fg=COLORS["bg_primary"],
            bg=COLORS["warning"],
            anchor="w",
            padx=8,
            pady=4,
        )

        self._sentence = tk.Label(
            self.body,
            text="",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=bg,
            wraplength=540,
            justify=tk.LEFT,
            anchor="w",
        )
        self._sentence.pack(fill=tk.X)

        self._chips = tk.Frame(self.body, bg=bg)
        self._chips.pack(fill=tk.X, pady=(8, 0))

        self._alternatives = tk.Label(
            self.body,
            text="",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=bg,
            wraplength=540,
            justify=tk.LEFT,
            anchor="w",
        )
        self._alternatives.pack(fill=tk.X, pady=(8, 0))
        self.clear()

    def clear(self) -> None:
        """Show the empty state."""
        self._mate_banner.pack_forget()
        for w in self._chips.winfo_children():
            w.destroy()
        self._sentence.configure(
            text="Quand l'IA aura réfléchi, vous verrez ici la suite de coups qu'elle anticipe."
        )
        self._alternatives.configure(text="")

    def update_plan(
        self,
        board: chess.Board,
        tree_data: list[dict[str, Any]],
        depth: int = 0,
        mate_in: int | None = None,
        mover_name: str = "L'IA",
        opponent_name: str = "l'adversaire",
    ) -> None:
        """
        Show the principal line of the search tree in plain language.

        Args:
            board: Position the AI searched (before its move)
            tree_data: Nested search tree data (most visited first)
            depth: Maximum depth reached by the search
            mate_in: Forced mate length, if any
            mover_name: How to call the side that searched
            opponent_name: How to call the other side
        """
        for w in self._chips.winfo_children():
            w.destroy()

        if not tree_data:
            self.clear()
            return

        # Principal variation: follow the most visited child at each level
        line: list[str] = []
        node: dict[str, Any] | None = tree_data[0]
        while node is not None and len(line) < self.MAX_PLIES:
            line.append(node["san"])
            children = node.get("children") or []
            node = children[0] if children else None

        # Mate banner
        if mate_in:
            text = (
                "♛  Échec et mat en un coup !"
                if mate_in <= 1
                else f"♛  Mat forcé trouvé : {mate_in} coups pour conclure"
            )
            self._mate_banner.configure(text=text)
            self._mate_banner.pack(fill=tk.X, pady=(0, 8), before=self._sentence)
        else:
            self._mate_banner.pack_forget()

        # Sentence
        # Lowercase inside a sentence: "L'IA joue fou c5"
        fr = [san_to_words(s)[0].lower() + san_to_words(s)[1:] for s in line]
        sentence = f"{mover_name} joue {fr[0]}"
        if len(fr) > 1:
            verb = "répondiez" if opponent_name == "vous" else "réponde"
            sentence += f", s'attend à ce que {opponent_name} {verb} {fr[1]}"
        if len(fr) > 2:
            sentence += f", puis continuerait avec {fr[2]}"
        sentence += "."
        if depth > 0:
            sentence += f"\nElle a anticipé jusqu'à {depth} coups à l'avance."
        self._sentence.configure(text=sentence)

        # Chips with move numbers, each describing the move on hover.
        # They flow onto several rows when the line is long.
        max_row_width = max(300, self._chips.winfo_width() or 540)
        chip_row = tk.Frame(self._chips, bg=self._bg)
        chip_row.pack(fill=tk.X, anchor="w")
        row_width = 0
        temp = board.copy(stack=False)
        for i, san in enumerate(line):
            is_mover = i % 2 == 0
            number = ""
            if temp.turn == chess.WHITE:
                number = f"{temp.fullmove_number}."
            elif i == 0:
                number = f"{temp.fullmove_number}…"

            desc = ""
            try:
                move = temp.parse_san(san)
                desc = describe_move(temp, move)
                temp.push(move)
            except Exception:
                pass

            chip_bg = COLORS["accent_soft"] if is_mover else COLORS["bg_tertiary"]
            chip = tk.Label(
                chip_row,
                text=f"{number} {san_to_words(san)}".strip(),
                font=FONTS["small_bold"],
                fg=COLORS["text_primary"],
                bg=chip_bg,
                padx=7,
                pady=3,
            )
            chip_width = chip.winfo_reqwidth() + 4
            if row_width and row_width + chip_width > max_row_width:
                # Start a new row and move the chip there
                chip.destroy()
                chip_row = tk.Frame(self._chips, bg=self._bg)
                chip_row.pack(fill=tk.X, anchor="w", pady=(4, 0))
                row_width = 0
                chip = tk.Label(
                    chip_row,
                    text=f"{number} {san_to_words(san)}".strip(),
                    font=FONTS["small_bold"],
                    fg=COLORS["text_primary"],
                    bg=chip_bg,
                    padx=7,
                    pady=3,
                )
            chip.pack(side=tk.LEFT, padx=(0, 4))
            row_width += chip_width
            who = (
                mover_name if is_mover else opponent_name[0].upper() + opponent_name[1:]
            )
            Tooltip(chip, f"{who} : {desc}" if desc else who, delay=150)

        # Other ideas
        alts = []
        for alt in tree_data[1:3]:
            reply = (alt.get("children") or [{}])[0].get("san")
            text = san_to_words(alt["san"])
            if reply:
                text += f" (réponse attendue : {san_to_words(reply)})"
            alts.append(text)
        if alts:
            self._alternatives.configure(
                text="Autres idées étudiées : " + " · ".join(alts)
            )
        else:
            self._alternatives.configure(text="")


class AdvantageGraph(tk.Canvas):
    """
    Area chart of White's winning chances over the game.

    The light area is White's share, the dark area is Black's share:
    the higher the curve, the better for White.
    """

    MARGIN_LEFT = 46
    MARGIN_RIGHT = 8
    MARGIN_TOP = 8
    MARGIN_BOTTOM = 18

    def __init__(self, parent: tk.Widget, height: int = 120, **kwargs) -> None:
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, height=height, bg=bg, highlightthickness=0, **kwargs)
        self._bg = bg
        # (ply, white_score)
        self._points: list[tuple[int, float]] = []
        self._hover_x: int | None = None
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)

    # ---------- data API ----------

    def add_point(self, ply: int, white_score: float) -> None:
        """Record White's winning chances after a given ply."""
        self._points = [p for p in self._points if p[0] < ply]
        self._points.append((ply, max(0.0, min(1.0, white_score))))
        self._draw()

    def trim(self, ply: int) -> None:
        """Forget points after a given ply (after taking moves back)."""
        self._points = [p for p in self._points if p[0] <= ply]
        self._draw()

    def last_score(self) -> float | None:
        """Most recent White score, if any."""
        return self._points[-1][1] if self._points else None

    def clear(self) -> None:
        """Reset the graph."""
        self._points = []
        self._draw()

    # ---------- drawing ----------

    def _geometry(self) -> tuple[int, int, int, int, int]:
        w = max(50, self.winfo_width())
        h = max(40, self.winfo_height())
        left = self.MARGIN_LEFT
        right = w - self.MARGIN_RIGHT
        top = self.MARGIN_TOP
        bottom = h - self.MARGIN_BOTTOM
        max_ply = max(20, max((p[0] for p in self._points), default=0))
        return left, right, top, bottom, max_ply

    def _xy(self, ply: int, score: float) -> tuple[float, float]:
        left, right, top, bottom, max_ply = self._geometry()
        x = left + (right - left) * ply / max_ply
        y = bottom - (bottom - top) * score
        return x, y

    def _draw(self) -> None:
        self.delete("all")
        left, right, top, bottom, max_ply = self._geometry()
        mid = (top + bottom) / 2

        # Black share background
        self.create_rectangle(
            left, top, right, bottom, fill=COLORS["black_side"], outline=""
        )

        if self._points:
            pts = [(0, 0.5)] + self._points
            poly: list[float] = [left, bottom]
            for ply, score in pts:
                poly.extend(self._xy(ply, score))
            last_x = self._xy(pts[-1][0], pts[-1][1])[0]
            poly.extend([last_x, bottom])
            self.create_polygon(poly, fill="#cfd5dc", outline="")

            line: list[float] = []
            for ply, score in pts:
                line.extend(self._xy(ply, score))
            if len(line) >= 4:
                self.create_line(*line, fill=COLORS["accent"], width=2)
            x, y = self._xy(*pts[-1])
            self.create_oval(
                x - 4, y - 4, x + 4, y + 4, fill=COLORS["accent"], outline="#ffffff"
            )
        else:
            self.create_text(
                (left + right) / 2,
                (top + mid) / 2,
                text="La courbe se dessinera au fil de la partie",
                font=FONTS["caption"],
                fill=COLORS["text_secondary"],
            )

        # 50% line
        self.create_line(
            left, mid, right, mid, fill=COLORS["graph_marker"], dash=(3, 3)
        )

        # Labels
        # Curve high = good for White, low = good for Black
        self.create_text(
            left - 6,
            top + 2,
            text="Blancs",
            anchor="ne",
            font=FONTS["caption"],
            fill=COLORS["text_secondary"],
        )
        self.create_text(
            left - 6,
            bottom - 2,
            text="Noirs",
            anchor="se",
            font=FONTS["caption"],
            fill=COLORS["text_secondary"],
        )
        self.create_text(
            left,
            bottom + 3,
            text="début",
            anchor="nw",
            font=FONTS["caption"],
            fill=COLORS["text_muted"],
        )
        self.create_text(
            right,
            bottom + 3,
            text=f"coup {(max_ply + 1) // 2}",
            anchor="ne",
            font=FONTS["caption"],
            fill=COLORS["text_muted"],
        )

        self._draw_hover()

    def _draw_hover(self) -> None:
        if self._hover_x is None or not self._points:
            return
        left, right, top, bottom, max_ply = self._geometry()
        ply_at = (self._hover_x - left) / max(1, right - left) * max_ply
        ply, score = min(self._points, key=lambda p: abs(p[0] - ply_at))
        x, y = self._xy(ply, score)
        self.create_line(x, top, x, bottom, fill=COLORS["text_secondary"], dash=(2, 2))
        self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=COLORS["warning"], outline="")

        if abs(score - 0.5) < 0.06:
            verdict = "équilibré"
        elif score > 0.5:
            verdict = f"Blancs {round(score * 100)} %"
        else:
            verdict = f"Noirs {round((1 - score) * 100)} %"
        text = f"Coup {(ply + 1) // 2} · {verdict}"
        tx = min(max(x, left + 60), right - 60)
        item = self.create_text(
            tx,
            top + 10,
            text=text,
            font=FONTS["caption_bold"],
            fill=COLORS["text_primary"],
        )
        bbox = self.bbox(item)
        if bbox:
            rect = self.create_rectangle(
                bbox[0] - 6,
                bbox[1] - 3,
                bbox[2] + 6,
                bbox[3] + 3,
                fill=COLORS["bg_tertiary"],
                outline=COLORS["border_light"],
            )
            self.tag_raise(item, rect)

    def _on_motion(self, event: tk.Event) -> None:
        self._hover_x = event.x
        self._draw()

    def _on_leave(self, event: tk.Event) -> None:
        self._hover_x = None
        self._draw()
