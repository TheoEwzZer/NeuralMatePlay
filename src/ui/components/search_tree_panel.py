"""
Search Tree visualization panel.

Interactive tree view of MCTS search results with collapsible nodes.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS


class SearchTreePanel(tk.Frame):
    """
    Interactive MCTS search tree visualization.

    Features:
    - Collapsible/expandable nodes
    - Color-coded Q-values (green=positive, red=negative)
    - Shows visits, Q-value, and prior for each node
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 280,
        height: int = 150,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, width=width, height=height, **kwargs)

        self.configure(
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

        # Allow panel to expand beyond initial size
        self.pack_propagate(False)

        self._tree_data: list[dict[str, Any]] = []
        self._expanded: set = set()  # Set of expanded node paths

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create panel widgets."""
        # Title bar
        title_frame = tk.Frame(self, bg=bg)
        title_frame.pack(fill=tk.X, padx=10, pady=(8, 5))

        self._title_label = tk.Label(
            title_frame,
            text="Search Tree",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
        )
        self._title_label.pack(side=tk.LEFT)

        # Expand/Collapse all buttons
        btn_frame = tk.Frame(title_frame, bg=bg)
        btn_frame.pack(side=tk.RIGHT)

        expand_btn = tk.Label(
            btn_frame,
            text="[+]",
            font=("Consolas", 9),
            fg=COLORS["accent"],
            bg=bg,
            cursor="hand2",
        )
        expand_btn.pack(side=tk.LEFT, padx=2)
        expand_btn.bind("<Button-1>", lambda e: self._expand_all())

        collapse_btn = tk.Label(
            btn_frame,
            text="[-]",
            font=("Consolas", 9),
            fg=COLORS["accent"],
            bg=bg,
            cursor="hand2",
        )
        collapse_btn.pack(side=tk.LEFT, padx=2)
        collapse_btn.bind("<Button-1>", lambda e: self._collapse_all())

        # Scrollable content
        container = tk.Frame(self, bg=bg)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for scrolling
        self._canvas = tk.Canvas(
            container,
            bg=bg,
            highlightthickness=0,
        )
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            container,
            orient=tk.VERTICAL,
            command=self._canvas.yview,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame for tree content
        self._tree_frame = tk.Frame(self._canvas, bg=bg)
        self._canvas_window = self._canvas.create_window(
            (0, 0),
            window=self._tree_frame,
            anchor="nw",
        )

        # Bind events
        self._tree_frame.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event: tk.Event) -> None:
        """Update scroll region."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Update frame width."""
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mousewheel."""
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_tree(
        self,
        tree_data: list[dict[str, Any]],
        depth: int = 0,
        top_n: int = 0,
        mate_in: int | None = None,
    ) -> None:
        """
        Update the tree display with new data.

        Args:
            tree_data: List of root nodes, each with keys:
                       san, visits, q_value, prior, children
            depth: Actual max depth of the search tree.
            top_n: Number of top moves shown per level.
            mate_in: Number of moves to mate, or None if no forced mate.
        """
        # Update title with depth, top_n, and mate info
        if mate_in == 1:
            mate_str = " - Mate"
        elif mate_in:
            mate_str = f" - Mate in {mate_in - 1}"
        else:
            mate_str = ""

        if depth > 0:
            self._title_label.config(
                text=f"Search Tree (depth: {depth}, top: {top_n}){mate_str}"
            )
        else:
            self._title_label.config(text=f"Search Tree{mate_str}")
        self._tree_data = tree_data
        # Keep first level expanded by default
        self._expanded = {node["san"] for node in tree_data}
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Rebuild the tree display."""
        # Clear existing
        for widget in self._tree_frame.winfo_children():
            widget.destroy()

        bg = COLORS["bg_secondary"]

        if not self._tree_data:
            no_data = tk.Label(
                self._tree_frame,
                text="No search data",
                font=("Segoe UI", 10),
                fg=COLORS["text_muted"],
                bg=bg,
            )
            no_data.pack(pady=20)
            return

        # Build tree
        for node in self._tree_data:
            self._create_node_widget(node, depth=0, path=node["san"])

    def _create_node_widget(
        self,
        node: dict[str, Any],
        depth: int,
        path: str,
    ) -> None:
        """Create a widget for a tree node."""
        bg = COLORS["bg_secondary"]
        indent = depth * 20

        # Node frame
        node_frame = tk.Frame(self._tree_frame, bg=bg)
        node_frame.pack(fill=tk.X, pady=1)

        # Indent spacer
        if indent > 0:
            spacer = tk.Frame(node_frame, bg=bg, width=indent)
            spacer.pack(side=tk.LEFT)

        # Tree branch character
        has_children = bool(node.get("children"))
        is_expanded = path in self._expanded

        if has_children:
            toggle_text = "▼" if is_expanded else "▶"
            toggle = tk.Label(
                node_frame,
                text=toggle_text,
                font=("Consolas", 8),
                fg=COLORS["accent"],
                bg=bg,
                cursor="hand2",
                width=2,
            )
            toggle.pack(side=tk.LEFT)
            toggle.bind("<Button-1>", lambda e, p=path: self._toggle_node(p))
        else:
            # Leaf node indicator
            leaf = tk.Label(
                node_frame,
                text="─",
                font=("Consolas", 8),
                fg=COLORS["text_muted"],
                bg=bg,
                width=2,
            )
            leaf.pack(side=tk.LEFT)

        # Move name
        san = node["san"]
        move_label = tk.Label(
            node_frame,
            text=san,
            font=("Consolas", 10, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
            width=6,
            anchor="w",
        )
        move_label.pack(side=tk.LEFT, padx=(0, 5))

        # Stats: N=visits
        visits = node["visits"]
        visits_label = tk.Label(
            node_frame,
            text=f"N={visits}",
            font=("Consolas", 9),
            fg=COLORS["text_secondary"],
            bg=bg,
        )
        visits_label.pack(side=tk.LEFT, padx=2)

        # Q-value with color
        q_value = node["q_value"]
        q_color = self._get_q_color(q_value)
        q_label = tk.Label(
            node_frame,
            text=f"Q={q_value:+.3f}",
            font=("Consolas", 9),
            fg=q_color,
            bg=bg,
        )
        q_label.pack(side=tk.LEFT, padx=2)

        # Prior
        prior = node["prior"]
        prior_label = tk.Label(
            node_frame,
            text=f"P={prior:.1%}",
            font=("Consolas", 9),
            fg=COLORS["prior_bar"],
            bg=bg,
        )
        prior_label.pack(side=tk.LEFT, padx=2)

        # Forced mate indicators
        our_mate_in = node.get("our_mate_in")
        opponent_mate_in = node.get("opponent_mate_in")
        if our_mate_in is not None:
            # Winning mate (green)
            mate_text = f"#{our_mate_in}" if our_mate_in > 1 else "#1"
            mate_label = tk.Label(
                node_frame,
                text=mate_text,
                font=("Consolas", 9, "bold"),
                fg=COLORS["q_value_positive"],
                bg=bg,
            )
            mate_label.pack(side=tk.LEFT, padx=2)
        elif opponent_mate_in is not None:
            # Losing mate (red)
            mate_text = f"#-{opponent_mate_in}" if opponent_mate_in > 0 else "#"
            mate_label = tk.Label(
                node_frame,
                text=mate_text,
                font=("Consolas", 9, "bold"),
                fg=COLORS["q_value_negative"],
                bg=bg,
            )
            mate_label.pack(side=tk.LEFT, padx=2)

        # Hover effect
        def on_enter(e, f=node_frame):
            f.configure(bg=COLORS["move_hover"])
            for child in f.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=COLORS["move_hover"])

        def on_leave(e, f=node_frame):
            f.configure(bg=bg)
            for child in f.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=bg)

        node_frame.bind("<Enter>", on_enter)
        node_frame.bind("<Leave>", on_leave)

        # Recursively add children if expanded
        if has_children and is_expanded:
            for child in node["children"]:
                child_path = f"{path}/{child['san']}"
                self._create_node_widget(child, depth + 1, child_path)

    def _get_q_color(self, q_value: float) -> str:
        """Get color for Q-value."""
        if q_value > 0.05:
            return COLORS["q_value_positive"]
        elif q_value < -0.05:
            return COLORS["q_value_negative"]
        return COLORS["text_secondary"]

    def _toggle_node(self, path: str) -> None:
        """Toggle expand/collapse for a node."""
        if path in self._expanded:
            self._expanded.discard(path)
            # Also collapse all children
            self._expanded = {p for p in self._expanded if not p.startswith(path + "/")}
        else:
            self._expanded.add(path)
        self._rebuild_display()

    def _expand_all(self) -> None:
        """Expand all nodes."""
        self._expand_nodes_recursive(self._tree_data, "")
        self._rebuild_display()

    def _expand_nodes_recursive(self, nodes: list[dict], parent_path: str) -> None:
        """Recursively expand all nodes."""
        for node in nodes:
            path = f"{parent_path}/{node['san']}" if parent_path else node["san"]
            self._expanded.add(path)
            if node.get("children"):
                self._expand_nodes_recursive(node["children"], path)

    def _collapse_all(self) -> None:
        """Collapse all nodes."""
        self._expanded.clear()
        self._rebuild_display()

    def clear(self) -> None:
        """Clear the tree display."""
        self._tree_data = []
        self._expanded.clear()
        self._rebuild_display()
