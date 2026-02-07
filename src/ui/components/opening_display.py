"""
Opening name display widget.

Shows the current opening name and ECO code.
"""

import tkinter as tk
from typing import Optional, Dict

import chess

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS

ECO_OPENINGS: dict[str, Dict[str, str]] = {
    # A - Volume A
    "b4": {"eco": "A00", "name": "Polish (Sokolsky) opening"},
    "g3": {"eco": "A00", "name": "Benko's opening"},
    "g4": {"eco": "A00", "name": "Grob's attack"},
    "h3": {"eco": "A00", "name": "Clemenz (Mead's, Basman's or de Klerk's) opening"},
    "Nh3": {"eco": "A00", "name": "Amar (Paris) opening"},
    "Nc3": {"eco": "A00", "name": "Dunst (Sleipner, Heinrichsen) opening"},
    "a3": {"eco": "A00", "name": "Anderssen's opening"},
    "a4": {"eco": "A00", "name": "Ware (Meadow Hay) opening"},
    "c3": {"eco": "A00", "name": "Saragossa opening"},
    "d3": {"eco": "A00", "name": "Mieses opening"},
    "e3": {"eco": "A00", "name": "Van't Kruijs opening"},
    "f3": {"eco": "A00", "name": "Gedult's opening"},
    "h4": {"eco": "A00", "name": "Anti-Borg (Desprez) opening"},
    "Na3": {"eco": "A00", "name": "Durkin's attack"},
    "b4 Nh6": {"eco": "A00", "name": "Polish, Tuebingen variation"},
    "b4 c6": {"eco": "A00", "name": "Polish, Outflank variation"},
    "g3 h5": {"eco": "A00", "name": "Lasker simul special"},
    "Nc3 e5": {"eco": "A00", "name": "Dunst (Sleipner,Heinrichsen) opening"},
    "d3 e5": {"eco": "A00", "name": "Mieses opening"},
    "g3 e5 Nf3": {"eco": "A00", "name": "Benko's opening, reversed Alekhine"},
    "h3 e5 a3": {"eco": "A00", "name": "Global opening"},
    "Nc3 e5 a3": {"eco": "A00", "name": "Battambang opening"},
    "a4 e5 h4": {"eco": "A00", "name": "Crab opening"},
    "d3 e5 Nd2": {"eco": "A00", "name": "Valencia opening"},
    "f3 e5 Kf2": {"eco": "A00", "name": "Hammerschlag (Fried fox/Pork chop opening)"},
    "g4 d5 Bg2 c6 g5": {"eco": "A00", "name": "Grob, spike attack"},
    "g4 d5 Bg2 Bxg4 c4": {"eco": "A00", "name": "Grob, Fritz gambit"},
    "d3 c5 Nc3 Nc6 g3": {"eco": "A00", "name": "Venezolana opening"},
    "g4 d5 Bg2 Bxg4 c4 d4": {"eco": "A00", "name": "Grob, Romford counter-gambit"},
    "Nc3 c5 d4 cxd4 Qxd4 Nc6 Qh4": {"eco": "A00", "name": "Novosibirsk opening"},
    "Nh3 d5 g3 e5 f4 Bxh3 Bxh3 exf4": {"eco": "A00", "name": "Amar gambit"},
    "e3 e5 c4 d6 Nc3 Nc6 b3 Nf6": {"eco": "A00", "name": "Amsterdam attack"},
    "b3": {"eco": "A01", "name": "Nimzovich-Larsen attack"},
    "b3 e5": {"eco": "A01", "name": "Nimzovich-Larsen attack, modern variation"},
    "b3 Nf6": {"eco": "A01", "name": "Nimzovich-Larsen attack, Indian variation"},
    "b3 d5": {"eco": "A01", "name": "Nimzovich-Larsen attack, classical variation"},
    "b3 c5": {"eco": "A01", "name": "Nimzovich-Larsen attack, English variation"},
    "b3 f5": {"eco": "A01", "name": "Nimzovich-Larsen attack, Dutch variation"},
    "b3 b5": {"eco": "A01", "name": "Nimzovich-Larsen attack, Polish variation"},
    "b3 b6": {"eco": "A01", "name": "Nimzovich-Larsen attack, symmetrical variation"},
    "f4": {"eco": "A02", "name": "Bird's opening"},
    "Nf3": {"eco": "A04", "name": "Reti opening"},
    "c4": {"eco": "A10", "name": "English opening"},
    "d4": {"eco": "A40", "name": "Queen's pawn"},
    "d4 d6 c4 g6 Nc3 Bg7 e4": {"eco": "A42", "name": "Modern defence, Averbakh system"},
    "d4 d6 c4 g6 Nc3 Bg7 e4 f5": {
        "eco": "A42",
        "name": "Modern defence, Averbakh system, Randspringer variation",
    },
    "d4 d6 c4 g6 Nc3 Bg7 e4 Nc6": {
        "eco": "A42",
        "name": "Modern defence, Averbakh system, Kotov variation",
    },
    "d4 d6 c4 g6 Nc3 Bg7 e4 c5 Nf3 Qa5": {"eco": "A42", "name": "Pterodactyl defence"},
    "d4 c5": {"eco": "A43", "name": "Old Benoni defence"},
    "d4 Nf6": {"eco": "A45", "name": "Queen's pawn game"},
    "d4 Nf6 Nf3 b6": {"eco": "A47", "name": "Queen's Indian defence"},
    "d4 Nf6 Nf3 b6 g3 Bb7 Bg2 c5": {
        "eco": "A47",
        "name": "Queen's Indian, Marienbad system",
    },
    "d4 Nf6 Nf3 b6 g3 Bb7 Bg2 c5 c4 cxd4 Qxd4": {
        "eco": "A47",
        "name": "Queen's Indian, Marienbad system, Berg variation",
    },
    "d4 Nf6 Nf3 g6": {"eco": "A48", "name": "King's Indian, East Indian defence"},
    "d4 Nf6 c4": {"eco": "A50", "name": "Queen's pawn game"},
    "d4 Nf6 c4 Nc6": {"eco": "A50", "name": "Kevitz-Trajkovich defence"},
    "d4 Nf6 c4 b6": {"eco": "A50", "name": "Queen's Indian accelerated"},
    "d4 Nf6 c4 e5": {"eco": "A51", "name": "Budapest defence"},
    "d4 Nf6 c4 d6": {"eco": "A53", "name": "Old Indian defence"},
    "d4 Nf6 c4 c5": {"eco": "A56", "name": "Benoni defence"},
    "d4 Nf6 c4 c5 d5 d6": {"eco": "A56", "name": "Benoni defence, Hromodka system"},
    "d4 Nf6 c4 c5 d5 Ne4": {"eco": "A56", "name": "Vulture defence"},
    "d4 Nf6 c4 c5 d5 e5": {"eco": "A56", "name": "Czech Benoni defence"},
    "d4 Nf6 c4 c5 d5 e5 Nc3 d6 e4 g6": {
        "eco": "A56",
        "name": "Czech Benoni, King's Indian system",
    },
    "d4 Nf6 c4 c5 d5 b5": {"eco": "A57", "name": "Benko gambit"},
    "d4 Nf6 c4 c5 d5 e6": {"eco": "A60", "name": "Benoni defence"},
    "d4 f5": {"eco": "A80", "name": "Dutch"},
    # B - Volume B
    "e4": {"eco": "B00", "name": "King's pawn opening"},
    "e4 a5": {"eco": "B00", "name": "Corn stalk defence"},
    "e4 Na6": {"eco": "B00", "name": "Lemming defence"},
    "e4 f5": {"eco": "B00", "name": "Fred"},
    "e4 f6": {"eco": "B00", "name": "Barnes defence"},
    "e4 h6": {"eco": "B00", "name": "Carr's defence"},
    "e4 g5": {"eco": "B00", "name": "Reversed Grob (Borg/Basman defence/macho Grob)"},
    "e4 a6": {"eco": "B00", "name": "St. George (Baker) defence"},
    "e4 b6": {"eco": "B00", "name": "Owen defence"},
    "e4 Nc6": {"eco": "B00", "name": "KP, Nimzovich defence"},
    "e4 Nc6 Nf3": {"eco": "B00", "name": "KP, Nimzovich defence"},
    "e4 Nc6 d4": {"eco": "B00", "name": "KP, Nimzovich defence"},
    "e4 f6 d4 Kf7": {"eco": "B00", "name": "Fried fox defence"},
    "e4 b6 d4 Ba6": {"eco": "B00", "name": "Guatemala defence"},
    "e4 Nc6 Nf3 f5": {"eco": "B00", "name": "KP, Colorado counter"},
    "e4 Nc6 d4 f6": {"eco": "B00", "name": "KP, Neo-Mongoloid defence"},
    "e4 Nc6 d4 d5 Nc3": {
        "eco": "B00",
        "name": "KP, Nimzovich defence, Bogolyubov variation",
    },
    "e4 Nh6 d4 g6 c4 f6": {"eco": "B00", "name": "Hippopotamus defence"},
    "e4 Nc6 b4 Nxb4 c3 Nc6 d4": {
        "eco": "B00",
        "name": "KP, Nimzovich defence, Wheeler gambit",
    },
    "e4 Nc6 d4 d5 exd5 Qxd5 Nc3": {
        "eco": "B00",
        "name": "KP, Nimzovich defence, Marshall gambit",
    },
    "e4 d5": {"eco": "B01", "name": "Scandinavian (centre counter) defence"},
    "e4 d5 exd5 Nf6": {"eco": "B01", "name": "Scandinavian defence"},
    "e4 d5 exd5 Nf6 d4": {"eco": "B01", "name": "Scandinavian defence"},
    "e4 d5 exd5 Qxd5 Nc3 Qd6": {
        "eco": "B01",
        "name": "Scandinavian, Pytel-Wade variation",
    },
    "e4 d5 exd5 Nf6 c4 e6": {"eco": "B01", "name": "Scandinavian, Icelandic gambit"},
    "e4 d5 exd5 Nf6 c4 c6": {"eco": "B01", "name": "Scandinavian gambit"},
    "e4 d5 exd5 Nf6 d4 Nxd5": {
        "eco": "B01",
        "name": "Scandinavian, Marshall variation",
    },
    "e4 d5 exd5 Nf6 d4 g6": {"eco": "B01", "name": "Scandinavian, Richter variation"},
    "e4 d5 exd5 Qxd5 Nc3 Qa5 b4": {
        "eco": "B01",
        "name": "Scandinavian, Mieses-Kotrvc gambit",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 e5": {
        "eco": "B01",
        "name": "Scandinavian, Anderssen counter-attack",
    },
    "e4 d5 exd5 Nf6 d4 Nxd5 c4 Nb4": {
        "eco": "B01",
        "name": "Scandinavian, Kiel variation",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 e5 Nf3": {
        "eco": "B01",
        "name": "Scandinavian, Anderssen counter-attack, Goteborg system",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bf5": {
        "eco": "B01",
        "name": "Scandinavian defence",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 e5 Nf3 Bg4": {
        "eco": "B01",
        "name": "Scandinavian, Anderssen counter-attack, Collijn variation",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bg4 h3": {
        "eco": "B01",
        "name": "Scandinavian defence, Lasker variation",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bf5 Ne5 c6 g4": {
        "eco": "B01",
        "name": "Scandinavian defence, Gruenfeld variation",
    },
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 e5 dxe5 Bb4 Bd2 Nc6 Nf3": {
        "eco": "B01",
        "name": "Scandinavian, Anderssen counter-attack orthodox attack",
    },
    "e4 Nf6": {"eco": "B02", "name": "Alekhine's defence"},
    "e4 g6": {"eco": "B06", "name": "Robatsch (modern) defence"},
    "e4 g6 d4 Bg7": {"eco": "B06", "name": "Robatsch (modern) defence"},
    "e4 g6 d4 Bg7 f4": {"eco": "B06", "name": "Robatsch defence, three pawns attack"},
    "e4 g6 d4 Bg7 Nc3": {"eco": "B06", "name": "Robatsch defence"},
    "e4 g6 d4 Bg7 Nc3 d6": {"eco": "B06", "name": "Robatsch (modern) defence"},
    "e4 g6 d4 Bg7 Nc3 d6 Nf3": {
        "eco": "B06",
        "name": "Robatsch defence, two knights variation",
    },
    "e4 g6 d4 Bg7 Nc3 d6 f4": {
        "eco": "B06",
        "name": "Robatsch defence, Pseudo-Austrian attack",
    },
    "e4 g6 d4 Nf6 e5 Nh5 g4 Ng7": {"eco": "B06", "name": "Norwegian defence"},
    "e4 g6 d4 Bg7 Nc3 d6 Nf3 c6": {
        "eco": "B06",
        "name": "Robatsch defence, two knights, Suttles variation",
    },
    "e4 g6 d4 Bg7 Nc3 c6 f4 d5 e5 h5": {
        "eco": "B06",
        "name": "Robatsch defence, Gurgenidze variation",
    },
    "e4 d6 d4 Nf6 Nc3": {"eco": "B07", "name": "Pirc defence"},
    "e4 c6": {"eco": "B10", "name": "Caro-Kann defence"},
    "e4 c5": {"eco": "B20", "name": "Sicilian defence"},
    # C - Volume C
    "e4 e6": {"eco": "C00", "name": "French defence"},
    "e4 e5": {"eco": "C20", "name": "King's pawn game"},
    "e4 e5 d3": {"eco": "C20", "name": "KP, Indian opening"},
    "e4 e5 a3": {"eco": "C20", "name": "KP, Mengarini's opening"},
    "e4 e5 f3": {"eco": "C20", "name": "KP, King's head opening"},
    "e4 e5 Qh5": {"eco": "C20", "name": "KP, Patzer opening"},
    "e4 e5 Qf3": {"eco": "C20", "name": "KP, Napoleon's opening"},
    "e4 e5 c3": {"eco": "C20", "name": "KP, Lopez opening"},
    "e4 e5 Ne2": {"eco": "C20", "name": "Alapin's opening"},
    "e4 e5 d4 exd4": {"eco": "C21", "name": "Centre game"},
    "e4 e5 Bc4": {"eco": "C23", "name": "Bishop's opening"},
    "e4 e5 Nc3": {"eco": "C25", "name": "Vienna game"},
    "e4 e5 f4": {"eco": "C30", "name": "King's gambit"},
    "e4 e5 Nf3": {"eco": "C40", "name": "King's knight opening"},
    "e4 e5 Nf3 Qe7": {"eco": "C40", "name": "Gunderam defence"},
    "e4 e5 Nf3 Qf6": {"eco": "C40", "name": "Greco defence"},
    "e4 e5 Nf3 f6": {"eco": "C40", "name": "Damiano's defence"},
    "e4 e5 Nf3 d5": {"eco": "C40", "name": "QP counter-gambit (elephant gambit)"},
    "e4 e5 Nf3 f5": {"eco": "C40", "name": "Latvian counter-gambit"},
    "e4 e5 Nf3 f5 Bc4": {"eco": "C40", "name": "Latvian gambit, 3.Bc4"},
    "e4 e5 Nf3 d5 exd5 Bd6": {
        "eco": "C40",
        "name": "QP counter-gambit, Maroczy gambit",
    },
    "e4 e5 Nf3 f5 Nxe5 Nc6": {"eco": "C40", "name": "Latvian, Fraser defence"},
    "e4 e5 Nf3 f5 Bc4 fxe4 Nxe5 d5": {
        "eco": "C40",
        "name": "Latvian, Polerio variation",
    },
    "e4 e5 Nf3 f5 Bc4 fxe4 Nxe5 Nf6": {
        "eco": "C40",
        "name": "Latvian, corkscrew counter-gambit",
    },
    "e4 e5 Nf3 f5 Nxe5 Qf6 d4 d6 Nc4 fxe4 Ne3": {
        "eco": "C40",
        "name": "Latvian, Nimzovich variation",
    },
    "e4 e5 Nf3 f5 Bc4 fxe4 Nxe5 Qg5 Nf7 Qxg2 Rf1 d5 Nxh8 Nf6": {
        "eco": "C40",
        "name": "Latvian, Behting variation",
    },
    "e4 e5 Nf3 d6": {"eco": "C41", "name": "Philidor's defence"},
    "e4 e5 Nf3 d6 d4": {"eco": "C41", "name": "Philidor's defence"},
    "e4 e5 Nf3 d6 Bc4 f5": {"eco": "C41", "name": "Philidor, Lopez counter-gambit"},
    "e4 e5 Nf3 d6 d4 f5": {"eco": "C41", "name": "Philidor, Philidor counter-gambit"},
    "e4 e5 Nf3 d6 d4 exd4": {"eco": "C41", "name": "Philidor, exchange variation"},
    "e4 e5 Nf3 d6 d4 Nf6": {
        "eco": "C41",
        "name": "Philidor, Nimzovich (Jaenisch) variation",
    },
    "e4 e5 Nf3 d6 d4 Nd7": {"eco": "C41", "name": "Philidor, Hanham variation"},
    "e4 e5 Nf3 d6 Bc4 Be7 c3": {"eco": "C41", "name": "Philidor, Steinitz variation"},
    "e4 e5 Nf3 d6 d4 f5 Nc3": {
        "eco": "C41",
        "name": "Philidor, Philidor counter-gambit, Zukertort variation",
    },
    "e4 e5 Nf3 d6 d4 exd4 Nxd4": {"eco": "C41", "name": "Philidor, exchange variation"},
    "e4 e5 Nf3 d6 d4 Nf6 dxe5": {"eco": "C41", "name": "Philidor, Nimzovich variation"},
    "e4 e5 Nf3 d6 d4 Nf6 Ng5": {
        "eco": "C41",
        "name": "Philidor, Nimzovich, Locock variation",
    },
    "e4 e5 Nf3 d6 d4 Nf6 Bc4": {
        "eco": "C41",
        "name": "Philidor, Nimzovich, Klein variation",
    },
    "e4 e5 Nf3 d6 d4 exd4 Qxd4 Bd7": {
        "eco": "C41",
        "name": "Philidor, Boden variation",
    },
    "e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6": {
        "eco": "C41",
        "name": "Philidor, exchange variation",
    },
    "e4 e5 Nf3 d6 d4 exd4 Nxd4 g6": {
        "eco": "C41",
        "name": "Philidor, Larsen variation",
    },
    "e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7": {
        "eco": "C41",
        "name": "Philidor, Improved Hanham variation",
    },
    "e4 e5 Nf3 d6 d4 exd4 Nxd4 d5 exd5": {
        "eco": "C41",
        "name": "Philidor, Paulsen attack",
    },
    "e4 e5 Nf3 d6 d4 Nf6 dxe5 Nxe4 Nbd2": {
        "eco": "C41",
        "name": "Philidor, Nimzovich, Sokolsky variation",
    },
    "e4 e5 Nf3 d6 d4 Nf6 dxe5 Nxe4 Qd5": {
        "eco": "C41",
        "name": "Philidor, Nimzovich, Rellstab variation",
    },
    "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 O-O": {
        "eco": "C41",
        "name": "Philidor, Hanham, Krause variation",
    },
    "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 Ng5": {
        "eco": "C41",
        "name": "Philidor, Hanham, Kmoch variation",
    },
    "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 Nc3": {
        "eco": "C41",
        "name": "Philidor, Hanham, Schlechter variation",
    },
    "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 c3": {
        "eco": "C41",
        "name": "Philidor, Hanham, Delmar variation",
    },
    "e4 e5 Nf3 d6 Bc4 f5 d4 exd4 Ng5 Nh6 Nxh7": {
        "eco": "C41",
        "name": "Philidor, Lopez counter-gambit, Jaenisch variation",
    },
    "e4 e5 Nf3 d6 d4 f5 dxe5 fxe4 Ng5 d5 e6": {
        "eco": "C41",
        "name": "Philidor, Philidor counter-gambit, del Rio attack",
    },
    "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 O-O Be7 dxe5": {
        "eco": "C41",
        "name": "Philidor, Hanham, Steiner variation",
    },
    "e4 e5 Nf3 d6 d4 f5 dxe5 fxe4 Ng5 d5 e6 Bc5 Nc3": {
        "eco": "C41",
        "name": "Philidor, Philidor counter-gambit, Berger variation",
    },
    "e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 Ng5 O-O Bxf7+": {
        "eco": "C41",
        "name": "Philidor, Nimzovich, Larobok variation",
    },
    "e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O Qe2 c6 a4 exd4": {
        "eco": "C41",
        "name": "Philidor, Nimzovich, Sozin variation",
    },
    "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 Ng5 Nh6 f4 Be7 O-O O-O c3 d5": {
        "eco": "C41",
        "name": "Philidor, Hanham, Berger variation",
    },
    "e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3 Be7 Be2 O-O O-O c5 Nf3 Nc6 Bg5 Be6 Re1": {
        "eco": "C41",
        "name": "Philidor, Berger variation",
    },
    "e4 e5 Nf3 Nf6": {"eco": "C42", "name": "Petrov's defence"},
    "e4 e5 Nf3 Nc6": {"eco": "C44", "name": "King's pawn game"},
    "e4 e5 Nf3 Nc6 g3": {"eco": "C44", "name": "Konstantinopolsky opening"},
    "e4 e5 Nf3 Nc6 c4": {"eco": "C44", "name": "Dresden opening"},
    "e4 e5 Nf3 Nc6 Be2": {"eco": "C44", "name": "Inverted Hungarian"},
    "e4 e5 Nf3 Nc6 c3": {"eco": "C44", "name": "Ponziani opening"},
    "e4 e5 Nf3 Nc6 d4": {"eco": "C44", "name": "Scotch opening"},
    "e4 e5 Nf3 Nc6 c3 Nf6": {"eco": "C44", "name": "Ponziani, Jaenisch counter-attack"},
    "e4 e5 Nf3 Nc6 c3 Nge7": {"eco": "C44", "name": "Ponziani, Reti variation"},
    "e4 e5 Nf3 Nc6 c3 Be7": {"eco": "C44", "name": "Ponziani, Romanishin variation"},
    "e4 e5 Nf3 Nc6 c3 f5": {"eco": "C44", "name": "Ponziani counter-gambit"},
    "e4 e5 Nf3 Nc6 d4 Nxd4": {"eco": "C44", "name": "Scotch, Lolli variation"},
    "e4 e5 Nf3 Nc6 Nxe5 Nxe5 d4": {"eco": "C44", "name": "Irish (Chicago) gambit"},
    "e4 e5 Nf3 Nc6 Be2 Nf6 d4": {"eco": "C44", "name": "Tayler opening"},
    "e4 e5 Nf3 Nc6 d4 exd4 Bb5": {
        "eco": "C44",
        "name": "Scotch, Relfsson gambit ('MacLopez')",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 c3": {"eco": "C44", "name": "Scotch, Goering gambit"},
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4": {"eco": "C44", "name": "Scotch gambit"},
    "e4 e5 Nf3 Nc6 c3 d5 Qa4 Bd7": {"eco": "C44", "name": "Ponziani, Caro variation"},
    "e4 e5 Nf3 Nc6 c3 d5 Qa4 Nf6": {
        "eco": "C44",
        "name": "Ponziani, Leonhardt variation",
    },
    "e4 e5 Nf3 Nc6 c3 d5 Qa4 f6": {
        "eco": "C44",
        "name": "Ponziani, Steinitz variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bb4+": {"eco": "C44", "name": "Scotch gambit"},
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Be7": {
        "eco": "C44",
        "name": "Scotch gambit, Benima defence",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Nf6": {
        "eco": "C44",
        "name": "Scotch gambit, Dubois-Reti defence",
    },
    "e4 e5 Nf3 Nc6 Be2 Nf6 d3 d5 Nbd2": {"eco": "C44", "name": "Inverted Hanham"},
    "e4 e5 Nf3 Nc6 c3 f5 d4 d6 d5": {
        "eco": "C44",
        "name": "Ponziani counter-gambit, Schmidt attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 Ng5": {"eco": "C44", "name": "Scotch gambit"},
    "e4 e5 Nf3 Nc6 c3 Nf6 d4 Nxe4 d5 Bc5": {
        "eco": "C44",
        "name": "Ponziani, Fraser defence",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 c3 dxc3 Nxc3 Bb4": {
        "eco": "C44",
        "name": "Scotch, Goering gambit",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 Ng5 Nh6 Qh5": {
        "eco": "C44",
        "name": "Scotch gambit, Vitzhum attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bb4+ c3 dxc3 bxc3": {
        "eco": "C44",
        "name": "Scotch gambit",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 c3 dxc3 Nxc3 Bb4 Bc4 Nf6": {
        "eco": "C44",
        "name": "Scotch, Goering gambit, Bardeleben variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 O-O d6 c3 Bg4": {
        "eco": "C44",
        "name": "Scotch gambit, Anderssen (Paulsen, Suhle) counter-attack",
    },
    "e4 e5 Nf3 Nc6 d4 Nxd4 Nxe5 Ne6 Bc4 c6 O-O Nf6 Nxf7": {
        "eco": "C44",
        "name": "Scotch, Cochrane variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bb4+ c3 dxc3 bxc3 Ba5 e5": {
        "eco": "C44",
        "name": "Scotch gambit, Cochrane variation",
    },
    "e4 e5 Nf3 Nc6 c3 f5 d4 d6 d5 fxe4 Ng5 Nb8 Nxe4 Nf6 Bd3 Be7": {
        "eco": "C44",
        "name": "Ponziani counter-gambit, Cordel variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 Ng5 Nh6 Nxf7 Nxf7 Bxf7+ Kxf7 Qh5+ g6 Qxc5 d5": {
        "eco": "C44",
        "name": "Scotch gambit, Cochrane-Shumov defence",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bb4+ c3 dxc3 O-O cxb2 Bxb2 Nf6 Ng5 O-O e5 Nxe5": {
        "eco": "C44",
        "name": "Scotch gambit, Hanneken variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 c3 dxc3 Nxc3 d6 Bc4 Bg4 O-O Ne5 Nxe5 Bxd1 Bxf7+ Ke7 Nd5+": {
        "eco": "C44",
        "name": "Scotch, Sea-cadet mate",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4": {"eco": "C45", "name": "Scotch game"},
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4": {
        "eco": "C45",
        "name": "Scotch, Pulling counter-attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6": {
        "eco": "C45",
        "name": "Scotch, Schmidt variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5": {"eco": "C45", "name": "Scotch game"},
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4 Nb5": {
        "eco": "C45",
        "name": "Scotch, Horwitz attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4 Nf3": {
        "eco": "C45",
        "name": "Scotch, Fraser attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4 Nc3": {
        "eco": "C45",
        "name": "Scotch, Steinitz variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nb3": {
        "eco": "C45",
        "name": "Scotch, Potter variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nb3 Bb4+": {
        "eco": "C45",
        "name": "Scotch, Romanishin variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nxd4 Qxd4 d6 Bd3": {
        "eco": "C45",
        "name": "Scotch, Ghulam Kassim variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4 Nb5 Bb4+ Bd2": {
        "eco": "C45",
        "name": "Scotch game",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5": {
        "eco": "C45",
        "name": "Scotch, Mieses variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 Nd2": {
        "eco": "C45",
        "name": "Scotch, Tartakower variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 Nb5": {
        "eco": "C45",
        "name": "Scotch, Blumenfeld attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Qd2": {
        "eco": "C45",
        "name": "Scotch, Blackburne attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bb5": {
        "eco": "C45",
        "name": "Scotch, Paulsen attack",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Nc2": {
        "eco": "C45",
        "name": "Scotch, Meitner variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bb5 Nd8": {
        "eco": "C45",
        "name": "Scotch, Paulsen, Gunsberg defence",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4 Nb5 Bb4+ Bd2 Qxe4+ Be2 Kd8 O-O Bxd2 Nxd2 Qg6": {
        "eco": "C45",
        "name": "Scotch, Rosenthal variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qh4 Nb5 Bb4+ Nd2 Qxe4+ Be2 Qxg2 Bf3 Qh3 Nxc7+ Kd8 Nxa8 Nf6 1 - a3": {
        "eco": "C45",
        "name": "Scotch, Berger variation",
    },
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Qd2 d5 Nb5 Bxe3 Qxe3 O-O Nxc7 Rb8 1 - Nxd5 Nxd5 exd5 Nb4": {
        "eco": "C45",
        "name": "Scotch, Gottschall variation",
    },
    "e4 e5 Nf3 Nc6 Nc3": {"eco": "C46", "name": "Three knights game"},
    "e4 e5 Nf3 Nc6 Nc3 f5": {
        "eco": "C46",
        "name": "Three knights, Winawer defence (Gothic defence)",
    },
    "e4 e5 Nf3 Nc6 Nc3 g6": {"eco": "C46", "name": "Three knights, Steinitz variation"},
    "e4 e5 Nf3 Nc6 Nc3 Nf6": {"eco": "C46", "name": "Four knights game"},
    "e4 e5 Nf3 Nc6 Nc3 Nf6 Nxe5": {
        "eco": "C46",
        "name": "Four knights, Schultze-Mueller gambit",
    },
    "e4 e5 Nf3 Nc6 Nc3 Nf6 Bc4": {
        "eco": "C46",
        "name": "Four knights, Italian variation",
    },
    "e4 e5 Nf3 Nc6 Nc3 Nf6 a3": {
        "eco": "C46",
        "name": "Four knights, Gunsberg variation",
    },
    "e4 e5 Nf3 Nc6 Nc3 Bb4 Nd5 Nf6": {
        "eco": "C46",
        "name": "Three knights, Schlechter variation",
    },
    "e4 e5 Nf3 Nc6 Nc3 g6 d4 exd4 Nd5": {
        "eco": "C46",
        "name": "Three knights, Steinitz, Rosenthal variation",
    },
    "e4 e5 Nf3 Nc6 Nc3 Nf6 d4": {
        "eco": "C47",
        "name": "Four knights, Scotch variation",
    },
    "e4 e5 Nf3 Nc6 Bc4": {"eco": "C50", "name": "Italian Game"},
    "e4 e5 Nf3 Nc6 Bc4 f5": {"eco": "C50", "name": "Rousseau gambit"},
    "e4 e5 Nf3 Nc6 Bc4 Be7": {"eco": "C50", "name": "Hungarian defence"},
    "e4 e5 Nf3 Nc6 Bc4 Bc5": {"eco": "C50", "name": "Giuoco Piano"},
    "e4 e5 Nf3 Nc6 Bc4 Bc5 Bxf7+": {
        "eco": "C50",
        "name": "Giuoco Piano, Jerome gambit",
    },
    "e4 e5 Nf3 Nc6 Bc4 Bc5 d3": {"eco": "C50", "name": "Giuoco Pianissimo"},
    "e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6": {
        "eco": "C50",
        "name": "Giuoco Piano, four knights variation",
    },
    "e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6": {"eco": "C50", "name": "Giuoco Pianissimo"},
    "e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 Nc3": {
        "eco": "C50",
        "name": "Giuoco Pianissimo, Italian four knights variation",
    },
    "e4 e5 Nf3 Nc6 Bc4 Bc5 d3 f5 Ng5 f4": {
        "eco": "C50",
        "name": "Giuoco Pianissimo, Dubois variation",
    },
    "e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 Nc3 d6 Bg5": {
        "eco": "C50",
        "name": "Giuoco Pianissimo, Canal variation",
    },
    "e4 e5 Nf3 Nc6 Bc4 Be7 d4 exd4 c3 Nf6 e5 Ne4": {
        "eco": "C50",
        "name": "Hungarian defence, Tartakower variation",
    },
    "e4 e5 Nf3 Nc6 Bc4 Nd4 Nxe5 Qg5 Nxf7 Qxg2 Rf1 Qxe4+ Be2 Nf3+": {
        "eco": "C50",
        "name": "Blackburne shilling gambit",
    },
    "e4 e5 Nf3 Nc6 Bc4 Bc5 b4": {"eco": "C51", "name": "Evans gambit"},
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3": {"eco": "C53", "name": "Giuoco Piano"},
    "e4 e5 Nf3 Nc6 Bc4 Nf6": {"eco": "C55", "name": "Two knights defence"},
    "e4 e5 Nf3 Nc6 Bb5": {"eco": "C60", "name": "Ruy Lopez (Spanish opening)"},
    # D - Volume D
    "d4 d5": {"eco": "D00", "name": "Queen's pawn game"},
    "d4 d5 Bf4": {"eco": "D00", "name": "Queen's pawn, Mason variation"},
    "d4 d5 Bg5": {"eco": "D00", "name": "Levitsky attack (Queen's bishop attack)"},
    "d4 d5 e4": {"eco": "D00", "name": "Blackmar gambit"},
    "d4 d5 Nc3": {"eco": "D00", "name": "Queen's pawn, Chigorin variation"},
    "d4 d5 Bf4 c5": {
        "eco": "D00",
        "name": "Queen's pawn, Mason variation, Steinitz counter-gambit",
    },
    "d4 d5 Nc3 Bg4": {"eco": "D00", "name": "Queen's pawn, Anti-Veresov"},
    "d4 d5 e3 Nf6 Bd3": {"eco": "D00", "name": "Queen's pawn, stonewall attack"},
    "d4 d5 Nc3 Nf6 e4": {"eco": "D00", "name": "Blackmar-Diemer gambit"},
    "d4 d5 Nc3 Nf6 e4 e5": {
        "eco": "D00",
        "name": "Blackmar-Diemer, Lemberg counter-gambit",
    },
    "d4 d5 Nc3 Nf6 e4 dxe4 f3 exf3 Nxf3 e6": {
        "eco": "D00",
        "name": "Blackmar-Diemer, Euwe defence",
    },
    "d4 d5 Nc3 Nf6 Bg5": {"eco": "D01", "name": "Richter-Veresov attack"},
    "d4 d5 Nc3 Nf6 Bg5 Bf5 Bxf6": {
        "eco": "D01",
        "name": "Richter-Veresov attack, Veresov variation",
    },
    "d4 d5 Nc3 Nf6 Bg5 Bf5 f3": {
        "eco": "D01",
        "name": "Richter-Veresov attack, Richter variation",
    },
    "d4 d5 Nf3": {"eco": "D02", "name": "Queen's pawn game"},
    "d4 d5 Nf3 Nc6": {"eco": "D02", "name": "Queen's pawn game, Chigorin variation"},
    "d4 d5 Nf3 c5": {"eco": "D02", "name": "Queen's pawn game, Krause variation"},
    "d4 d5 Nf3 Nf6": {"eco": "D02", "name": "Queen's pawn game"},
    "d4 d5 Nf3 Nf6 Bf4": {"eco": "D02", "name": "London System"},
    "d4 d5 Nf3 Nf6 Bg5": {"eco": "D03", "name": "Torre attack (Tartakower variation)"},
    "d4 d5 Nf3 Nf6 e3": {"eco": "D04", "name": "Queen's pawn game"},
    "d4 d5 c4": {"eco": "D06", "name": "Queen's Gambit"},
    "d4 d5 c4 Bf5": {
        "eco": "D06",
        "name": "Queen's Gambit Declined, Grau (Sahovic) defence",
    },
    "d4 d5 c4 Nf6": {"eco": "D06", "name": "Queen's Gambit Declined, Marshall defence"},
    "d4 d5 c4 c5": {
        "eco": "D06",
        "name": "Queen's Gambit Declined, symmetrical (Austrian) defence",
    },
    "d4 d5 c4 Nc6": {"eco": "D07", "name": "Queen's Gambit Declined, Chigorin defence"},
    "d4 d5 c4 c6": {"eco": "D10", "name": "Queen's Gambit Declined Slav defence"},
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4": {
        "eco": "D16",
        "name": "Queen's Gambit Declined Slav accepted, Alapin variation",
    },
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 e6": {
        "eco": "D16",
        "name": "Queen's Gambit Declined Slav, Soultanbeieff variation",
    },
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bg4": {
        "eco": "D16",
        "name": "Queen's Gambit Declined Slav, Steiner variation",
    },
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Na6 e4 Bg4": {
        "eco": "D16",
        "name": "Queen's Gambit Declined Slav, Smyslov variation",
    },
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5": {
        "eco": "D17",
        "name": "Queen's Gambit Declined Slav, Czech defence",
    },
    "d4 d5 c4 dxc4": {"eco": "D20", "name": "Queen's gambit accepted"},
    "d4 d5 c4 e6": {"eco": "D30", "name": "Queen's gambit declined"},
    "d4 d5 c4 e6 Nc3 Nf6 Nf3 c6": {
        "eco": "D43",
        "name": "Queen's Gambit Declined semi-Slav",
    },
    "d4 d5 c4 e6 Nc3 Nf6 Bg5": {"eco": "D50", "name": "Queen's Gambit Declined, 4.Bg5"},
    "d4 Nf6 c4 g6 f3 d5": {"eco": "D70", "name": "Neo-Gruenfeld defence"},
    "d4 Nf6 c4 g6 Nc3 d5": {"eco": "D80", "name": "Gruenfeld defence"},
    # E - Volume E
    "d4 Nf6 c4 e6": {"eco": "E00", "name": "Queen's pawn game"},
    "d4 Nf6 c4 e6 Bg5": {"eco": "E00", "name": "Neo-Indian (Seirawan) attack"},
    "d4 Nf6 c4 e6 g3": {"eco": "E00", "name": "Catalan opening"},
    "d4 Nf6 c4 e6 g3 d5 Bg2": {"eco": "E01", "name": "Catalan, closed"},
    "d4 Nf6 c4 e6 Nf3": {"eco": "E10", "name": "Queen's pawn game"},
    "d4 Nf6 c4 e6 Nf3 a6": {"eco": "E10", "name": "Dzindzikhashvili defence"},
    "d4 Nf6 c4 e6 Nf3 Ne4": {"eco": "E10", "name": "Doery defence"},
    "d4 Nf6 c4 e6 Nf3 c5 d5 b5": {"eco": "E10", "name": "Blumenfeld counter-gambit"},
    "d4 Nf6 c4 e6 Nf3 c5 d5 b5 Bg5": {
        "eco": "E10",
        "name": "Blumenfeld counter-gambit, Dus-Chotimursky variation",
    },
    "d4 Nf6 c4 e6 Nf3 c5 d5 b5 dxe6 fxe6 cxb5 d5": {
        "eco": "E10",
        "name": "Blumenfeld counter-gambit accepted",
    },
    "d4 Nf6 c4 e6 Nf3 c5 d5 b5 Bg5 exd5 cxd5 h6": {
        "eco": "E10",
        "name": "Blumenfeld counter-gambit, Spielmann variation",
    },
    "d4 Nf6 c4 e6 Nf3 Bb4+": {"eco": "E11", "name": "Bogo-Indian defence"},
    "d4 Nf6 c4 e6 Nf3 Bb4+ Nbd2": {
        "eco": "E11",
        "name": "Bogo-Indian defence, Gruenfeld variation",
    },
    "d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Qe7": {
        "eco": "E11",
        "name": "Bogo-Indian defence, Nimzovich variation",
    },
    "d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Bxd2+ Qxd2 b6 g3 Bb7 Bg2 O-O Nc3 Ne4 Qc2 Nxc3 Ng5": {
        "eco": "E11",
        "name": "Bogo-Indian defence, Monticelli trap",
    },
    "d4 Nf6 c4 e6 Nf3 b6": {"eco": "E12", "name": "Queen's Indian defence"},
    "d4 Nf6 c4 e6 Nc3 Bb4": {"eco": "E20", "name": "Nimzo-Indian defence"},
    "d4 Nf6 c4 g6": {"eco": "E60", "name": "King's Indian defence"},
}


class OpeningDisplay(tk.Frame):
    """
    Widget displaying the current opening name and ECO code.

    Updates as moves are played, shows "Unknown" when out of book.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 200,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, width=width, **kwargs)

        self.configure(
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

        self._current_opening: Optional[dict[str, str]] = None
        self._in_book = True

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create display widgets."""
        # Title
        title = tk.Label(
            self,
            text="OPENING",
            font=("Segoe UI", 9, "bold"),
            fg=COLORS["text_muted"],
            bg=bg,
        )
        title.pack(pady=(8, 2))

        # ECO code
        self._eco_label = tk.Label(
            self,
            text="",
            font=("Consolas", 10, "bold"),
            fg=COLORS["opening_eco"],
            bg=bg,
        )
        self._eco_label.pack()

        # Opening name
        self._name_label = tk.Label(
            self,
            text="Starting Position",
            font=("Segoe UI", 10),
            fg=COLORS["text_primary"],
            bg=bg,
            wraplength=180,
        )
        self._name_label.pack(pady=(2, 8))

    def update_opening(self, board: chess.Board) -> None:
        """
        Update opening display from board position.

        Args:
            board: Current chess board with move history
        """
        # Get move sequence
        move_string = self._get_move_string(board)

        # Look up opening
        opening = self._lookup_opening(move_string)

        if opening:
            self._current_opening = opening
            self._in_book = True
            self._eco_label.configure(text=opening["eco"])
            self._name_label.configure(
                text=opening["name"],
                fg=COLORS["text_primary"],
            )
        elif self._in_book and len(board.move_stack) > 0:
            # Just left the book
            self._in_book = False
            if self._current_opening:
                # Keep last known opening but gray out
                self._name_label.configure(fg=COLORS["text_muted"])
            else:
                self._eco_label.configure(text="")
                self._name_label.configure(
                    text="Unorthodox Opening",
                    fg=COLORS["text_muted"],
                )

    def _get_move_string(self, board: chess.Board) -> str:
        """Convert board move history to lookup key."""
        temp_board = chess.Board()
        moves = []

        try:
            for move in board.move_stack:
                san = temp_board.san(move)
                moves.append(san)
                temp_board.push(move)
        except (AssertionError, ValueError):
            # Position was set from editor, moves don't match standard start
            return ""

        return " ".join(moves)

    def _lookup_opening(self, move_string: str) -> Optional[dict[str, str]]:
        """
        Look up opening from move sequence.

        Tries progressively shorter sequences to find best match.
        """
        # Direct lookup
        if move_string in ECO_OPENINGS:
            return ECO_OPENINGS[move_string]

        # Try progressively shorter sequences
        moves = move_string.split()
        for length in range(len(moves) - 1, 0, -1):
            partial = " ".join(moves[:length])
            if partial in ECO_OPENINGS:
                return ECO_OPENINGS[partial]

        return None

    def set_opening(self, eco: str, name: str) -> None:
        """
        Manually set the opening display.

        Args:
            eco: ECO code (e.g., "B90")
            name: Opening name
        """
        self._eco_label.configure(text=eco)
        self._name_label.configure(text=name, fg=COLORS["text_primary"])
        self._current_opening = {"eco": eco, "name": name}
        self._in_book = True

    def clear(self) -> None:
        """Reset to starting position."""
        self._eco_label.configure(text="")
        self._name_label.configure(
            text="Starting Position",
            fg=COLORS["text_primary"],
        )
        self._current_opening = None
        self._in_book = True

    def get_current_opening(self) -> Optional[dict[str, str]]:
        """Get the current opening info."""
        return self._current_opening

    def is_in_book(self) -> bool:
        """Check if still in opening book."""
        return self._in_book
