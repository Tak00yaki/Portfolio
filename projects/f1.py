"""
F1 Race Simulator
-----------------
Pick a driver, pick a circuit, watch a 30 minute simulated race with a randomised
starting grid, live leaderboard, weather changes, yellow flags, DNFs and crash
animations. Hover over any dot on track to see that driver's number, name and
current position.

Run with:  ./.venv/bin/python f1.py
"""

import math
import random
import sys
import time

import pygame
import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WIDTH, HEIGHT = 1280, 800
FPS = 60
RACE_DURATION_SECONDS = 10 * 60  # the whole race is compressed into 10 real minutes

BG = (18, 18, 22)
PANEL_BG = (28, 28, 34)
PANEL_BORDER = (55, 55, 65)
TEXT = (235, 235, 240)
DIM_TEXT = (150, 150, 160)
ACCENT = (0, 200, 255)
YELLOW_FLAG = (255, 210, 0)
RED_FLAG = (220, 40, 40)
GREEN = (60, 200, 100)
TRACK_COLOR = (70, 70, 80)
TRACK_EDGE = (110, 110, 120)

FONT_NAME = "arial"


# --------------------------------------------------------------------------- #
# Data: teams / drivers / circuits
# --------------------------------------------------------------------------- #

TEAMS = {
    "McLaren":       {"color": (255, 135, 0),   "pace": 0.995},
    "Red Bull":      {"color": (30, 0, 150),    "pace": 0.985},
    "Mercedes":      {"color": (0, 210, 190),   "pace": 0.98},
    "Ferrari":       {"color": (200, 0, 0),     "pace": 0.975},
    "Williams":      {"color": (0, 130, 230),   "pace": 0.95},
    "Racing Bulls":  {"color": (90, 40, 220),   "pace": 0.93},
    "Aston Martin":  {"color": (0, 110, 90),    "pace": 0.925},
    "Haas":          {"color": (215, 215, 220), "pace": 0.91},
    "Alpine":        {"color": (230, 50, 140),  "pace": 0.905},
    "Audi":          {"color": (90, 15, 20),    "pace": 0.89},
    "Cadillac":      {"color": (200, 170, 70),  "pace": 0.87},
}

# 2026 grid - 11 teams (Cadillac's debut season), 22 drivers
DRIVERS = [
    ("NOR", "Lando Norris", "McLaren"),
    ("PIA", "Oscar Piastri", "McLaren"),
    ("VER", "Max Verstappen", "Red Bull"),
    ("HAD", "Isack Hadjar", "Red Bull"),
    ("RUS", "George Russell", "Mercedes"),
    ("ANT", "Kimi Antonelli", "Mercedes"),
    ("LEC", "Charles Leclerc", "Ferrari"),
    ("HAM", "Lewis Hamilton", "Ferrari"),
    ("ALB", "Alex Albon", "Williams"),
    ("SAI", "Carlos Sainz", "Williams"),
    ("LAW", "Liam Lawson", "Racing Bulls"),
    ("LIN", "Arvid Lindblad", "Racing Bulls"),
    ("ALO", "Fernando Alonso", "Aston Martin"),
    ("STR", "Lance Stroll", "Aston Martin"),
    ("OCO", "Esteban Ocon", "Haas"),
    ("BEA", "Oliver Bearman", "Haas"),
    ("GAS", "Pierre Gasly", "Alpine"),
    ("COL", "Franco Colapinto", "Alpine"),
    ("HUL", "Nico Hulkenberg", "Audi"),
    ("BOR", "Gabriel Bortoleto", "Audi"),
    ("PER", "Sergio Perez", "Cadillac"),
    ("BOT", "Valtteri Bottas", "Cadillac"),
]

DRIVER_NUMBERS = {
    "NOR": 1, "PIA": 81, "VER": 3, "HAD": 6, "RUS": 63, "ANT": 12,
    "LEC": 16, "HAM": 44, "ALB": 23, "SAI": 55, "LAW": 30, "LIN": 41,
    "ALO": 14, "STR": 18, "OCO": 31, "BEA": 87, "GAS": 10, "COL": 43,
    "HUL": 27, "BOR": 5, "PER": 11, "BOT": 77,
}


def _make_from_raw(cx, cy, raw, scale=1.0):
    """Turn a list of hand-placed (x, y) waypoints tracing a real circuit's
    shape into a closed loop of screen-space points. Each waypoint doubles
    as a numbered "turn" for the official-style track map."""
    return [(cx + x * scale, cy + y * scale) for x, y in raw]


CHAIKIN_ITERATIONS = 3
POINTS_PER_CORNER = 2 ** CHAIKIN_ITERATIONS


def _chaikin_smooth(points, iterations=CHAIKIN_ITERATIONS):
    """Round the sharp polygon corners between hand-placed waypoints into a
    smooth, road-like curve via Chaikin's corner-cutting algorithm. Each
    iteration replaces every edge (p0, p1) with two points at 25% and 75%
    along it, which - unlike a spline fit - always stays inside the hull of
    the original points, so tight chicanes never overshoot into loops.
    Each original edge maps to exactly POINTS_PER_CORNER points in the
    output, in order, which lets the renderer map a smoothed segment back
    to the corner/sector it belongs to."""
    pts = points
    for _ in range(iterations):
        n = len(pts)
        new_pts = []
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            new_pts.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            new_pts.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        pts = new_pts
    return pts


# Each raw list is a hand-traced sequence of waypoints (in an arbitrary local
# unit grid, roughly centered on 0,0) that follows the actual corner-by-corner
# shape of the real circuit, so the silhouette on screen is recognisable
# rather than a generic oval.

# Autodromo Nazionale Monza: two long straights joined by the Rettifilo
# chicane, Curva Grande, the two Lesmos, the Ascari chicane and Parabolica.
MONZA_RAW = [
    (-1.5, 5.2), (2.8, 5.2),                                    # start/finish straight
    (3.3, 4.9), (3.0, 4.5), (3.6, 4.2),                          # Variante del Rettifilo chicane
    (4.3, 3.0), (4.9, 1.2), (5.0, -0.8), (4.6, -2.4),            # Curva Grande
    (4.0, -3.4), (3.2, -3.6),                                    # Lesmo 1
    (3.6, -4.4), (3.0, -5.0),                                    # Lesmo 2
    (1.0, -5.6), (-1.5, -5.8), (-3.4, -5.4),                     # Serraglio back straight
    (-4.0, -4.7), (-3.5, -4.2), (-4.2, -3.7), (-3.8, -3.1),      # Variante Ascari chicane
    (-4.6, -1.6), (-4.9, 0.4), (-4.6, 2.2), (-3.8, 3.7),         # Parabolica
    (-2.6, 4.7),
]

# Silverstone: Abbey/Village/Loop complex, the Wellington straight loop
# around Brooklands/Luffield, Copse into the Maggotts-Becketts-Chapel esses,
# Hangar straight, Stowe and Vale back to the start/finish straight.
SILVERSTONE_RAW = [
    (0, 5.0), (3.0, 4.7),                                        # start/finish straight
    (4.3, 3.6), (4.9, 2.0), (4.7, 0.2),                          # Abbey / Farm sweep
    (3.8, -1.2), (2.2, -1.8), (0.4, -1.6),                       # Village / The Loop / Aintree
    (-1.2, -0.8), (-1.6, 0.8),                                   # Wellington straight
    (-2.6, 1.6), (-3.6, 0.6),                                    # Copse
    (-4.4, -0.6), (-3.8, -1.6), (-4.8, -2.2), (-4.2, -3.2), (-5.2, -3.6),  # Maggotts-Becketts-Chapel esses
    (-5.8, -2.0), (-5.9, -0.2),                                  # Hangar straight
    (-5.2, 1.2), (-4.4, 2.1),                                    # Stowe
    (-3.0, 3.0), (-2.4, 4.2), (-1.2, 5.0),                       # Vale / Club
]

# Spa-Francorchamps: long and narrow. La Source hairpin, Eau Rouge/Raidillon,
# Kemmel straight, Les Combes chicane, Pouhon, Fagnes, Stavelot, the long
# Blanchimont sweep and the Bus Stop chicane back to the line.
SPA_RAW = [
    (0, 5.5),                                                    # start/finish
    (-0.6, 5.0), (-0.3, 4.5),                                    # La Source hairpin
    (0.6, 3.8), (1.4, 2.4),                                      # Eau Rouge / Raidillon
    (1.7, 0.6), (1.9, -1.4),                                     # Kemmel straight
    (1.4, -2.0), (0.9, -2.5),                                    # Les Combes chicane
    (1.6, -3.2), (1.2, -4.4),                                    # Malmedy / Rivage
    (0.2, -4.9), (-1.0, -4.6), (-1.8, -3.8),                     # Pouhon
    (-2.6, -4.4), (-3.4, -4.0),                                  # Fagnes
    (-3.0, -2.8), (-2.2, -2.0),                                  # Campus
    (-2.8, -0.8), (-3.6, -0.2),                                  # Stavelot
    (-3.4, 1.4), (-2.6, 3.0), (-1.8, 4.0),                       # Blanchimont
    (-1.6, 4.9), (-0.8, 5.3),                                    # Bus Stop chicane
]

# Monaco: the tight, squiggly street circuit around the harbour.
MONACO_RAW = [
    (0, 0), (2, -1), (3.2, -2.6), (3, -4), (1.5, -4.6), (0, -4),
    (-0.5, -3), (0.2, -2.2), (1.6, -2.4), (2, -3.4), (1.2, -4.2),
    (-1, -4.4), (-2.6, -3.6), (-3, -2), (-2.6, -0.4), (-1.6, 0.6),
    (-2.2, 1.8), (-3.4, 2.2), (-3.6, 3.4), (-2.4, 4.2), (-0.6, 4),
    (0.6, 3), (1.8, 3.4), (2.6, 2.6), (2.2, 1.4), (0.8, 1),
]

# Suzuka: F1's only figure-eight circuit. Esses down the bottom-right, Dunlop
# and Degner curves, the hairpin, the long Spoon curve loop on the left, then
# the back straight crosses back over itself before 130R and the Casio chicane.
SUZUKA_RAW = [
    (0.6, -4.2), (1.6, -3.6), (1.4, -2.4),                       # start/finish, T1, T2
    (0.6, -1.8), (1.0, -1.0), (0.2, -0.4), (0.6, 0.4), (-0.2, 1.0),  # Esses
    (0.4, 1.8), (1.4, 1.6),                                      # Dunlop curve
    (2.2, 2.4), (2.0, 3.4),                                      # Degner 1-2
    (1.0, 4.0), (-0.2, 3.6),                                     # Hairpin
    (-1.4, 4.2), (-2.6, 3.6), (-3.2, 2.2),                       # Spoon curve
    (-2.6, 0.6), (-1.6, -0.6),                                   # back straight - crosses over the esses here
    (-0.4, -1.8), (0.4, -3.0),                                   # 130R
    (1.4, -3.4), (1.8, -2.6), (1.2, -1.8), (0.4, -2.0),          # Casio Triangle chicane
]

# Autodromo Hermanos Rodriguez (Mexico): sharp right-hander complex off the
# start/finish straight, a long run down the back, sector-2 esses, then the
# tight Foro Sol stadium section hairpin before the run back to the line.
MEXICO_RAW = [
    (-0.4, -4.6), (1.6, -4.4),                                   # start/finish straight
    (2.8, -3.6), (3.2, -2.2),                                    # Turns 1-3
    (2.6, -0.4), (2.8, 1.6),                                     # long back straight
    (2.0, 2.8), (0.8, 2.4), (0.2, 3.2), (-0.8, 2.8),             # sector 2 esses
    (-1.6, 3.6), (-1.2, 4.6), (-2.4, 4.8), (-3.0, 3.8),          # Foro Sol stadium section
    (-2.4, 2.8), (-3.2, 1.8),
    (-3.6, 0.0), (-3.2, -2.0), (-2.4, -3.6), (-1.4, -4.4),       # run back up to the line
]

# Lusail International Circuit (Qatar): a tight hairpin leads into a run of
# high-speed esses, a loop through the middle sector, then a long back
# straight down to a final hairpin before the pit straight.
QATAR_RAW = [
    (-0.5, 4.4), (0.3, 4.2),                                     # start/finish straight
    (1.6, 3.4), (1.4, 2.6),
    (2.4, 1.8), (2.0, 0.8),
    (0.8, 0.6), (0.9, -0.4), (-0.2, -0.8), (0.0, -1.8),          # esses
    (-1.0, -3.2),
    (0.2, -3.6), (1.4, -2.8),
    (2.6, -2.4), (2.8, -1.2),
    (2.0, -0.4), (0.8, -0.8),
    (-1.4, -0.2), (-3.2, 0.4),                                   # long straight to the hairpin
    (-3.8, 1.6), (-3.0, 2.4),                                    # hairpin
    (-3.4, 3.4), (-2.2, 3.8),
]

# Red Bull Ring (Austria): a short, punchy circuit - a hairpin at the bottom,
# a long uphill straight to two fast rights, a wiggly infield sector 2, and a
# run back down the hill to the line.
AUSTRIA_RAW = [
    (0.2, 4.2),                                                  # start/finish straight
    (-1.6, 4.6), (-1.9, 3.6),                                    # Turn 1 hairpin
    (-2.6, 1.6),                                                 # uphill straight
    (-2.2, 0.2), (-1.2, -0.6),                                   # Turns 3-4
    (-0.2, -1.4), (0.6, -0.9), (0.2, -0.1), (1.0, 0.4),          # sector 2 infield esses
    (2.4, -0.4), (3.4, 1.0),                                     # Turn 9
    (3.0, 2.8), (1.8, 3.8),                                      # Turn 10 into the finish straight
]

# Albert Park (Australia): semi-permanent circuit looping the lake, fast and
# flowing since its 2022 reprofile.
AUSTRALIA_RAW = [
    (0, 5.0), (2.2, 4.6),                                        # start/finish straight
    (3.4, 3.6), (3.8, 2.0),                                      # Turns 1-3
    (3.2, 0.6), (3.6, -0.8),                                     # Turns 4-6
    (2.8, -2.2), (1.2, -2.6),                                    # Turns 7-8
    (0.0, -2.0), (-1.2, -2.6),                                   # Turn 9
    (-2.6, -2.0), (-3.4, -0.6),                                  # Turns 10-11
    (-3.0, 1.2), (-2.0, 2.4),                                    # Turn 12
    (-2.6, 3.6), (-1.4, 4.6),                                    # Turns 13-14
]

# Shanghai International Circuit (China): the signature decreasing-radius
# "snail shell" spiral through Turns 1-3, then a long back straight.
CHINA_RAW = [
    (0, 4.8), (1.6, 4.6),                                        # start/finish
    (2.8, 3.8), (3.2, 2.6), (2.6, 1.6), (1.4, 1.4),              # spiral Turns 1-3
    (0.6, 2.2), (-0.4, 2.0),                                     # Turns 4-5
    (-1.2, 1.0), (-0.8, -0.2),                                   # Turns 6-7
    (0.2, -0.8), (1.6, -0.6), (2.4, -1.6),                       # Turns 8-9
    (3.4, -3.0),                                                 # long back straight
    (2.4, -4.2), (0.8, -4.4),                                    # Turns 11-12
    (-0.6, -3.6), (-1.8, -4.2), (-3.0, -3.4),                    # Turn 13
    (-3.6, -1.8), (-3.0, -0.2),                                  # Turn 14
    (-3.6, 1.4), (-2.8, 2.8), (-1.4, 3.8),                       # final stadium section
]

# Miami International Autodrome: built around Hard Rock Stadium with a
# fabricated marina section in the infield.
MIAMI_RAW = [
    (-0.6, 4.6), (1.4, 4.4),                                     # start/finish straight
    (2.6, 3.6), (2.8, 2.2),                                      # Turns 1-4
    (2.0, 1.2), (2.6, 0.2), (2.0, -0.8),                         # Turns 5-6
    (2.8, -1.8), (2.2, -3.0),                                    # back straight to the marina
    (0.8, -3.4), (-0.4, -2.8), (-1.2, -3.4), (-2.4, -2.8),       # marina section
    (-2.8, -1.4),                                                # long straight
    (-3.4, 0.4), (-2.6, 1.6),                                    # Turns 17-18
    (-3.2, 2.8), (-2.2, 4.0),                                    # Turn 19 back to the line
]

# Circuit Gilles Villeneuve (Canada): rounded, on Ile Notre-Dame, ending with
# the Wall of Champions chicane.
CANADA_RAW = [
    (0, 4.6), (2.0, 4.4),                                        # start/finish straight
    (3.2, 3.4), (3.0, 2.0),                                      # Turns 1-2
    (2.2, 1.2), (2.6, 0.0),                                      # Turns 3-4
    (1.8, -1.0), (2.4, -2.2),                                    # Turn 5
    (1.6, -3.4), (0.0, -3.6),                                    # Turn 6 hairpin
    (-1.6, -3.0), (-2.4, -1.8),                                  # long back straight
    (-1.8, -0.6), (-2.6, 0.6),                                   # Turns 8-9
    (-2.0, 1.8), (-2.8, 3.0),                                    # Turn 10
    (-1.8, 4.0), (-0.6, 3.6),                                    # Wall of Champions chicane
]

# Circuit de Barcelona-Catalunya: fast, flowing, with the Campsa hairpin and
# a quick final sector since the 2021 reprofile.
BARCELONA_RAW = [
    (0, 4.8), (2.4, 4.6),                                        # start/finish straight
    (3.4, 3.6), (3.2, 2.2),                                      # Turn 1
    (2.4, 1.4), (2.8, 0.2), (2.0, -0.8),                         # Turns 2-4
    (2.6, -2.0), (1.8, -3.0),                                    # Turn 5
    (0.4, -3.2), (-0.6, -2.4),                                   # Turns 6-7
    (-1.8, -2.8), (-2.8, -1.8),                                  # Turns 8-9
    (-2.2, -0.4), (-3.0, 0.8),                                   # Turn 10 (Campsa)
    (-2.4, 2.0),                                                 # Turns 11-12
    (-3.0, 3.2), (-1.8, 4.2),                                    # fast final turns
]

# Hungaroring: tight, twisty and technical - "Monaco without the walls".
HUNGARORING_RAW = [
    (0, 4.6), (2.0, 4.2),                                        # start/finish
    (3.0, 3.0), (2.4, 1.8),                                      # Turns 1-2
    (3.2, 0.8), (2.6, -0.4),                                     # Turns 3-4
    (1.4, -0.2), (0.8, -1.4),                                    # Turns 5-6
    (1.6, -2.6), (0.6, -3.4),                                    # Turn 7
    (-0.8, -3.0), (-1.2, -1.8),                                  # Turns 8-9
    (-2.4, -2.2), (-3.2, -1.0),                                  # Turns 10-11
    (-2.6, 0.2), (-3.4, 1.4),                                    # Turns 12-13
    (-2.6, 2.6), (-1.2, 3.6),                                    # final esses
]

# Circuit Zandvoort (Netherlands): compact dune circuit with banked corners.
ZANDVOORT_RAW = [
    (0, 4.4), (1.8, 4.0),                                        # start/finish, Turn 1
    (2.8, 2.8), (2.2, 1.6),                                      # Turns 2-3 (banked)
    (3.0, 0.6), (2.2, -0.6),                                     # Turns 4-5
    (2.8, -1.8), (1.6, -2.6),                                    # Turns 6-7 (Hugenholtz)
    (0.2, -2.0), (-0.8, -2.8),                                   # Turns 8-9
    (-2.0, -2.0), (-2.6, -0.8),                                  # Turns 10-11
    (-1.8, 0.2), (-2.8, 1.2),                                    # Turn 12
    (-2.2, 2.4), (-0.8, 3.4),                                    # banked final corner
]

# Madring (Spain / Madrid): brand-new for 2026, a hybrid layout with a tight
# infield loop around the IFEMA convention-centre "bowl" section.
MADRING_RAW = [
    (0, 4.6), (2.2, 4.2),                                        # start/finish straight
    (3.4, 3.0), (3.0, 1.6),                                      # Turns 1-2
    (3.6, 0.4), (2.8, -0.8),                                     # Turns 3-4
    (3.2, -2.0), (2.0, -2.8),                                    # Turn 5
    (0.8, -2.2), (0.2, -3.4), (-1.0, -3.8), (-1.8, -2.8), (-1.0, -2.0), (0.0, -2.4),  # infield bowl loop
    (-2.0, -1.4),
    (-3.0, -0.2), (-2.4, 1.0),                                   # Turns
    (-3.2, 2.2), (-2.4, 3.4), (-1.0, 4.0),                       # final sector
]

# Baku City Circuit (Azerbaijan): F1's longest straight leads into a squiggly
# old-town section with the notoriously tight Turn 8 castle chicane.
BAKU_RAW = [
    (0, 5.4), (0.4, 3.4),                                        # the 2.2km main straight
    (1.2, 2.6), (0.8, 1.6),                                      # Turns 1-2
    (1.6, 0.8), (1.0, -0.2),                                     # Turn 3
    (1.8, -1.0), (1.0, -1.8),                                    # castle section esses
    (1.6, -2.6), (0.6, -3.0), (0.2, -2.2), (-0.6, -2.6),         # Turn 8 (super-narrow chicane)
    (-1.2, -1.6),                                                # Turns 9-11
    (-0.6, -0.6), (-1.4, 0.2),                                   # old town squiggle
    (-0.8, 1.2), (-1.6, 2.0),
    (-1.0, 3.0), (-1.8, 4.0), (-0.8, 4.8),                       # back toward the main straight
]

# Marina Bay Street Circuit (Singapore): rectangular street layout around
# the bay, run at night.
SINGAPORE_RAW = [
    (0, 4.4), (2.2, 4.2),                                        # start/finish straight
    (3.2, 3.2), (3.0, 1.8),                                      # Turns 1-3
    (2.2, 1.0), (2.8, -0.2),                                     # Turns 5-6
    (2.0, -1.2), (2.6, -2.2),                                    # Turn 7
    (1.4, -2.8), (0.2, -2.2),                                    # Turns 8-10 (Anderson Bridge)
    (-0.8, -2.8), (-2.0, -2.2),                                  # Turn 13
    (-2.6, -1.0), (-1.8, -0.2),                                  # Turn 14
    (-2.6, 0.8), (-3.2, 2.0),                                    # Turns 16-18
    (-2.4, 3.0), (-1.0, 3.6),                                    # Turns 19-21 back to the line
]

# Circuit of the Americas (USA): uphill Turn 1 into Silverstone-style esses,
# then a big sweeping back section and a stadium finish.
COTA_RAW = [
    (0, 4.8),                                                    # start/finish
    (-0.8, 3.8), (0.2, 3.2), (-0.6, 2.4), (0.4, 1.8),            # Turn 1 (uphill) + esses
    (1.4, 2.2), (1.8, 1.2),                                      # Turns 7-8
    (2.8, 1.4), (3.2, 0.2),                                      # Turns 9-10
    (2.4, -0.8), (3.0, -2.0),                                    # back straight
    (2.0, -3.0), (0.6, -2.6),                                    # Turn 12
    (-0.4, -3.4), (-1.6, -2.8),                                  # Turns 13-14
    (-1.0, -1.6),                                                # Turn 15
    (-2.2, -1.0), (-2.8, 0.4),                                   # Turns 16-18
    (-2.0, 1.6), (-2.6, 3.0), (-1.2, 4.0),                       # stadium section
]

# Interlagos (Brazil): the Senna S at the start, undulating and hilly,
# famous for unpredictable weather.
INTERLAGOS_RAW = [
    (0, 4.6), (-1.2, 4.0), (-0.4, 3.2),                          # Senna S
    (0.6, 2.6), (0.2, 1.6),                                      # Turn 3 (Descida do Lago)
    (1.2, 0.8), (0.8, -0.4),                                     # Turn 4
    (1.8, -1.2), (1.2, -2.4),                                    # Turns 5-6
    (-0.2, -2.6), (-1.0, -1.8),                                  # Turns 7-8
    (-2.2, -2.2), (-2.8, -1.0),                                  # Turns 9-10 (Ferradura)
    (-2.0, 0.0), (-2.8, 1.0),                                    # Turns 11-12
    (-2.0, 2.0),                                                 # Reta Oposta back straight
    (-2.6, 3.2), (-1.4, 3.8),                                    # Juncao back to the line
]

# Las Vegas Strip Circuit: a long rectangular street layout running down the
# Strip past the Sphere and Bellagio.
VEGAS_RAW = [
    (0, 4.6), (2.6, 4.4),                                        # Turns 1-4
    (3.4, 3.4), (3.6, 1.0),                                      # long straight down the Strip
    (3.0, -0.4), (3.4, -2.0),                                    # Turns 5-6, second straight
    (2.4, -3.2), (0.4, -3.4),                                    # Turns 7-9
    (-1.0, -2.8), (-1.6, -1.4),                                  # Turns 10-12
    (-3.2, -0.8), (-3.6, 1.2),                                   # long straight back up
    (-3.0, 2.8), (-1.6, 3.4), (-1.0, 4.2),                       # Turns 14-17 back to the line
]

# Yas Marina Circuit (Abu Dhabi): modern layout around the marina with a
# hotel section, season finale under lights.
YAS_MARINA_RAW = [
    (0, 4.6), (2.0, 4.2),                                        # start/finish
    (3.0, 3.0), (2.6, 1.6),                                      # Turns 1-3
    (3.4, 0.6), (2.6, -0.6),                                     # Turns 4-5
    (3.0, -1.8), (1.8, -2.4),                                    # Turns 6-7
    (0.6, -1.8), (-0.4, -2.6),                                   # Turns 8-9 (hotel section)
    (-1.6, -2.0), (-2.0, -0.8),                                  # Turns 10-11
    (-3.2, -0.4), (-3.0, 1.0),                                   # Turns 12-13
    (-2.0, 1.6), (-2.6, 2.8),                                    # Turns 14-16
    (-1.6, 3.8), (-0.6, 3.4),                                    # final turns back to the line
]

# the full 2026 calendar (22 rounds), in championship order
CIRCUITS = {
    # lat/lon are each circuit's real-world coordinates, used to look up
    # that circuit's actual current weather at race start (see
    # fetch_current_weather_state below).
    "Australia":    {"country": "Australia",  "laps": 58,
                      "raw": AUSTRALIA_RAW, "scale": 82,
                      "rain_chance": 0.20, "lat": -37.8497, "lon": 144.9680},
    "China":        {"country": "China",      "laps": 56,
                      "raw": CHINA_RAW, "scale": 80,
                      "rain_chance": 0.20, "lat": 31.3389, "lon": 121.2198},
    "Suzuka":       {"country": "Japan",      "laps": 53,
                      "raw": SUZUKA_RAW, "scale": 90,
                      "rain_chance": 0.30, "lat": 34.8431, "lon": 136.5410},
    "Miami":        {"country": "USA",        "laps": 57,
                      "raw": MIAMI_RAW, "scale": 82,
                      "rain_chance": 0.10, "lat": 25.9581, "lon": -80.2389},
    "Canada":       {"country": "Canada",     "laps": 70,
                      "raw": CANADA_RAW, "scale": 82,
                      "rain_chance": 0.35, "lat": 45.5000, "lon": -73.5228},
    "Monaco":       {"country": "Monaco",     "laps": 78,
                      "raw": MONACO_RAW, "scale": 46,
                      "rain_chance": 0.15, "lat": 43.7347, "lon": 7.4206},
    "Barcelona":    {"country": "Spain",      "laps": 66,
                      "raw": BARCELONA_RAW, "scale": 82,
                      "rain_chance": 0.10, "lat": 41.5700, "lon": 2.2611},
    "Austria":      {"country": "Austria",    "laps": 71,
                      "raw": AUSTRIA_RAW, "scale": 90,
                      "rain_chance": 0.35, "lat": 47.2197, "lon": 14.7647},
    "Silverstone":  {"country": "UK",         "laps": 52,
                      "raw": SILVERSTONE_RAW, "scale": 75,
                      "rain_chance": 0.55, "lat": 52.0786, "lon": -1.0169},
    "Spa":          {"country": "Belgium",    "laps": 44,
                      "raw": SPA_RAW, "scale": 85,
                      "rain_chance": 0.45, "lat": 50.4372, "lon": 5.9714},
    "Hungary":      {"country": "Hungary",    "laps": 70,
                      "raw": HUNGARORING_RAW, "scale": 78,
                      "rain_chance": 0.20, "lat": 47.5789, "lon": 19.2486},
    "Netherlands":  {"country": "Netherlands", "laps": 72,
                      "raw": ZANDVOORT_RAW, "scale": 78,
                      "rain_chance": 0.40, "lat": 52.3888, "lon": 4.5409},
    "Monza":        {"country": "Italy",      "laps": 53,
                      "raw": MONZA_RAW, "scale": 78,
                      "rain_chance": 0.10, "lat": 45.6156, "lon": 9.2811},
    "Madrid":       {"country": "Spain",      "laps": 61,
                      "raw": MADRING_RAW, "scale": 82,
                      "rain_chance": 0.10, "lat": 40.4700, "lon": -3.6200},
    "Azerbaijan":   {"country": "Azerbaijan", "laps": 51,
                      "raw": BAKU_RAW, "scale": 75,
                      "rain_chance": 0.10, "lat": 40.3725, "lon": 49.8533},
    "Singapore":    {"country": "Singapore",  "laps": 62,
                      "raw": SINGAPORE_RAW, "scale": 80,
                      "rain_chance": 0.55, "lat": 1.2914, "lon": 103.8640},
    "Austin":       {"country": "USA",        "laps": 56,
                      "raw": COTA_RAW, "scale": 80,
                      "rain_chance": 0.15, "lat": 30.1328, "lon": -97.6411},
    "Mexico":       {"country": "Mexico",     "laps": 71,
                      "raw": MEXICO_RAW, "scale": 82,
                      "rain_chance": 0.10, "lat": 19.4042, "lon": -99.0907},
    "Brazil":       {"country": "Brazil",     "laps": 71,
                      "raw": INTERLAGOS_RAW, "scale": 80,
                      "rain_chance": 0.45, "lat": -23.7036, "lon": -46.6997},
    "Las Vegas":    {"country": "USA",        "laps": 50,
                      "raw": VEGAS_RAW, "scale": 78,
                      "rain_chance": 0.05, "lat": 36.1147, "lon": -115.1728},
    "Qatar":        {"country": "Qatar",      "laps": 57,
                      "raw": QATAR_RAW, "scale": 80,
                      "rain_chance": 0.05, "lat": 25.4900, "lon": 51.4542},
    "Abu Dhabi":    {"country": "UAE",        "laps": 58,
                      "raw": YAS_MARINA_RAW, "scale": 80,
                      "rain_chance": 0.05, "lat": 24.4672, "lon": 54.6031},
}

WEATHER_STATES = ["Sunny", "Cloudy", "Light Rain", "Heavy Rain"]
WEATHER_SPEED_FACTOR = {"Sunny": 1.0, "Cloudy": 0.98, "Light Rain": 0.85, "Heavy Rain": 0.68}
WEATHER_INCIDENT_MULT = {"Sunny": 1.0, "Cloudy": 1.1, "Light Rain": 2.2, "Heavy Rain": 4.0}

# Open-Meteo's current_weather "weathercode" is a WMO code - map each one to
# whichever of our four WEATHER_STATES it's closest to.
WEATHERCODE_TO_STATE = {
    0: "Sunny", 1: "Sunny", 2: "Sunny",
    3: "Cloudy", 45: "Cloudy", 48: "Cloudy",
    51: "Light Rain", 53: "Light Rain", 55: "Light Rain",
    56: "Light Rain", 57: "Light Rain",
    61: "Light Rain", 63: "Light Rain",
    65: "Heavy Rain", 66: "Heavy Rain", 67: "Heavy Rain",
    71: "Heavy Rain", 73: "Heavy Rain", 75: "Heavy Rain", 77: "Heavy Rain",
    80: "Light Rain", 81: "Light Rain", 82: "Heavy Rain",
    85: "Heavy Rain", 86: "Heavy Rain",
    95: "Heavy Rain", 96: "Heavy Rain", 99: "Heavy Rain",
}


def fetch_current_weather_state(lat, lon, fallback):
    """Look up the real current weather at (lat, lon) via Open-Meteo and map
    it onto one of WEATHER_STATES, following the same request/parse pattern
    as the weather app exercise: build the URL, requests.get with a short
    timeout, raise_for_status, read current_weather out of the JSON. Any
    failure (offline, timeout, bad response) falls back to `fallback` instead
    of crashing the race setup.

    Returns (state, is_live) - is_live is True only when the API call
    actually succeeded, so callers can show that the live lookup worked
    rather than just silently using it.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        code = data["current_weather"]["weathercode"]
        state = WEATHERCODE_TO_STATE.get(code, fallback)
        print(f"[weather] live conditions at ({lat}, {lon}): code {code} -> {state}")
        return state, True
    except Exception as e:
        print(f"[weather] live fetch failed ({e}), using fallback: {fallback}")
        return fallback, False


# --------------------------------------------------------------------------- #
# Track geometry helpers
# --------------------------------------------------------------------------- #

SECTOR_COLORS = [(230, 55, 55), (60, 140, 230), (255, 205, 40)]
DRS_GREEN = (60, 210, 100)
SPEED_TRAP_MAGENTA = (210, 70, 220)


class Track:
    def __init__(self, name, info, center):
        self.name = name
        self.info = info
        self.center = center
        cx, cy = center
        # corners: the sparse hand-placed waypoints - used to anchor turn
        # numbers, sectors, DRS zones and the speed trap
        self.corners = _make_from_raw(cx, cy, info["raw"], info["scale"])
        # points: a dense, rounded-corner curve derived from the waypoints -
        # used for the drawn road and for car movement, so both look and
        # move like a real track instead of a sharp-angled polygon
        self.points = _chaikin_smooth(self.corners)

        # cumulative arc length table over the smoothed curve, for
        # smooth even-speed interpolation of car positions
        self.seg_lengths = []
        total = 0.0
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            d = math.hypot(x2 - x1, y2 - y1)
            self.seg_lengths.append(d)
            total += d
        self.total_length = total

        # official-map-style features derived from the corner waypoints:
        # three roughly-equal sectors (by corner count), a speed trap on the
        # longest straight, and up to two DRS zones on the next-longest ones
        cn = len(self.corners)
        corner_seg_lengths = []
        for i in range(cn):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % cn]
            corner_seg_lengths.append(math.hypot(x2 - x1, y2 - y1))

        self.sector_cuts = (cn // 3, (2 * cn) // 3)
        by_length = sorted(range(cn), key=lambda i: -corner_seg_lengths[i])
        self.speed_trap_idx = by_length[0]
        longest = corner_seg_lengths[self.speed_trap_idx]
        self.drs_zone_idxs = [
            i for i in by_length[1:6] if corner_seg_lengths[i] > longest * 0.35
        ][:2]
        self.centroid = (
            sum(p[0] for p in self.corners) / cn,
            sum(p[1] for p in self.corners) / cn,
        )

    def sector_of(self, dense_seg_idx):
        corner_idx = (dense_seg_idx // POINTS_PER_CORNER) % len(self.corners)
        c1, c2 = self.sector_cuts
        if corner_idx < c1:
            return 0
        if corner_idx < c2:
            return 1
        return 2

    def point_at(self, frac):
        """frac in [0,1) -> (x,y) position around the loop, arc-length based."""
        frac = frac % 1.0
        target = frac * self.total_length
        acc = 0.0
        n = len(self.points)
        for i in range(n):
            seg = self.seg_lengths[i]
            if acc + seg >= target or i == n - 1:
                local_t = 0 if seg == 0 else (target - acc) / seg
                x1, y1 = self.points[i]
                x2, y2 = self.points[(i + 1) % n]
                return (x1 + (x2 - x1) * local_t, y1 + (y2 - y1) * local_t)
            acc += seg
        return self.points[0]


# --------------------------------------------------------------------------- #
# Race entities
# --------------------------------------------------------------------------- #

class Driver:
    def __init__(self, code, name, team):
        self.code = code
        self.name = name
        self.team = team
        self.number = DRIVER_NUMBERS[code]
        self.color = TEAMS[team]["color"]
        self.base_pace = TEAMS[team]["pace"] * random.uniform(0.985, 1.015)
        self.grid_pos = 0
        self.laps_completed = 0
        self.lap_frac = 0.0          # progress through current lap [0,1)
        self.distance = 0.0          # total normalized distance (laps + frac)
        self.finished = False
        self.finish_time = None
        self.dnf = False
        self.dnf_reason = None
        self.dnf_timer = 0.0
        self.pos_xy = (0, 0)
        self.current_pos = 0
        self.pit_penalty_timer = 0.0

    def total_progress(self):
        return self.laps_completed + self.lap_frac


# --------------------------------------------------------------------------- #
# Incident / effects
# --------------------------------------------------------------------------- #

class FlagEvent:
    def __init__(self, kind, message, duration=8.0):
        self.kind = kind          # "yellow" | "safety_car" | "info"
        self.message = message
        self.timer = duration


class CrashFX:
    def __init__(self, pos):
        self.pos = pos
        self.age = 0.0
        self.duration = 1.4

    def done(self):
        return self.age >= self.duration

    def draw(self, surf, transform=lambda x, y: (x, y), scale=1.0):
        t = self.age / self.duration
        x, y = transform(*self.pos)
        n_particles = 10
        for i in range(n_particles):
            ang = (i / n_particles) * 2 * math.pi
            r = (6 + 40 * t) * scale
            px = x + math.cos(ang) * r
            py = y + math.sin(ang) * r
            size = max(1, int((4 * (1 - t) + 1) * scale))
            color = (255, int(160 * (1 - t) + 60), 0)
            pygame.draw.circle(surf, color, (int(px), int(py)), size)
        ring_r = int((10 + 50 * t) * scale)
        if ring_r > 0:
            s = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
            alpha = max(0, int(200 * (1 - t)))
            pygame.draw.circle(s, (255, 255, 255, alpha), (ring_r + 2, ring_r + 2), ring_r, 2)
            surf.blit(s, (x - ring_r - 2, y - ring_r - 2))


# --------------------------------------------------------------------------- #
# Main game
# --------------------------------------------------------------------------- #

class Game:
    STATE_DRIVER_SELECT = "driver_select"
    STATE_CIRCUIT_SELECT = "circuit_select"
    STATE_GRID = "grid"
    STATE_RACE = "race"
    STATE_FINISH = "finish"

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("F1 Race Simulator")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_tiny = pygame.font.SysFont(FONT_NAME, 10, bold=True)
        self.font_tag = pygame.font.SysFont(FONT_NAME, 12, bold=True)
        self.font_small = pygame.font.SysFont(FONT_NAME, 14)
        self.font = pygame.font.SysFont(FONT_NAME, 18)
        self.font_bold = pygame.font.SysFont(FONT_NAME, 20, bold=True)
        self.font_big = pygame.font.SysFont(FONT_NAME, 36, bold=True)
        self.font_huge = pygame.font.SysFont(FONT_NAME, 56, bold=True)

        self.state = self.STATE_DRIVER_SELECT
        self.player_driver_code = None
        self.selected_circuit_name = None

        self.drivers = []
        self.track = None
        self.circuit_info = None
        self._preview_tracks = {}

        self.sim_elapsed = 0.0        # seconds of race-time simulated
        self.wall_elapsed = 0.0       # seconds of real time elapsed this run
        self.finish_order = []
        self.events_log = []

        self.weather = "Sunny"
        self.weather_timer = 0.0
        self.weather_is_live = False  # True only while showing the live-fetched initial weather

        self.flags = []
        self.crash_fx = []
        self.safety_car_active_timer = 0.0

        self.next_incident_check = random.uniform(4, 9)

        self.hovered_driver = None

        self.driver_select_rects = []
        self.circuit_select_rects = []
        self.next_button_rect = None
        self.back_button_rect = None
        self.restart_button_rect = None
        self.circuit_scroll = 0
        self.circuit_scroll_max = 0

        # camera state for the live race view
        self.zoom = 1.0
        self.zoom_min, self.zoom_max = 0.6, 3.0
        self.pan = [0.0, 0.0]
        self.track_center = (WIDTH * 0.5 - 100, HEIGHT * 0.5 + 10)
        self.zoom_in_rect = None
        self.zoom_out_rect = None
        self.zoom_reset_rect = None
        self._panning = False
        self._pan_start_mouse = (0, 0)
        self._pan_start_pan = (0.0, 0.0)

    # ---------------------------------------------------------------- setup

    def setup_race(self):
        random.shuffle(DRIVERS)
        self.drivers = [Driver(code, name, team) for code, name, team in DRIVERS]
        random.shuffle(self.drivers)  # random starting grid
        for i, d in enumerate(self.drivers):
            d.grid_pos = i + 1

        info = CIRCUITS[self.selected_circuit_name]
        self.circuit_info = info
        self.track = Track(self.selected_circuit_name, info, self.track_center)
        self.zoom = 1.0
        self.pan = [0.0, 0.0]

        self.sim_elapsed = 0.0
        self.wall_elapsed = 0.0
        self.finish_order = []
        self.events_log = []
        self.flags = []
        self.crash_fx = []
        self.safety_car_active_timer = 0.0
        self.next_incident_check = random.uniform(4, 9)

        # initial weather: try the circuit's real current weather first,
        # falling back to the old rain-chance-biased random pick if the
        # live lookup fails (offline, API down, etc.)
        fallback_weather = "Sunny" if random.random() > info["rain_chance"] else "Cloudy"
        self.weather, self.weather_is_live = fetch_current_weather_state(
            info["lat"], info["lon"], fallback_weather
        )
        self.weather_timer = random.uniform(90, 180)

        # total race distance in laps must complete in RACE_DURATION_SECONDS of
        # wall-clock time on average; per-driver pace varies around this.
        self.target_avg_lap_time = RACE_DURATION_SECONDS / info["laps"]

    # ---------------------------------------------------------------- update

    def update_race(self, dt):
        if self.wall_elapsed >= RACE_DURATION_SECONDS and not self._all_done():
            # time's up - anyone still running is classified where they are
            for d in self.drivers:
                if not d.finished and not d.dnf:
                    d.finished = True
                    d.finish_time = RACE_DURATION_SECONDS
            self.state = self.STATE_FINISH
            self._build_finish_order()
            return

        self.wall_elapsed += dt

        # weather evolution
        self.weather_timer -= dt
        if self.weather_timer <= 0:
            self._maybe_change_weather()
            self.weather_timer = random.uniform(120, 240)

        # flags countdown
        for f in self.flags:
            f.timer -= dt
        self.flags = [f for f in self.flags if f.timer > 0]

        yellow_active = any(f.kind in ("yellow", "safety_car") for f in self.flags)
        speed_mult = WEATHER_SPEED_FACTOR[self.weather] * (0.55 if yellow_active else 1.0)

        laps_total = self.circuit_info["laps"]

        for d in self.drivers:
            if d.finished or d.dnf:
                if d.dnf:
                    d.dnf_timer -= dt
                continue

            if d.pit_penalty_timer > 0:
                d.pit_penalty_timer -= dt
                continue

            # base progress per second, scaled so an "average" driver completes
            # the race in RACE_DURATION_SECONDS
            base_frac_per_sec = 1.0 / self.target_avg_lap_time
            variability = random.uniform(0.92, 1.08)
            frac_delta = base_frac_per_sec * d.base_pace * variability * speed_mult * dt

            d.lap_frac += frac_delta
            if d.lap_frac >= 1.0:
                d.lap_frac -= 1.0
                d.laps_completed += 1
                if d.laps_completed >= laps_total:
                    d.finished = True
                    d.finish_time = self.wall_elapsed
                    self.finish_order.append(d)

            d.pos_xy = self.track.point_at(d.total_progress())

        # random incidents
        self.next_incident_check -= dt
        if self.next_incident_check <= 0:
            self._maybe_trigger_incident()
            base = random.uniform(10, 22)
            mult = WEATHER_INCIDENT_MULT[self.weather]
            self.next_incident_check = base / mult

        # crash fx aging
        for fx in self.crash_fx:
            fx.age += dt
        self.crash_fx = [fx for fx in self.crash_fx if not fx.done()]

        # live standings
        self._update_positions()

        if self._all_done():
            self.state = self.STATE_FINISH
            self._build_finish_order()

    def _all_done(self):
        return all(d.finished or d.dnf for d in self.drivers)

    def _update_positions(self):
        ranked = sorted(
            self.drivers,
            key=lambda d: (
                d.dnf,                               # DNFs go to bottom
                -(d.laps_completed + d.lap_frac) if not d.dnf else 0,
            ),
        )
        for i, d in enumerate(ranked):
            d.current_pos = i + 1

    def _maybe_change_weather(self):
        rain_chance = self.circuit_info["rain_chance"]
        roll = random.random()
        if roll < rain_chance * 0.5:
            new_weather = random.choice(["Light Rain", "Heavy Rain"])
        elif roll < rain_chance:
            new_weather = "Cloudy"
        else:
            new_weather = random.choice(["Sunny", "Cloudy"])
        if new_weather != self.weather:
            self._log(f"Weather changing to {new_weather} at {self.selected_circuit_name}")
            self.weather = new_weather
            self.weather_is_live = False  # sim has taken over - no longer showing the live-fetched value

    def _maybe_trigger_incident(self):
        active_drivers = [d for d in self.drivers if not d.finished and not d.dnf]
        if not active_drivers:
            return
        roll = random.random()
        d = random.choice(active_drivers)

        if roll < 0.45:
            # yellow flag - minor incident, no DNF
            self.flags.append(FlagEvent("yellow", f"Yellow flag: {d.name} off track", duration=10))
            self.crash_fx.append(CrashFX(d.pos_xy))
            self._log(f"Lap {d.laps_completed+1}: Yellow flag for {d.name} ({d.team})")
        elif roll < 0.65:
            # DNF
            reasons = ["engine failure", "collision damage", "hydraulic failure",
                       "gearbox issue", "spun off", "brake failure"]
            reason = random.choice(reasons)
            d.dnf = True
            d.dnf_reason = reason
            d.dnf_timer = 999
            self.flags.append(FlagEvent("yellow", f"DNF: {d.name} - {reason}", duration=10))
            self.crash_fx.append(CrashFX(d.pos_xy))
            self._log(f"Lap {d.laps_completed+1}: {d.name} retires - {reason}")
        elif roll < 0.8:
            # safety car
            self.flags.append(FlagEvent("safety_car", "SAFETY CAR DEPLOYED", duration=14))
            self._log(f"Lap {d.laps_completed+1}: Safety car deployed")
        else:
            # brief pit stop penalty (adds a little variance/drama)
            d.pit_penalty_timer = random.uniform(1.5, 3.0)
            self._log(f"Lap {d.laps_completed+1}: {d.name} pits")

    def _log(self, msg):
        self.events_log.append(msg)
        self.events_log = self.events_log[-6:]

    def _build_finish_order(self):
        ranked = sorted(
            self.drivers,
            key=lambda d: (
                d.dnf,
                -(d.laps_completed + d.lap_frac) if not d.dnf else 0,
                d.finish_time if d.finish_time is not None else 9e9,
            ),
        )
        self.finish_order = ranked

    # ---------------------------------------------------------------- draw: driver select

    def draw_driver_select(self):
        self.screen.fill(BG)
        title = self.font_huge.render("Choose Your Driver", True, TEXT)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

        sub = self.font.render("Click a driver to select them for the race", True, DIM_TEXT)
        self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 90))

        cols = 4
        card_w, card_h = 280, 70
        gap = 16
        start_x = (WIDTH - (cols * card_w + (cols - 1) * gap)) // 2
        start_y = 140

        self.driver_select_rects = []
        for i, (code, name, team) in enumerate(DRIVERS):
            row = i // cols
            col = i % cols
            x = start_x + col * (card_w + gap)
            y = start_y + row * (card_h + gap)
            rect = pygame.Rect(x, y, card_w, card_h)
            self.driver_select_rects.append((rect, code))

            selected = (self.player_driver_code == code)
            color = TEAMS[team]["color"]
            bg_color = (50, 50, 60) if not selected else (70, 70, 40)
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
            border_col = ACCENT if selected else PANEL_BORDER
            pygame.draw.rect(self.screen, border_col, rect, 2, border_radius=8)

            pygame.draw.circle(self.screen, color, (x + 22, y + card_h // 2), 12)
            num_txt = self.font_small.render(str(DRIVER_NUMBERS[code]), True, (255, 255, 255))
            self.screen.blit(num_txt, (x + 22 - num_txt.get_width() // 2, y + card_h // 2 - 7))

            name_txt = self.font_bold.render(name, True, TEXT)
            self.screen.blit(name_txt, (x + 44, y + 12))
            team_txt = self.font_small.render(team, True, DIM_TEXT)
            self.screen.blit(team_txt, (x + 44, y + 36))

        if self.player_driver_code:
            btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 90, 200, 50)
            self.next_button_rect = btn
            pygame.draw.rect(self.screen, GREEN, btn, border_radius=8)
            txt = self.font_bold.render("Next: Circuit", True, (10, 30, 10))
            self.screen.blit(txt, (btn.centerx - txt.get_width() // 2, btn.centery - txt.get_height() // 2))
        else:
            self.next_button_rect = None

    def handle_driver_select_click(self, pos):
        for rect, code in self.driver_select_rects:
            if rect.collidepoint(pos):
                self.player_driver_code = code
                return
        if self.next_button_rect and self.next_button_rect.collidepoint(pos):
            self.state = self.STATE_CIRCUIT_SELECT

    # ---------------------------------------------------------------- draw: circuit select

    def draw_circuit_select(self):
        self.screen.fill(BG)
        title = self.font_huge.render("Choose Your Circuit", True, TEXT)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 24))

        driver_name = dict((c, n) for c, n, t in DRIVERS)[self.player_driver_code]
        sub = self.font.render(f"Driving as {driver_name}  ·  {len(CIRCUITS)} circuits on the calendar",
                                True, DIM_TEXT)
        self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 78))

        names = list(CIRCUITS.keys())
        card_w, card_h = 300, 240
        gap = 16
        cols = 4
        start_x = (WIDTH - (cols * card_w + (cols - 1) * gap)) // 2

        viewport_top, viewport_bottom = 116, HEIGHT - 108
        viewport_h = viewport_bottom - viewport_top
        rows = -(-len(names) // cols)
        content_h = rows * (card_h + gap) - gap
        self.circuit_scroll_max = max(0, content_h - viewport_h)
        self.circuit_scroll = max(0, min(self.circuit_scroll, self.circuit_scroll_max))

        self.screen.set_clip(pygame.Rect(0, viewport_top, WIDTH, viewport_h))

        self.circuit_select_rects = []
        for i, cname in enumerate(names):
            info = CIRCUITS[cname]
            row = i // cols
            col = i % cols
            x = start_x + col * (card_w + gap)
            y = viewport_top + row * (card_h + gap) - self.circuit_scroll
            rect = pygame.Rect(x, y, card_w, card_h)
            if y + card_h < viewport_top or y > viewport_bottom:
                continue
            viewport_rect = pygame.Rect(0, viewport_top, WIDTH, viewport_h)
            self.circuit_select_rects.append((rect.clip(viewport_rect), cname))

            selected = (self.selected_circuit_name == cname)
            bg_color = (50, 50, 60) if not selected else (70, 70, 40)
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=10)
            border_col = ACCENT if selected else PANEL_BORDER
            pygame.draw.rect(self.screen, border_col, rect, 2, border_radius=10)

            name_txt = self.font_bold.render(cname, True, TEXT)
            self.screen.blit(name_txt, (x + 12, y + 8))
            meta_txt = self.font_small.render(
                f"{info['country']}  ·  {info['laps']} laps  ·  Rain {int(info['rain_chance']*100)}%",
                True, DIM_TEXT,
            )
            self.screen.blit(meta_txt, (x + 12, y + 32))

            # mini official-style track preview, auto-fit to the card
            ptrack = self._preview_tracks.setdefault(cname, Track(cname, info, (0, 0)))
            xs = [p[0] for p in ptrack.points]
            ys = [p[1] for p in ptrack.points]
            bbox_w = max(xs) - min(xs) or 1
            bbox_h = max(ys) - min(ys) or 1
            bbox_cx = (max(xs) + min(xs)) / 2
            bbox_cy = (max(ys) + min(ys)) / 2
            preview_top = y + 56
            preview_h = card_h - 66
            avail_w, avail_h = card_w - 24, preview_h - 8
            factor = min(avail_w / bbox_w, avail_h / bbox_h) * 0.92
            anchor = (x + card_w / 2, preview_top + preview_h / 2)
            transform = (lambda px, py, f=factor, a=anchor, bc=(bbox_cx, bbox_cy):
                         (a[0] + (px - bc[0]) * f, a[1] + (py - bc[1]) * f))
            self.draw_circuit_map(self.screen, ptrack, transform, factor, self.font_tiny,
                                   show_tags=False, min_gap=10)

        self.screen.set_clip(None)

        # scrollbar
        if self.circuit_scroll_max > 0:
            track_rect = pygame.Rect(WIDTH - 14, viewport_top, 6, viewport_h)
            pygame.draw.rect(self.screen, PANEL_BORDER, track_rect, border_radius=3)
            thumb_h = max(30, int(viewport_h * viewport_h / content_h))
            thumb_y = viewport_top + int((viewport_h - thumb_h) * (self.circuit_scroll / self.circuit_scroll_max))
            pygame.draw.rect(self.screen, ACCENT, pygame.Rect(WIDTH - 14, thumb_y, 6, thumb_h), border_radius=3)

        if self.selected_circuit_name:
            btn = pygame.Rect(WIDTH // 2 - 110, HEIGHT - 90, 220, 50)
            self.next_button_rect = btn
            pygame.draw.rect(self.screen, GREEN, btn, border_radius=8)
            txt = self.font_bold.render("Start Race Weekend", True, (10, 30, 10))
            self.screen.blit(txt, (btn.centerx - txt.get_width() // 2, btn.centery - txt.get_height() // 2))
        else:
            self.next_button_rect = None

        back_btn = pygame.Rect(30, HEIGHT - 90, 140, 50)
        self.back_button_rect = back_btn
        pygame.draw.rect(self.screen, (70, 70, 80), back_btn, border_radius=8)
        txt = self.font.render("Back", True, TEXT)
        self.screen.blit(txt, (back_btn.centerx - txt.get_width() // 2, back_btn.centery - txt.get_height() // 2))

    def handle_circuit_select_click(self, pos):
        for rect, cname in self.circuit_select_rects:
            if rect.collidepoint(pos):
                self.selected_circuit_name = cname
                return
        if self.next_button_rect and self.next_button_rect.collidepoint(pos):
            self.setup_race()
            self.state = self.STATE_GRID
        if self.back_button_rect and self.back_button_rect.collidepoint(pos):
            self.state = self.STATE_DRIVER_SELECT

    def handle_circuit_select_scroll(self, dy):
        self.circuit_scroll = max(0, min(self.circuit_scroll - dy * 40, self.circuit_scroll_max))

    # ---------------------------------------------------------------- draw: grid

    def draw_grid(self):
        self.screen.fill(BG)
        title = self.font_big.render(f"Starting Grid - {self.selected_circuit_name}", True, TEXT)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 24))
        sub = self.font.render("Grid order randomised. Click Lights Out to begin.", True, DIM_TEXT)
        self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 64))

        ordered = sorted(self.drivers, key=lambda d: d.grid_pos)
        col_w = WIDTH // 2
        row_h = 34
        start_y = 110
        drivers_per_col = -(-len(ordered) // 2)  # ceil division, so 2 columns always fit
        for i, d in enumerate(ordered):
            col = i // drivers_per_col
            row = i % drivers_per_col
            x = 140 + col * col_w
            y = start_y + row * row_h
            is_player = d.code == self.player_driver_code
            if is_player:
                rect = pygame.Rect(x - 10, y - 4, col_w - 60, row_h - 4)
                pygame.draw.rect(self.screen, (60, 60, 30), rect, border_radius=6)
                pygame.draw.rect(self.screen, ACCENT, rect, 2, border_radius=6)
            pos_txt = self.font_bold.render(f"P{d.grid_pos}", True, DIM_TEXT)
            self.screen.blit(pos_txt, (x, y))
            pygame.draw.circle(self.screen, d.color, (x + 60, y + 10), 8)
            name_txt = self.font.render(f"{d.name} ({d.team})", True, TEXT if not is_player else ACCENT)
            self.screen.blit(name_txt, (x + 80, y))

        btn = pygame.Rect(WIDTH // 2 - 110, HEIGHT - 70, 220, 50)
        self.next_button_rect = btn
        pygame.draw.rect(self.screen, RED_FLAG, btn, border_radius=8)
        txt = self.font_bold.render("Lights Out!", True, (255, 255, 255))
        self.screen.blit(txt, (btn.centerx - txt.get_width() // 2, btn.centery - txt.get_height() // 2))

    def handle_grid_click(self, pos):
        if self.next_button_rect and self.next_button_rect.collidepoint(pos):
            self.state = self.STATE_RACE

    # ---------------------------------------------------------------- camera (zoom/pan)

    def world_to_screen(self, x, y):
        cx, cy = self.track_center
        return (cx + (x - cx) * self.zoom + self.pan[0],
                cy + (y - cy) * self.zoom + self.pan[1])

    def screen_to_world(self, sx, sy):
        cx, cy = self.track_center
        return (cx + (sx - cx - self.pan[0]) / self.zoom,
                cy + (sy - cy - self.pan[1]) / self.zoom)

    def zoom_at(self, screen_pos, factor):
        new_zoom = max(self.zoom_min, min(self.zoom_max, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        wx, wy = self.screen_to_world(*screen_pos)
        self.zoom = new_zoom
        cx, cy = self.track_center
        self.pan[0] = screen_pos[0] - cx - (wx - cx) * self.zoom
        self.pan[1] = screen_pos[1] - cy - (wy - cy) * self.zoom

    # ---------------------------------------------------------------- draw: official-style circuit map

    def draw_circuit_map(self, surf, track, transform, line_scale, font,
                          show_tags=True, min_gap=14):
        """Draw a track the way F1's official broadcast maps look: a dark
        tarmac band with a colored sector accent, numbered turn markers,
        DRS zone / speed trap tags and a chequered start/finish line."""
        pts = [transform(x, y) for x, y in track.points]
        n = len(pts)
        corner_pts = [transform(x, y) for x, y in track.corners]
        cn = len(corner_pts)

        edge_w = max(3, round(28 * line_scale))
        body_w = max(2, round(20 * line_scale))
        accent_w = max(1, round(6 * line_scale))

        pygame.draw.polygon(surf, TRACK_EDGE, pts, edge_w)
        pygame.draw.polygon(surf, TRACK_COLOR, pts, body_w)
        for i in range(n):
            color = SECTOR_COLORS[track.sector_of(i)]
            pygame.draw.line(surf, color, pts[i], pts[(i + 1) % n], accent_w)

        # chequered start/finish line, perpendicular to the track direction
        sx, sy = corner_pts[0]
        nx, ny = corner_pts[1 % cn]
        dx, dy = nx - sx, ny - sy
        dist = math.hypot(dx, dy) or 1
        ux, uy = dx / dist, dy / dist
        pxv, pyv = -uy, ux
        cell = max(2, round(4 * line_scale))
        rows = 6
        for r in range(rows):
            v = (r - rows / 2 + 0.5) * cell
            col = (255, 255, 255) if r % 2 == 0 else (25, 25, 28)
            base = (sx + pxv * v, sy + pyv * v)
            poly = [
                (base[0] - ux * cell / 2, base[1] - uy * cell / 2),
                (base[0] + ux * cell / 2, base[1] + uy * cell / 2),
            ]
            poly = [
                (poly[0][0] - pxv * cell / 2, poly[0][1] - pyv * cell / 2),
                (poly[1][0] - pxv * cell / 2, poly[1][1] - pyv * cell / 2),
                (poly[1][0] + pxv * cell / 2, poly[1][1] + pyv * cell / 2),
                (poly[0][0] + pxv * cell / 2, poly[0][1] + pyv * cell / 2),
            ]
            pygame.draw.polygon(surf, col, poly)

        if not show_tags:
            return

        centroid = transform(*track.centroid)

        def push_out(p, dist_px):
            vx, vy = p[0] - centroid[0], p[1] - centroid[1]
            d = math.hypot(vx, vy) or 1
            return (p[0] + vx / d * dist_px, p[1] + vy / d * dist_px)

        # numbered turn markers - decluttered so tight chicanes don't overlap
        last = None
        label_r = max(4, round(7 * line_scale))
        label_dist = max(10, round(20 * line_scale))
        for i in range(1, cn):
            p = corner_pts[i]
            if last is not None and math.hypot(p[0] - last[0], p[1] - last[1]) < min_gap:
                continue
            last = p
            lp = push_out(p, label_dist)
            pygame.draw.circle(surf, (20, 20, 24), lp, label_r)
            pygame.draw.circle(surf, (225, 225, 230), lp, label_r, 1)
            txt = font.render(str(i), True, (225, 225, 230))
            surf.blit(txt, (lp[0] - txt.get_width() / 2, lp[1] - txt.get_height() / 2))

        tag_dist = max(20, round(46 * line_scale))
        for zi, idx in enumerate(track.drs_zone_idxs):
            p = corner_pts[idx]
            lp = push_out(p, tag_dist)
            pygame.draw.line(surf, DRS_GREEN, p, lp, max(1, round(2 * line_scale)))
            pygame.draw.circle(surf, DRS_GREEN, (int(p[0]), int(p[1])), max(2, round(3 * line_scale)))
            self._draw_map_tag(surf, lp, f"DRS ZONE {zi + 1}", DRS_GREEN, font)

        p = corner_pts[track.speed_trap_idx]
        lp = push_out(p, tag_dist)
        pygame.draw.line(surf, SPEED_TRAP_MAGENTA, p, lp, max(1, round(2 * line_scale)))
        pygame.draw.circle(surf, SPEED_TRAP_MAGENTA, (int(p[0]), int(p[1])), max(2, round(3 * line_scale)))
        self._draw_map_tag(surf, lp, "SPEED TRAP", SPEED_TRAP_MAGENTA, font)

    def _draw_map_tag(self, surf, center, text, color, font):
        txt = font.render(text, True, (15, 15, 18))
        pad_x, pad_y = 6, 3
        rect = pygame.Rect(0, 0, txt.get_width() + pad_x * 2, txt.get_height() + pad_y * 2)
        rect.center = (int(center[0]), int(center[1]))
        pygame.draw.rect(surf, color, rect, border_radius=4)
        surf.blit(txt, (rect.x + pad_x, rect.y + pad_y))

    def draw_zoom_controls(self):
        size, gap = 34, 6
        bx, by = 20, HEIGHT - 150
        self.zoom_in_rect = pygame.Rect(bx, by, size, size)
        self.zoom_out_rect = pygame.Rect(bx, by + size + gap, size, size)
        self.zoom_reset_rect = pygame.Rect(bx, by + 2 * (size + gap), size, size)
        for rect, label in ((self.zoom_in_rect, "+"), (self.zoom_out_rect, "-"), (self.zoom_reset_rect, "R")):
            pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
            pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)
            txt = self.font_bold.render(label, True, TEXT)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
        pct_txt = self.font_small.render(f"{int(self.zoom * 100)}%", True, DIM_TEXT)
        self.screen.blit(pct_txt, (bx, by - 18))
        hint_txt = self.font_small.render("scroll / right-drag to pan", True, DIM_TEXT)
        self.screen.blit(hint_txt, (bx, by + 3 * (size + gap) + 4))

    def handle_race_click(self, pos):
        if self.zoom_in_rect and self.zoom_in_rect.collidepoint(pos):
            self.zoom_at(self.track_center, 1.25)
        elif self.zoom_out_rect and self.zoom_out_rect.collidepoint(pos):
            self.zoom_at(self.track_center, 0.8)
        elif self.zoom_reset_rect and self.zoom_reset_rect.collidepoint(pos):
            self.zoom = 1.0
            self.pan = [0.0, 0.0]

    def draw_race(self):
        self.screen.fill(BG)
        self.draw_circuit_map(self.screen, self.track, self.world_to_screen, self.zoom,
                               self.font_tag, show_tags=True, min_gap=16)

        # driver dots
        self.hovered_driver = None
        mouse_pos = pygame.mouse.get_pos()
        for d in self.drivers:
            if d.dnf:
                continue
            x, y = self.world_to_screen(*d.pos_xy)
            radius = max(2, (7 if d.code != self.player_driver_code else 9) * self.zoom)
            pygame.draw.circle(self.screen, d.color, (int(x), int(y)), int(radius))
            if d.code == self.player_driver_code:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(radius + 3), 2)
            dist = math.hypot(mouse_pos[0] - x, mouse_pos[1] - y)
            if dist <= radius + 6:
                self.hovered_driver = d

        # stopped DNF cars (greyed out, sitting where they stopped)
        for d in self.drivers:
            if d.dnf:
                x, y = self.world_to_screen(*d.pos_xy)
                pygame.draw.circle(self.screen, (90, 90, 90), (int(x), int(y)), max(2, int(6 * self.zoom)))

        # crash fx
        for fx in self.crash_fx:
            fx.draw(self.screen, self.world_to_screen, self.zoom)

        self.draw_hud()
        self.draw_leaderboard()
        self.draw_flag_banner()
        self.draw_zoom_controls()
        if self.hovered_driver:
            self.draw_tooltip(self.hovered_driver, mouse_pos)

    def draw_hud(self):
        remaining = max(0, RACE_DURATION_SECONDS - self.wall_elapsed)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        timer_txt = self.font_big.render(f"{mins:02d}:{secs:02d}", True, TEXT)
        self.screen.blit(timer_txt, (20, 16))

        circuit_txt = self.font.render(self.selected_circuit_name, True, DIM_TEXT)
        self.screen.blit(circuit_txt, (20, 60))

        weather_col = {
            "Sunny": (255, 210, 60), "Cloudy": (180, 180, 190),
            "Light Rain": (100, 160, 255), "Heavy Rain": (60, 100, 220),
        }[self.weather]
        weather_txt = self.font.render(f"Weather: {self.weather}", True, weather_col)
        self.screen.blit(weather_txt, (20, 84))
        if self.weather_is_live:
            badge_center = (20 + weather_txt.get_width() + 28, 84 + weather_txt.get_height() // 2)
            self._draw_map_tag(self.screen, badge_center, "LIVE", DRS_GREEN, self.font_tiny)

        # leader lap count
        leader = max(self.drivers, key=lambda d: (d.laps_completed + d.lap_frac))
        lap_txt = self.font.render(
            f"Lap {min(leader.laps_completed + 1, self.circuit_info['laps'])}/{self.circuit_info['laps']}",
            True, TEXT,
        )
        self.screen.blit(lap_txt, (20, 108))

    def draw_leaderboard(self):
        panel_w = 260
        panel_h = 40 + len(self.drivers) * 24 + 20
        panel = pygame.Rect(WIDTH - panel_w - 16, 16, panel_w, panel_h)
        s = pygame.Surface((panel.w, panel.h), pygame.SRCALPHA)
        pygame.draw.rect(s, (*PANEL_BG, 230), s.get_rect(), border_radius=10)
        self.screen.blit(s, panel.topleft)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1, border_radius=10)

        title = self.font_bold.render("Live Standings", True, TEXT)
        self.screen.blit(title, (panel.x + 12, panel.y + 8))

        ranked = sorted(self.drivers, key=lambda d: (d.dnf, d.current_pos))
        y = panel.y + 40
        for d in ranked:
            row_h = 24
            is_player = d.code == self.player_driver_code
            if is_player:
                row_rect = pygame.Rect(panel.x + 4, y - 2, panel.w - 8, row_h)
                pygame.draw.rect(self.screen, (60, 60, 30), row_rect, border_radius=4)
            label_col = TEXT if not d.dnf else (110, 110, 110)
            pos_str = f"{d.current_pos:>2}" if not d.dnf else "DNF"
            pos_txt = self.font_small.render(pos_str, True, label_col)
            self.screen.blit(pos_txt, (panel.x + 10, y))
            pygame.draw.circle(self.screen, d.color if not d.dnf else (90, 90, 90),
                                (panel.x + 42, y + 7), 5)
            code_txt = self.font_small.render(d.code, True, label_col if not is_player else ACCENT)
            self.screen.blit(code_txt, (panel.x + 55, y))
            gap_str = ""
            if not d.dnf and d is not ranked[0]:
                leader = ranked[0]
                lap_diff = (leader.laps_completed + leader.lap_frac) - (d.laps_completed + d.lap_frac)
                gap_str = f"+{lap_diff:.2f}L"
            gap_txt = self.font_small.render(gap_str, True, DIM_TEXT)
            self.screen.blit(gap_txt, (panel.x + panel.w - gap_txt.get_width() - 10, y))
            y += row_h
            if y > panel.bottom - 20:
                break

    def draw_flag_banner(self):
        if not self.flags:
            return
        f = self.flags[0]
        color = YELLOW_FLAG if f.kind == "yellow" else RED_FLAG if f.kind == "safety_car" else ACCENT
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 150)
        banner = pygame.Rect(WIDTH // 2 - 260, HEIGHT - 60, 520, 44)
        s = pygame.Surface((banner.w, banner.h), pygame.SRCALPHA)
        alpha = int(180 + 60 * pulse)
        pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=8)
        self.screen.blit(s, banner.topleft)
        txt = self.font_bold.render(f.message, True, (10, 10, 10))
        self.screen.blit(txt, (banner.centerx - txt.get_width() // 2, banner.centery - txt.get_height() // 2))

    def draw_tooltip(self, d, mouse_pos):
        lines = [f"#{d.number} {d.name}", d.team, f"Position: P{d.current_pos}"]
        if d.dnf:
            lines.append(f"DNF - {d.dnf_reason}")
        else:
            lines.append(f"Lap {d.laps_completed + 1}/{self.circuit_info['laps']}")
        w = max(self.font_small.size(l)[0] for l in lines) + 20
        h = len(lines) * 18 + 14
        x, y = mouse_pos
        x += 14
        y += 14
        if x + w > WIDTH:
            x -= (w + 28)
        if y + h > HEIGHT:
            y -= (h + 28)
        box = pygame.Rect(x, y, w, h)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, (20, 20, 24, 235), s.get_rect(), border_radius=6)
        self.screen.blit(s, box.topleft)
        pygame.draw.rect(self.screen, ACCENT, box, 1, border_radius=6)
        for i, line in enumerate(lines):
            col = TEXT if i != len(lines) - 1 or not d.dnf else RED_FLAG
            txt = self.font_small.render(line, True, col)
            self.screen.blit(txt, (box.x + 10, box.y + 8 + i * 18))

    # ---------------------------------------------------------------- draw: finish

    def draw_finish(self):
        self.screen.fill(BG)
        title = self.font_huge.render("Chequered Flag", True, TEXT)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 24))
        sub = self.font.render(self.selected_circuit_name, True, DIM_TEXT)
        self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 76))

        y = 120
        for i, d in enumerate(self.finish_order):
            is_player = d.code == self.player_driver_code
            row_rect = pygame.Rect(WIDTH // 2 - 340, y, 680, 30)
            if is_player:
                pygame.draw.rect(self.screen, (60, 60, 30), row_rect, border_radius=6)
            pos_label = f"P{i+1}" if not d.dnf else "DNF"
            pos_col = TEXT if not d.dnf else (150, 90, 90)
            pos_txt = self.font_bold.render(pos_label, True, pos_col)
            self.screen.blit(pos_txt, (row_rect.x + 10, row_rect.y + 3))
            pygame.draw.circle(self.screen, d.color, (row_rect.x + 90, row_rect.y + 15), 8)
            name_txt = self.font.render(f"{d.name}", True, TEXT if not is_player else ACCENT)
            self.screen.blit(name_txt, (row_rect.x + 110, row_rect.y + 5))
            team_txt = self.font_small.render(d.team, True, DIM_TEXT)
            self.screen.blit(team_txt, (row_rect.x + 340, row_rect.y + 8))
            if d.dnf:
                reason_txt = self.font_small.render(d.dnf_reason, True, (150, 90, 90))
                self.screen.blit(reason_txt, (row_rect.x + 480, row_rect.y + 8))
            y += 32
            if y > HEIGHT - 90:
                break

        btn = pygame.Rect(WIDTH // 2 - 110, HEIGHT - 60, 220, 44)
        self.restart_button_rect = btn
        pygame.draw.rect(self.screen, GREEN, btn, border_radius=8)
        txt = self.font_bold.render("Race Again", True, (10, 30, 10))
        self.screen.blit(txt, (btn.centerx - txt.get_width() // 2, btn.centery - txt.get_height() // 2))

    def handle_finish_click(self, pos):
        if self.restart_button_rect and self.restart_button_rect.collidepoint(pos):
            self.player_driver_code = None
            self.selected_circuit_name = None
            self.state = self.STATE_DRIVER_SELECT

    # ---------------------------------------------------------------- main loop

    def run(self):
        last_time = time.time()
        while True:
            now = time.time()
            dt = min(now - last_time, 0.1)
            last_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.state == self.STATE_DRIVER_SELECT:
                        self.handle_driver_select_click(event.pos)
                    elif self.state == self.STATE_CIRCUIT_SELECT:
                        self.handle_circuit_select_click(event.pos)
                    elif self.state == self.STATE_GRID:
                        self.handle_grid_click(event.pos)
                    elif self.state == self.STATE_RACE:
                        self.handle_race_click(event.pos)
                    elif self.state == self.STATE_FINISH:
                        self.handle_finish_click(event.pos)
                if event.type == pygame.MOUSEWHEEL and self.state == self.STATE_CIRCUIT_SELECT:
                    self.handle_circuit_select_scroll(event.y)
                if self.state == self.STATE_RACE:
                    if event.type == pygame.MOUSEWHEEL:
                        self.zoom_at(pygame.mouse.get_pos(), 1.1 ** event.y)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                        self._panning = True
                        self._pan_start_mouse = event.pos
                        self._pan_start_pan = tuple(self.pan)
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                        self._panning = False
                    elif event.type == pygame.MOUSEMOTION and self._panning:
                        dx = event.pos[0] - self._pan_start_mouse[0]
                        dy = event.pos[1] - self._pan_start_mouse[1]
                        self.pan[0] = self._pan_start_pan[0] + dx
                        self.pan[1] = self._pan_start_pan[1] + dy

            if self.state == self.STATE_DRIVER_SELECT:
                self.draw_driver_select()
            elif self.state == self.STATE_CIRCUIT_SELECT:
                self.draw_circuit_select()
            elif self.state == self.STATE_GRID:
                self.draw_grid()
            elif self.state == self.STATE_RACE:
                self.update_race(dt)
                if self.state == self.STATE_RACE:
                    self.draw_race()
                else:
                    self.draw_finish()
            elif self.state == self.STATE_FINISH:
                self.draw_finish()

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    Game().run()
