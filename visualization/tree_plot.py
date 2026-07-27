"""
=========================================================
MetaCog-C45 Framework
Tree Plot
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Visualiza el árbol de decisión generado por MetaCog-C45.
=========================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class TreePlotter:

    def __init__(self):

        self.fig = None
        self.ax = None

    # =====================================================
    # Dibujar nodo
    # =====================================================

    def _draw_node(self, x, y, text, color="#D6EAF8"):

        self.ax.text(

            x,
            y,
            text,

            ha="center",

            va="center",

            fontsize=8,

            bbox=dict(

                boxstyle="round",

                facecolor=color,

                edgecolor="black"

            )

        )

    # =====================================================
    # Dibujar conexiones
    # =====================================================

    def _draw_edge(self, x1, y1, x2, y2):

        self.ax.plot(

            [x1, x2],

            [y1, y2],

            color="black"

        )

    # =====================================================
    # Recorrido recursivo
    # =====================================================

    def _plot_tree(

        self,

        node,

        x,

        y,

        dx

    ):

        if node is None:

            return

        # ------------------------

        if node.is_leaf:

            text = (

                f"Leaf\n"

                f"Clase={node.prediction}\n"

                f"n={node.samples}"

            )

            self._draw_node(

                x,

                y,

                text,

                "#ABEBC6"

            )

            return

        # ------------------------

        text = (

            f"{node.feature}\n"

            f"≤ {node.threshold:.2f}\n"

            f"GR={node.gain_ratio:.3f}"

        )

        self._draw_node(

            x,

            y,

            text

        )

        # ------------------------

        if node.left:

            self._draw_edge(

                x,

                y,

                x-dx,

                y-1

            )

            self._plot_tree(

                node.left,

                x-dx,

                y-1,

                dx/2

            )

        # ------------------------

        if node.right:

            self._draw_edge(

                x,

                y,

                x+dx,

                y-1

            )

            self._plot_tree(

                node.right,

                x+dx,

                y-1,

                dx/2

            )

    # =====================================================
    # Mostrar árbol
    # =====================================================

    def plot(

        self,

        root,

        figsize=(16,10)

    ):

        self.fig, self.ax = plt.subplots(

            figsize=figsize

        )

        self.ax.axis("off")

        self._plot_tree(

            root,

            0,

            0,

            8

        )

        plt.show()