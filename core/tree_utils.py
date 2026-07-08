"""
=========================================================
PyC45 Framework
Tree Utilities
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Funciones auxiliares para manipular árboles C4.5.
=========================================================
"""

from __future__ import annotations

from .decision_node import DecisionNode


class TreeUtils:

    # =====================================================
    # Contar nodos
    # =====================================================

    @staticmethod
    def count_nodes(node):

        if node is None:
            return 0

        return (
            1
            + TreeUtils.count_nodes(node.left)
            + TreeUtils.count_nodes(node.right)
        )

    # =====================================================
    # Contar hojas
    # =====================================================

    @staticmethod
    def count_leaves(node):

        if node is None:
            return 0

        if node.is_leaf:
            return 1

        return (
            TreeUtils.count_leaves(node.left)
            + TreeUtils.count_leaves(node.right)
        )

    # =====================================================
    # Profundidad
    # =====================================================

    @staticmethod
    def depth(node):

        if node is None:
            return 0

        return max(
            TreeUtils.depth(node.left),
            TreeUtils.depth(node.right)
        ) + 1

    # =====================================================
    # Imprimir árbol
    # =====================================================

    @staticmethod
    def print_tree(node, indent=""):

        if node is None:
            return

        if node.is_leaf:

            print(
                indent
                + f"Leaf -> Clase={node.prediction}"
                + f" ({node.samples})"
            )

            return

        print(
            indent
            + f"{node.feature} <= {node.threshold:.4f}"
            + f"  [Gain={node.gain_ratio:.4f}]"
        )

        TreeUtils.print_tree(
            node.left,
            indent + "│   "
        )

        TreeUtils.print_tree(
            node.right,
            indent + "│   "
        )

    # =====================================================
    # Número de niveles
    # =====================================================

    @staticmethod
    def levels(node):

        return TreeUtils.depth(node)

    # =====================================================
    # Árbol vacío
    # =====================================================

    @staticmethod
    def is_empty(node):

        return node is None