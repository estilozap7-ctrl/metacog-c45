"""
=========================================================
MetaCog-C45 Framework
Tree Pruning (Poda de Árboles)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Implementa la poda de árboles utilizando Poda por Error Reducido (REP).
=========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class DecisionTreePruner:
    """
    Poda por Error Reducido (Reduced Error Pruning - REP).
    """

    @staticmethod
    def prune(model, X_val, y_val):
        """
        Poda el árbol de decisión del modelo clasificador utilizando un conjunto de validación.

        Parámetros:
        -----------
        model : C45Classifier
            El modelo clasificador entrenado cuya raíz se quiere podar.
        X_val : DataFrame o array-like
            Variables predictoras de validación.
        y_val : Series o array-like
            Etiquetas de clase de validación.
        """
        if model.root is None:
            return

        # Asegurar formato pandas
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)

        model.root = DecisionTreePruner._prune_node(model.root, X_val, y_val)

    @staticmethod
    def _prune_node(node, X_val, y_val):
        if node is None or node.is_leaf:
            return node

        # Si el conjunto de validación está vacío, no podemos evaluar la decisión
        if len(y_val) == 0:
            return node

        # Filtrar datos de validación que llegan a cada hijo
        mask_left = X_val[node.feature] <= node.threshold
        X_val_left = X_val[mask_left]
        y_val_left = y_val[mask_left]

        X_val_right = X_val[~mask_left]
        y_val_right = y_val[~mask_left]

        # Podar recursivamente los subárboles primero (enfoque bottom-up)
        node.left = DecisionTreePruner._prune_node(node.left, X_val_left, y_val_left)
        node.right = DecisionTreePruner._prune_node(node.right, X_val_right, y_val_right)

        # Si ambos hijos se convirtieron en hojas (o ya lo eran), evaluar si podemos podar este nodo
        if node.left and node.left.is_leaf and node.right and node.right.is_leaf:
            # Calcular exactitud (accuracy) local antes de podar (con los hijos activos)
            preds_before = []
            for _, sample in X_val.iterrows():
                # Recorrer subárbol a partir de este nodo
                curr = node
                while not curr.is_leaf:
                    if sample[curr.feature] <= curr.threshold:
                        curr = curr.left
                    else:
                        curr = curr.right
                preds_before.append(curr.prediction)
            
            acc_before = np.mean(np.array(preds_before) == y_val)

            # Calcular exactitud local después de podar (colapsar el nodo a hoja)
            # El nodo ya tiene precalculada la clase mayoritaria en 'prediction'
            acc_after = np.mean(node.prediction == y_val)

            # Si colapsar el nodo no reduce la exactitud, se poda
            if acc_after >= acc_before:
                node.is_leaf = True
                node.left = None
                node.right = None

        return node
