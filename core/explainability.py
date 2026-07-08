"""
=========================================================
PyC45 Framework
Explainability Engine
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Genera explicaciones del recorrido seguido por el árbol
de decisión durante una predicción.
=========================================================
"""

from __future__ import annotations


class ExplainabilityEngine:

    """
    Motor de explicabilidad.
    """

    # =====================================================
    # Obtener recorrido
    # =====================================================

    def decision_path(self, root, sample):

        current = root

        path = []

        while not current.is_leaf:

            value = sample[current.feature]

            if value <= current.threshold:

                path.append({

                    "feature": current.feature,

                    "operator": "<=",

                    "threshold": current.threshold,

                    "value": value

                })

                current = current.left

            else:

                path.append({

                    "feature": current.feature,

                    "operator": ">",

                    "threshold": current.threshold,

                    "value": value

                })

                current = current.right

        return current, path

    # =====================================================
    # Explicación estructurada
    # =====================================================

    def explain(self, root, sample):

        leaf, path = self.decision_path(
            root,
            sample
        )

        return {

            "prediction": leaf.prediction,

            "rules": path

        }

    # =====================================================
    # Explicación en lenguaje natural
    # =====================================================

    def explain_text(self, root, sample):

        result = self.explain(
            root,
            sample
        )

        text = []

        text.append(
            f"Predicción: {result['prediction']}"
        )

        text.append("")

        text.append("Camino recorrido:")

        for i, rule in enumerate(result["rules"], start=1):

            text.append(

                f"{i}. "
                f"{rule['feature']} "
                f"{rule['operator']} "
                f"{rule['threshold']:.4f} "
                f"(valor={rule['value']})"

            )

        return "\n".join(text)