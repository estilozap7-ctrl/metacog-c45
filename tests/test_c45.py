"""
=========================================================
MetaCog-C45 Tests
Unit Testing Suite
=========================================================

Verifica la exactitud matemática y lógica de todo el framework MetaCog-C45.
=========================================================
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.decision_node import DecisionNode
from core.entropy import EntropyCalculator
from core.information_gain import InformationGainCalculator
from core.gain_ratio import GainRatioCalculator
from core.best_split import BestSplitFinder
from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator


class TestEntropyCalculator(unittest.TestCase):

    def test_entropy_pure(self):
        y = np.array([1, 1, 1, 1])
        self.assertEqual(EntropyCalculator.calculate(y), 0.0)
        self.assertTrue(EntropyCalculator.is_pure(y))

    def test_entropy_even(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(EntropyCalculator.calculate(y), 1.0)
        self.assertFalse(EntropyCalculator.is_pure(y))

    def test_majority_class(self):
        y = np.array([0, 1, 1, 1, 0])
        self.assertEqual(EntropyCalculator.majority_class(y), 1)
        
        y2 = np.array([0, 0, 0, 1])
        self.assertEqual(EntropyCalculator.majority_class(y2), 0)

    def test_validate_empty(self):
        with self.assertRaises(ValueError):
            EntropyCalculator.calculate(np.array([]))


class TestInformationGain(unittest.TestCase):

    def test_information_gain(self):
        # Padre con 4 elementos (dos 0, dos 1) -> Entropía = 1.0
        y_parent = np.array([0, 0, 1, 1])
        # División perfecta
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        
        gain = InformationGainCalculator.information_gain(y_parent, y_left, y_right)
        self.assertEqual(gain, 1.0)
        
        split_info = InformationGainCalculator.split_information(y_left, y_right)
        self.assertEqual(split_info, 1.0)


class TestGainRatio(unittest.TestCase):

    def test_gain_ratio(self):
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        
        ratio = GainRatioCalculator.calculate(y_parent, y_left, y_right)
        self.assertEqual(ratio, 1.0)


class TestDecisionNode(unittest.TestCase):

    def test_node_creation(self):
        node = DecisionNode(feature="age", threshold=30.0, is_leaf=False)
        self.assertEqual(node.feature, "age")
        self.assertEqual(node.threshold, 30.0)
        self.assertFalse(node.is_leaf)
        self.assertFalse(node.has_children())

        left_child = DecisionNode(prediction=1, is_leaf=True)
        node.left = left_child
        self.assertTrue(node.has_children())
        self.assertTrue(node.left.is_leaf)


class TestC45Classifier(unittest.TestCase):

    def setUp(self):
        # Datos de prueba simples
        data = {
            'x1': [1, 2, 3, 4, 5, 6, 7, 8],
            'x2': [10, 20, 10, 20, 10, 20, 10, 20],
            'y':  [0, 0, 0, 0, 1, 1, 1, 1]
        }
        self.df = pd.DataFrame(data)
        self.X = self.df[['x1', 'x2']]
        self.y = self.df['y']

    def test_fit_and_predict(self):
        clf = C45Classifier(max_depth=3, min_samples_split=2)
        clf.fit(self.X, self.y)
        
        # Verificar que el árbol tiene nodos
        self.assertIsNotNone(clf.root)
        
        # Hacer predicciones
        preds = clf.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        
        # Exactitud en entrenamiento debería ser 1.0 ya que es linealmente separable por x1
        acc = clf.score(self.X, self.y)
        self.assertEqual(acc, 1.0)

    def test_predict_proba(self):
        clf = C45Classifier(max_depth=3, min_samples_split=2)
        clf.fit(self.X, self.y)
        
        probas = clf.predict_proba(self.X)
        self.assertEqual(probas.shape, (len(self.y), 2))
        # La suma de probabilidades para cada muestra debe ser 1.0
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)


class TestMetrics(unittest.TestCase):

    def test_confusion_matrix(self):
        y_true = [1, 0, 1, 1, 0, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
        
        cm = ConfusionMatrix.from_predictions(y_true, y_pred)
        # Reales: 0 (4), 1 (4)
        # TP (1,1): predichos 1 que son 1 -> muestras index 0, 3, 5 -> 3
        # TN (0,0): predichos 0 que son 0 -> muestras index 1, 4, 7 -> 3
        # FP (0,1): predichos 1 que son 0 -> muestra index 6 -> 1
        # FN (1,0): predichos 0 que son 1 -> muestra index 2 -> 1
        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.tn, 3)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.total, 8)

    def test_classification_metrics(self):
        cm = ConfusionMatrix()
        cm.tp = 3
        cm.tn = 3
        cm.fp = 1
        cm.fn = 1
        
        metrics = ClassificationMetrics(cm)
        self.assertEqual(metrics.accuracy, 0.75)
        self.assertEqual(metrics.precision, 0.75)
        self.assertEqual(metrics.recall, 0.75)
        self.assertEqual(metrics.specificity, 0.75)
        self.assertEqual(metrics.f1_score, 0.75)
        self.assertEqual(metrics.error_rate, 0.25)


class TestROCAUC(unittest.TestCase):

    def test_auc_perfect(self):
        y_true = [0, 0, 1, 1]
        y_probs = [0.1, 0.2, 0.8, 0.9]
        
        roc_data = ROCAUCCalculator.calculate(y_true, y_probs)
        self.assertEqual(roc_data["auc"], 1.0)

    def test_auc_random(self):
        y_true = [0, 0, 1, 1]
        y_probs = [0.5, 0.5, 0.5, 0.5]
        
        roc_data = ROCAUCCalculator.calculate(y_true, y_probs)
        self.assertEqual(roc_data["auc"], 0.5)


class TestPruning(unittest.TestCase):

    def test_pruning_effect(self):
        # Datos de entrenamiento
        X_train = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6, 7, 8]})
        y_train = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
        
        clf = C45Classifier(max_depth=5, min_samples_split=2)
        clf.fit(X_train, y_train)
        
        # Datos de validación que justifican la poda (el corte no ayuda en validación)
        # Si en validación el comportamiento es uniforme o contrario, REP colapsará el árbol a una hoja
        X_val = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6, 7, 8]})
        y_val = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])  # Toda la validación es de clase 1
        
        # Antes de podar el árbol tiene un nodo divisor
        self.assertFalse(clf.root.is_leaf)
        
        # Aplicar poda
        DecisionTreePruner.prune(clf, X_val, y_val)
        
        # Después de la poda, el árbol debe haber colapsado a una sola hoja
        # ya que predecir la clase mayoritaria (1) da exactitud 1.0 en validación,
        # mientras que el árbol dividido daría menor exactitud local.
        self.assertTrue(clf.root.is_leaf)
        self.assertEqual(clf.root.prediction, 1)


if __name__ == '__main__':
    unittest.main()
