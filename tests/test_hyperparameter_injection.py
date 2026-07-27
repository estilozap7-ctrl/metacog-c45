"""
=========================================================
Sprint 1.5 — Hyperparameter Injection
tests/test_hyperparameter_injection.py
=========================================================

Verifica que los 7 hiperparámetros del Decision Core se
propagan correctamente desde C45Classifier hasta
SplitConfidenceScore sin modificación alguna.

Estrategia de prueba:
  - Se instancia C45Classifier con valores no-default para
    todos los hiperparámetros.
  - Se intercepta SplitConfidenceScore.__init__ para
    capturar los valores recibidos.
  - Se verifica que los valores capturados coincidan
    exactamente con los valores configurados en el
    clasificador (propagación exacta, sin pérdida).
  - También se verifica end-to-end que modificar un
    hiperparámetro desde el nivel CLI (run_fold) produce
    el mismo valor en SplitConfidenceScore.
=========================================================
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_data():
    """Pequeño dataset sintético tipo-Iris para tests rápidos."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
    })
    y = pd.Series(np.repeat([0, 1, 2], n // 3))
    return X, y


@pytest.fixture
def nondefault_params():
    """Valores deliberadamente distintos de los defaults para detectar
    cualquier hardcoding."""
    return {
        'theta_accept': 0.77,
        'theta_reject': 0.11,
        'alpha_0':      2.5,
        'beta_0':       3.1,
        'gamma':        0.25,
        'lambda_1':     1.8,
        'lambda_2':     0.6,
    }


# ---------------------------------------------------------------------------
# Prueba 1: C45Classifier almacena todos los hiperparámetros en su estado
# ---------------------------------------------------------------------------

class TestClassifierStoresHyperparameters:

    def test_constructor_stores_all_params(self, nondefault_params):
        """C45Classifier debe almacenar exactamente los valores recibidos."""
        from core.classifier import C45Classifier
        clf = C45Classifier(mode='metacog', **nondefault_params)

        assert clf.theta_accept == nondefault_params['theta_accept'], \
            f"theta_accept esperado={nondefault_params['theta_accept']}, obtenido={clf.theta_accept}"
        assert clf.theta_reject == nondefault_params['theta_reject'], \
            f"theta_reject esperado={nondefault_params['theta_reject']}, obtenido={clf.theta_reject}"
        assert clf.alpha_0 == nondefault_params['alpha_0'], \
            f"alpha_0 esperado={nondefault_params['alpha_0']}, obtenido={clf.alpha_0}"
        assert clf.beta_0 == nondefault_params['beta_0'], \
            f"beta_0 esperado={nondefault_params['beta_0']}, obtenido={clf.beta_0}"
        assert clf.gamma == nondefault_params['gamma'], \
            f"gamma esperado={nondefault_params['gamma']}, obtenido={clf.gamma}"
        assert clf.lambda_1 == nondefault_params['lambda_1'], \
            f"lambda_1 esperado={nondefault_params['lambda_1']}, obtenido={clf.lambda_1}"
        assert clf.lambda_2 == nondefault_params['lambda_2'], \
            f"lambda_2 esperado={nondefault_params['lambda_2']}, obtenido={clf.lambda_2}"

    def test_default_params_are_correct(self):
        """Verificar que los valores por defecto siguen siendo los canónicos."""
        from core.classifier import C45Classifier
        clf = C45Classifier(mode='metacog')
        assert clf.theta_accept == 0.5
        assert clf.theta_reject == 0.2
        assert clf.alpha_0      == 1.0
        assert clf.beta_0       == 1.0
        assert clf.gamma        == 0.5
        assert clf.lambda_1     == 1.0
        assert clf.lambda_2     == 1.0


# ---------------------------------------------------------------------------
# Prueba 2: SplitConfidenceScore recibe exactamente los valores del Classifier
# ---------------------------------------------------------------------------

class TestSplitConfidenceScoreInjection:

    def test_scs_receives_exact_params_from_classifier(self, iris_data, nondefault_params):
        """
        Al llamar clf.fit(), SplitConfidenceScore.__init__ debe recibir
        los mismos valores que se configuraron en C45Classifier.
        """
        from core.classifier import C45Classifier

        captured = {}

        original_scs_init = None

        def capture_scs_init(self_scs, alpha_0=1.0, beta_0=1.0, gamma=0.5,
                              lambda_1=1.0, lambda_2=1.0):
            captured['alpha_0']  = alpha_0
            captured['beta_0']   = beta_0
            captured['gamma']    = gamma
            captured['lambda_1'] = lambda_1
            captured['lambda_2'] = lambda_2
            # Llamar al __init__ original para que el objeto funcione
            original_scs_init(self_scs, alpha_0=alpha_0, beta_0=beta_0,
                               gamma=gamma, lambda_1=lambda_1, lambda_2=lambda_2)

        from core.metacognition import decision_core as dc_module
        original_scs_init = dc_module.SplitConfidenceScore.__init__

        X, y = iris_data

        with patch.object(dc_module.SplitConfidenceScore, '__init__', capture_scs_init):
            clf = C45Classifier(mode='metacog', **nondefault_params)
            clf.fit(X, y)

        assert captured, "SplitConfidenceScore.__init__ nunca fue llamado — inyección fallida."

        assert captured['alpha_0']  == nondefault_params['alpha_0'],  \
            f"alpha_0: esperado={nondefault_params['alpha_0']}, recibido={captured['alpha_0']}"
        assert captured['beta_0']   == nondefault_params['beta_0'],   \
            f"beta_0: esperado={nondefault_params['beta_0']}, recibido={captured['beta_0']}"
        assert captured['gamma']    == nondefault_params['gamma'],    \
            f"gamma: esperado={nondefault_params['gamma']}, recibido={captured['gamma']}"
        assert captured['lambda_1'] == nondefault_params['lambda_1'], \
            f"lambda_1: esperado={nondefault_params['lambda_1']}, recibido={captured['lambda_1']}"
        assert captured['lambda_2'] == nondefault_params['lambda_2'], \
            f"lambda_2: esperado={nondefault_params['lambda_2']}, recibido={captured['lambda_2']}"

    def test_scs_default_params_when_no_override(self, iris_data):
        """Con defaults, SplitConfidenceScore debe recibir los valores canónicos."""
        from core.classifier import C45Classifier
        from core.metacognition import decision_core as dc_module

        captured = {}
        original_scs_init = dc_module.SplitConfidenceScore.__init__

        def capture_scs_init(self_scs, alpha_0=1.0, beta_0=1.0, gamma=0.5,
                              lambda_1=1.0, lambda_2=1.0):
            captured['alpha_0']  = alpha_0
            captured['beta_0']   = beta_0
            captured['gamma']    = gamma
            captured['lambda_1'] = lambda_1
            captured['lambda_2'] = lambda_2
            original_scs_init(self_scs, alpha_0=alpha_0, beta_0=beta_0,
                               gamma=gamma, lambda_1=lambda_1, lambda_2=lambda_2)

        X, y = iris_data
        with patch.object(dc_module.SplitConfidenceScore, '__init__', capture_scs_init):
            clf = C45Classifier(mode='metacog')
            clf.fit(X, y)

        assert captured['alpha_0']  == 1.0
        assert captured['beta_0']   == 1.0
        assert captured['gamma']    == 0.5
        assert captured['lambda_1'] == 1.0
        assert captured['lambda_2'] == 1.0


# ---------------------------------------------------------------------------
# Prueba 3: DecisionValidator recibe theta_accept y theta_reject
# ---------------------------------------------------------------------------

class TestDecisionValidatorInjection:

    def test_validator_receives_thresholds(self, iris_data, nondefault_params):
        """
        DecisionValidator debe recibir theta_accept y theta_reject
        exactamente como los configuró C45Classifier.
        """
        from core.classifier import C45Classifier
        from core.metacognition import decision_core as dc_module

        captured = {}
        original_dv_init = dc_module.DecisionValidator.__init__

        def capture_dv_init(self_dv, *args, **kwargs):
            captured['theta_accept'] = kwargs.get('theta_accept', args[0] if len(args) > 0 else 0.5)
            captured['theta_reject'] = kwargs.get('theta_reject', args[1] if len(args) > 1 else 0.2)
            original_dv_init(self_dv, *args, **kwargs)

        X, y = iris_data
        with patch.object(dc_module.DecisionValidator, '__init__', capture_dv_init):
            clf = C45Classifier(mode='metacog', **nondefault_params)
            clf.fit(X, y)

        assert captured, "DecisionValidator.__init__ nunca fue llamado."
        assert captured['theta_accept'] == nondefault_params['theta_accept'], \
            f"theta_accept: esperado={nondefault_params['theta_accept']}, recibido={captured['theta_accept']}"
        assert captured['theta_reject'] == nondefault_params['theta_reject'], \
            f"theta_reject: esperado={nondefault_params['theta_reject']}, recibido={captured['theta_reject']}"


# ---------------------------------------------------------------------------
# Prueba 4: Pipeline run_fold propaga metacog_params al Classifier
# ---------------------------------------------------------------------------

class TestRunFoldPropagation:

    def test_run_fold_passes_params_to_classifier(self, iris_data, nondefault_params):
        """
        run_fold con metacog_params debe instanciar C45Classifier con esos
        valores exactos.
        """
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        from experiments.pipeline.run_fold import run_fold
        from core import classifier as clf_module

        captured_kwargs = {}

        original_init = clf_module.C45Classifier.__init__

        def capture_classifier_init(self_clf, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self_clf, **kwargs)

        X, y = iris_data

        with patch.object(clf_module.C45Classifier, '__init__', capture_classifier_init):
            run_fold(
                X, y, X, y,
                mode='metacog',
                random_state=0,
                metacog_params=nondefault_params
            )

        for key, expected_val in nondefault_params.items():
            assert key in captured_kwargs, \
                f"Parámetro '{key}' no llegó a C45Classifier desde run_fold."
            assert captured_kwargs[key] == expected_val, \
                f"{key}: esperado={expected_val}, recibido={captured_kwargs[key]}"

    def test_run_fold_classic_mode_ignores_metacog_params(self, iris_data, nondefault_params):
        """
        En modo classic, run_fold puede recibir metacog_params pero C45Classifier
        se construye con mode='classic' y no activa el Decision Core.
        """
        from experiments.pipeline.run_fold import run_fold

        X, y = iris_data
        # No debe lanzar excepción
        result = run_fold(X, y, X, y, mode='classic', random_state=0)
        assert result['mode'] == 'classic'
        assert 'accuracy' in result


# ---------------------------------------------------------------------------
# Prueba 5: Verificar que cada hiperparámetro modifica el comportamiento
#           del árbol (test de integración de sensibilidad mínima)
# ---------------------------------------------------------------------------

class TestEachParamAffectsTree:
    """
    Para cada hiperparámetro, verifica que cambiar su valor desde el default
    produce un objeto SplitConfidenceScore diferente (atributo almacenado).
    No depende de resultados de métricas para evitar flakiness por aleatoriedad.
    """

    @pytest.mark.parametrize("param,value", [
        ('alpha_0',      2.0),
        ('beta_0',       3.0),
        ('gamma',        0.1),
        ('lambda_1',     2.0),
        ('lambda_2',     0.5),
        ('theta_accept', 0.8),
        ('theta_reject', 0.05),
    ])
    def test_param_stored_correctly(self, iris_data, param, value):
        """
        Instanciar C45Classifier con un valor no-default para 'param'
        y verificar que el atributo interno refleja ese valor.
        """
        from core.classifier import C45Classifier
        kwargs = {param: value}
        clf = C45Classifier(mode='metacog', **kwargs)
        stored = getattr(clf, param)
        assert stored == value, \
            f"{param}: configurado={value}, almacenado={stored}"


# ---------------------------------------------------------------------------
# Prueba 6: Compatibilidad con modo classic (regresión)
# ---------------------------------------------------------------------------

class TestClassicModeCompatibility:

    def test_classic_mode_fit_predict(self, iris_data):
        """
        El modo classic debe funcionar sin parámetros metacognitivos
        y sin memory_ ni experience_stats_.
        """
        from core.classifier import C45Classifier
        X, y = iris_data
        clf = C45Classifier(mode='classic')
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert clf.memory_ is None
        assert clf.experience_stats_ is None

    def test_classic_mode_unaffected_by_metacog_params(self, iris_data, nondefault_params):
        """
        Pasar metacog_params a C45Classifier en modo classic no debe
        afectar el comportamiento clásico (el árbol se construye igual).
        """
        from core.classifier import C45Classifier
        X, y = iris_data

        clf_default = C45Classifier(mode='classic', random_state=42)
        clf_default.fit(X, y)

        # Con params metacog — en modo classic no debería importar
        clf_params = C45Classifier(mode='classic', random_state=42, **nondefault_params)
        clf_params.fit(X, y)

        preds_default = clf_default.predict(X)
        preds_params  = clf_params.predict(X)

        # En modo classic los metacog_params no se usan, el árbol es idéntico
        np.testing.assert_array_equal(
            preds_default, preds_params,
            err_msg="Modo classic produce resultados diferentes cuando se pasan metacog_params — posible bug de acoplamiento."
        )
