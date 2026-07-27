# Informe de Análisis de Sensibilidad Independiente — Decision Core
## MetaCog-C45 (v1.1.2) | Fecha: 2026-07-13

Este informe presenta los resultados del estudio experimental de variación independiente de los 7 hiperparámetros del Split Confidence Score (SCS) y Decision Core. El estudio fue realizado sobre 4 datasets de control (iris, breast-w, balance-scale, diabetes) utilizando validación cruzada de 5 folds (semillas reproducibles).

> [!IMPORTANT]
> **Criterio de Aceptación Relativo del Comité Científico:**
> $$0.20 \le \frac{\text{Nodes}_{\text{MetaCog}}}{\text{Nodes}_{\text{Classic}}} \le 0.80$$
> El objetivo es identificar qué combinaciones paramétricas permiten que MetaCog-C45 salga del colapso trivial ($R_{\text{nodes}} \to 0$) sin sobrepoda catastrófica y manteniendo exactitud predictiva.

--- 

## Análisis del Parámetro: `theta_accept`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.3 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.4 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.6 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.3 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.4 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.6 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.3 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.4 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.6 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.3 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.4 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.6 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Análisis del Parámetro: `theta_reject`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.01 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.05 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.1 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.2 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.01 | 60.33 | 9.67 | 0.160 | 0.8627 | 0.9047 | 0.9381 | 0.9571 | 4.00 | 
| breast-w | 0.05 | 60.33 | 3.00 | 0.050 | 0.8627 | 0.2801 | 0.9381 | 0.7476 | 2.00 | 
| breast-w | 0.1 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.2 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.01 | 109.67 | 6.33 | 0.058 | 0.4445 | 0.3494 | 0.7489 | 0.7229 | 3.67 | 
| diabetes | 0.05 | 109.67 | 3.00 | 0.027 | 0.4445 | 0.3494 | 0.7489 | 0.7229 | 2.00 | 
| diabetes | 0.1 | 109.67 | 2.33 | 0.021 | 0.4445 | 0.2062 | 0.7489 | 0.6883 | 1.67 | 
| diabetes | 0.2 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.01 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.05 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.1 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.2 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Análisis del Parámetro: `alpha_0`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Análisis del Parámetro: `beta_0`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Análisis del Parámetro: `gamma`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.2 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.8 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.2 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.8 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.2 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.8 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.2 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.8 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Análisis del Parámetro: `lambda_1`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.3 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.3 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.3 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.3 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Análisis del Parámetro: `lambda_2`

Valores evaluados mientras los demás se mantienen en la línea base:
- Línea base: {'theta_accept': 0.5, 'theta_reject': 0.2, 'alpha_0': 1.0, 'beta_0': 1.0, 'gamma': 0.5, 'lambda_1': 1.0, 'lambda_2': 1.0}

| Dataset | Valor | Classic Nodes | MetaCog Nodes | Node Ratio ($R_{\text{nodes}}$) | Classic MCC | MetaCog MCC | Classic Acc | MetaCog Acc | MetaCog Depth | 
|---|---|---|---|---|---|---|---|---|---| 
| balance-scale | 0.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 0.3 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.0 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| balance-scale | 1.5 | 159.00 | 1.00 | 0.006 | 0.6162 | 0.0000 | 0.7672 | 0.4603 | 1.00 | 
| breast-w | 0.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 0.3 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.0 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| breast-w | 1.5 | 60.33 | 1.00 | 0.017 | 0.8627 | 0.0000 | 0.9381 | 0.6571 | 1.00 | 
| diabetes | 0.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 0.3 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.0 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| diabetes | 1.5 | 109.67 | 1.67 | 0.015 | 0.4445 | 0.0943 | 0.7489 | 0.6667 | 1.33 | 
| iris | 0.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 0.3 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.0 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 
| iris | 1.5 | 14.33 | 1.00 | 0.070 | 0.9687 | 0.0000 | 0.9778 | 0.3333 | 1.00 | 

--- 

## Conclusiones del Estudio de Sensibilidad

### 1. Parámetros de Umbral (`theta_accept` y `theta_reject`)
- **theta_reject**: Es el filtro de colapso a hoja. Si se sitúa en 0.20, prácticamente el 97.6% de los nodos colapsan inmediatamente debido a que el SCS rara vez supera este umbral en nodos internos. Reducir `theta_reject` a rangos de [0.01, 0.05] debería rehabilitar el crecimiento del árbol.
- **theta_accept**: Controla el paso directo a ACCEPT sin revisión o defer. Ajustar este parámetro permite suavizar la transición y permitir divisiones de menor ganancia.

### 2. Elasticidades de Estabilidad (`alpha_0` y `beta_0`)
- Exponentes altos aplastan de forma exponencial la utilidad SCS del nodo. Si NS o TS son menores a 1.0 (lo cual es normal por el ruido), un `alpha_0=1.0` o `beta_0=1.0` amplificado por profundidad resulta en una sobrepenalización severa. Valores de [0.2, 0.5] son más tolerantes.

### 3. Elasticidad de Competencia (`gamma`)
- El término de competencia $(1 - CI)^{\gamma}$ tiene un impacto masivo si `gamma=0.5`. Dado que la competitividad media observada en la sonda es 0.9799, el factor $(1 - CI) \approx 0.02$. Elevarlo a la potencia 0.5 da $\approx 0.14$. Multiplicar por 0.14 destruye cualquier SCS. Reducir `gamma` a 0.1 o 0.2, o incluso 0.0 (desactivar penalización por colinealidad) alivia la poda.

### 4. Coeficientes de Amplificación por Profundidad y Escasez (`lambda_1` y `lambda_2`)
- `lambda_1=1.0` y `lambda_2=1.0` provocan que la penalización crezca muy rápido. Valores cercanos a 0.1 o 0.3 amortiguan esta amplificación, protegiendo las ramas medias del árbol contra el colapso.
