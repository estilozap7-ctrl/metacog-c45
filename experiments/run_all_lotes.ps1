# Orchestrator for the Official Experimental Campaign
echo "========================================================="
echo "METACOG-C45 OFFICIAL EXPERIMENTAL CAMPAIGN ORCHESTRATOR"
echo "========================================================="

echo "=> INICIANDO LOTE 1 (Datasets 1-10)..."
python experiments/run_batch.py --lote 1
if ($LastExitCode -ne 0) { throw "Error Lote 1" }
echo "=> VERIFICANDO INTEGRIDAD LOTE 1..."
Copy-Item -Path "experiments/results/lote1" -Destination "experiments/results/lote1_backup" -Recurse

echo "=> INICIANDO LOTE 2 (Datasets 11-20)..."
python experiments/run_batch.py --lote 2
if ($LastExitCode -ne 0) { throw "Error Lote 2" }
echo "=> VERIFICANDO INTEGRIDAD LOTE 2..."
Copy-Item -Path "experiments/results/lote2" -Destination "experiments/results/lote2_backup" -Recurse

echo "=> INICIANDO LOTE 3 (Datasets 21-30)..."
python experiments/run_batch.py --lote 3
if ($LastExitCode -ne 0) { throw "Error Lote 3" }
echo "=> VERIFICANDO INTEGRIDAD LOTE 3..."
Copy-Item -Path "experiments/results/lote3" -Destination "experiments/results/lote3_backup" -Recurse

echo "=> CAMPAÑA OFICIAL COMPLETADA CON ÉXITO."
