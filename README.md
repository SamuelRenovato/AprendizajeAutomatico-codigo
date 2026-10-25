# Deadlock: pico diario de jugadores concurrentes (reporte de avance)

Alumno: Oscar Samuel Renovato Zapata · Materia: Aprendizaje Automático

## Contenido
- `analisis_deadlock.py` — análisis completo (construcción de la serie, exploración, partición, modelos). Tiene celdas `# %%` para abrirlo como notebook.
- `datos/crudo/` — exports originales de SteamDB (App ID 1422450) descargados el 19 y el 22 de octubre de 2026.
- `datos/deadlock_pico_diario_2026-10-21.csv` — serie diaria de trabajo (880 días, 25-may-2024 a 21-oct-2026).
- `resultados/` — se genera al ejecutar (figuras y tablas del reporte).

## Ejecución
```
pip install -r requirements.txt
python analisis_deadlock.py
```
Python 3.12. El script reconstruye la serie diaria desde los exports, verifica que coincide con el CSV de trabajo y con el segundo export (378 días, 0 diferencias), y escribe sus resultados en `resultados/`.

## Notas
- Semilla: 42 (esta etapa no usa aleatoriedad).
- Partición cronológica: entrenamiento hasta 29-jul-2026, validación 30-jul a 23-sep-2026, prueba 24-sep a 21-oct-2026 (reservada; `EVALUAR_PRUEBA = False`).
- Fuente de los datos: SteamDB, https://steamdb.info/app/1422450/charts/
