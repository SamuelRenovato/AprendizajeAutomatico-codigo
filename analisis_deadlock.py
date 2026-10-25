# -*- coding: utf-8 -*-
"""
Reporte de avance - Prediccion del pico diario de jugadores concurrentes de Deadlock (Steam)
Autor: Oscar Samuel Renovato Zapata | Materia: Aprendizaje Automatico

Ejecucion:  python analisis_deadlock.py
Requiere:   pandas, numpy, scikit-learn, matplotlib  (ver requirements.txt)
Salidas:    carpeta resultados/ (figuras .png y tablas .csv)

El archivo usa marcas "# %%" para abrirse como notebook en VS Code / Spyder / Jupytext.
"""
# %% 0. Configuracion
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42                      # no hay componentes aleatorios en esta etapa; se fija para etapas futuras
np.random.seed(SEED)
BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RES = BASE / "resultados"
RES.mkdir(exist_ok=True)

EXPORT_1 = DATOS / "crudo" / "steamdb_export_2026-10-19.csv"   # historial largo, con Average Players
EXPORT_2 = DATOS / "crudo" / "steamdb_export_2026-10-22.csv"   # version limitada; se usa para completar 19-21 oct y verificar
CSV_DIARIO = DATOS / "deadlock_pico_diario_2026-10-21.csv"

FECHA_CORTE = pd.Timestamp("2026-10-21")      # serie congelada (ultimo dia completo)
INICIO_REGIMEN = pd.Timestamp("2024-08-01")   # antes de esta fecha el juego tenia < 2,700 jugadores (otro regimen)
N_PRUEBA, N_VALID = 28, 56                    # 4 y 8 semanas completas
RUTA_RECONSTRUIR = True                       # reconstruye el CSV diario desde los exports crudos y lo compara

# %% 1. Construccion del dataset diario a partir de los exports de SteamDB
def leer_export(ruta):
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", parse_dates=["DateTime"])
    return df.sort_values("DateTime").reset_index(drop=True)

def construir_diario():
    """Unifica frecuencias mixtas (diaria / horaria / 10 min) a frecuencia diaria.
    peak_players = maximo del dia ; avg_players = promedio del dia."""
    e1, e2 = leer_export(EXPORT_1), leer_export(EXPORT_2)
    corte_fino = pd.Timestamp("2026-09-19")          # desde aqui el export 1 deja de ser diario
    diario = e1[e1.DateTime < corte_fino].rename(
        columns={"DateTime": "date", "Players": "peak_players", "Average Players": "avg_players"})[
        ["date", "peak_players", "avg_players"]]
    diario["n_obs"] = 1
    fino = e1[e1.DateTime >= corte_fino].copy()
    # dias 19-21 oct se toman del export 2 (el export 1 termina el 19 con el dia incompleto)
    fino = fino[fino.DateTime < pd.Timestamp("2026-10-19")]
    fino2 = e2[(e2.DateTime >= "2026-10-19") & (e2.DateTime < FECHA_CORTE + pd.Timedelta(days=1))]
    fino = pd.concat([fino[["DateTime", "Players"]], fino2[["DateTime", "Players"]]])
    fino["date"] = fino.DateTime.dt.normalize()
    agg = fino.groupby("date").agg(peak_players=("Players", "max"),
                                   avg_players=("Players", "mean"),
                                   n_obs=("Players", "count")).reset_index()
    out = pd.concat([diario, agg], ignore_index=True)
    out["avg_players"] = out.avg_players.round(0)
    return out[out.date <= FECHA_CORTE].reset_index(drop=True), e2

def verificar_contra_export2(df, e2):
    """Los picos diarios del export 2 deben coincidir con la serie construida (dias traslapados)."""
    d2 = e2[(e2.DateTime >= "2025-10-02") & (e2.DateTime < "2026-10-15")].rename(
        columns={"DateTime": "date", "Players": "peak_export2"})
    c = df.merge(d2, on="date")
    return len(c), int((c.peak_players != c.peak_export2).sum())

if RUTA_RECONSTRUIR and EXPORT_1.exists() and EXPORT_2.exists():
    diario, e2 = construir_diario()
    n_cmp, n_dif = verificar_contra_export2(diario, e2)
    print(f"Verificacion contra export 2: {n_cmp} dias comparados, {n_dif} diferencias")
    previo = pd.read_csv(CSV_DIARIO, parse_dates=["date"])
    print("Coincide con el CSV entregado:", bool((previo.peak_players.values == diario.peak_players.values).all()))
else:
    diario = pd.read_csv(CSV_DIARIO, parse_dates=["date"])

assert diario.date.is_unique and (diario.date.diff().dropna() == pd.Timedelta(days=1)).all(), "serie con huecos"
print(f"Serie diaria: {len(diario)} dias, {diario.date.min().date()} a {diario.date.max().date()}")
serie = diario.set_index("date").peak_players.astype(float)

# %% 2. Particion temporal: entrenamiento / validacion / prueba (reservada)
fin_prueba = serie.index.max()
ini_prueba = fin_prueba - pd.Timedelta(days=N_PRUEBA - 1)
ini_valid = ini_prueba - pd.Timedelta(days=N_VALID)
fin_train = ini_valid - pd.Timedelta(days=1)
print("Entrenamiento hasta", fin_train.date(), "| Validacion", ini_valid.date(), "a",
      (ini_prueba - pd.Timedelta(days=1)).date(), "| Prueba (RESERVADA)", ini_prueba.date(), "a", fin_prueba.date())
EVALUAR_PRUEBA = False   # la prueba no se usa hasta la entrega final

# %% 3. Variables predictoras (solo informacion disponible hasta t-1, salvo el dia de la semana)
def construir_variables(s):
    X = pd.DataFrame({"y": s})
    X["lag_1"] = s.shift(1)
    X["lag_7"] = s.shift(7)
    X["media_7d"] = s.shift(1).rolling(7).mean()      # media de t-1 ... t-7
    dow = pd.get_dummies(s.index.dayofweek, prefix="dow", dtype=float)
    dow.index = s.index
    X = pd.concat([X, dow.drop(columns="dow_0")], axis=1)  # lunes = categoria base
    return X.dropna()

datos = construir_variables(serie)
cols_dow = [c for c in datos.columns if c.startswith("dow_")]
cols_x = ["lag_1", "lag_7", "media_7d"] + cols_dow
train = datos[(datos.index >= INICIO_REGIMEN) & (datos.index <= fin_train)]
train_completo = datos[datos.index <= fin_train]            # incluye el regimen previo (sensibilidad)
valid = datos[(datos.index >= ini_valid) & (datos.index < ini_prueba)]
prueba = datos[datos.index >= ini_prueba]                   # NO se evalua
print(f"Filas modelables -> entrenamiento: {len(train)} (con regimen previo: {len(train_completo)}), "
      f"validacion: {len(valid)}, prueba reservada: {len(prueba)}")

# %% 4. Analisis exploratorio (solo entrenamiento + validacion, salvo la grafica general)
explor = serie[(serie.index >= INICIO_REGIMEN) & (serie.index < ini_prueba)]

# 4.1 Serie completa con particiones
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(serie.index, serie.values, color="#1f77b4", lw=1.1)
ax.set_yscale("log")
ax.axvspan(serie.index.min(), INICIO_REGIMEN, color="#cccccc", alpha=.5, label="Régimen previo (excluido)")
ax.axvspan(INICIO_REGIMEN, fin_train, color="#2ca02c", alpha=.08, label="Entrenamiento")
ax.axvspan(ini_valid, ini_prueba - pd.Timedelta(days=1), color="#ff7f0e", alpha=.18, label="Validación")
ax.axvspan(ini_prueba, fin_prueba, color="#d62728", alpha=.18, label="Prueba (reservada)")
ax.set_ylabel("Pico diario (jugadores, escala log)"); ax.grid(alpha=.3)
ax.legend(loc="lower right", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(RES / "fig1_serie_particiones.png", dpi=170); plt.close(fig)

# 4.2 Perfil semanal (razon pico / media movil centrada de 7 dias) y autocorrelacion de la variacion diaria
razon = (explor / explor.rolling(7, center=True).mean()).dropna()
perfil = razon.groupby(razon.index.dayofweek).agg(["median", "mean", "std", "count"])
perfil.index = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
perfil.to_csv(RES / "perfil_semanal.csv")
dl = np.log(explor).diff().dropna()
def acf(x, nlags):
    x = x - x.mean(); den = (x ** 2).sum()
    return np.array([1.0] + [(x[:-k] * x[k:]).sum() / den for k in range(1, nlags + 1)])
ac = acf(dl.values, 21); lim = 1.96 / np.sqrt(len(dl))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].boxplot([razon[razon.index.dayofweek == k].values for k in range(7)],
               tick_labels=perfil.index, showfliers=False)
axs[0].axhline(1, color="gray", lw=.8, ls="--")
axs[0].set_title("(a) Pico del día / media móvil de 7 días", fontsize=10); axs[0].set_ylabel("Razón"); axs[0].grid(alpha=.3)
axs[1].bar(range(1, 22), ac[1:], color="#1f77b4"); axs[1].axhline(lim, color="r", ls="--", lw=.8); axs[1].axhline(-lim, color="r", ls="--", lw=.8)
axs[1].set_title("(b) Autocorrelación de la variación diaria (log)", fontsize=10); axs[1].set_xlabel("Rezago (días)"); axs[1].grid(alpha=.3)
axs[1].set_xticks([1, 7, 14, 21])
fig.tight_layout(); fig.savefig(RES / "fig2_semana_acf.png", dpi=170); plt.close(fig)
print("\nPerfil semanal (mediana de la razon):\n", perfil["median"].round(3).to_string())
print("ACF rezagos 1, 7, 14:", ac[1].round(3), ac[7].round(3), ac[14].round(3), "| limite 95%:", round(lim, 3))

# 4.3 Saltos de nivel (aumento diario > 35 %) en el periodo exploratorio
v = pd.DataFrame({"anterior": explor.shift(), "pico": explor})
v["cambio_%"] = (v.pico / v.anterior - 1) * 100
saltos = v[v["cambio_%"] > 35].round(1)
saltos.to_csv(RES / "saltos_nivel.csv")
print("\nSaltos de nivel (>35 %):\n", saltos.to_string())
caida = (v["cambio_%"]).min()
print("Mayor caida diaria en el periodo exploratorio: %.1f %%" % caida)
resumen = explor.describe().round(0)
resumen.to_csv(RES / "resumen_objetivo.csv")
print("\nResumen del objetivo (entrenamiento + validacion):\n", resumen.to_string())

# %% 5. Modelos: referencias y primera implementacion (prediccion a 1 dia)
def metricas(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100),
            "R2": r2_score(y, p)}

def ajustar_lineal(tr, usar_log):
    f = (lambda a: np.log(a)) if usar_log else (lambda a: a)
    Xtr = tr[cols_x].copy(); 
    for c in ["lag_1", "lag_7", "media_7d"]: Xtr[c] = f(Xtr[c])
    m = LinearRegression().fit(Xtr, f(tr["y"]))
    def predecir(df):
        X = df[cols_x].copy()
        for c in ["lag_1", "lag_7", "media_7d"]: X[c] = f(X[c])
        p = m.predict(X)
        return np.exp(p) if usar_log else p
    return m, predecir

filas = []
y_v = valid["y"].values
filas.append(("Referencia 1: media del entrenamiento", metricas(y_v, np.full(len(valid), train["y"].mean()))))
filas.append(("Referencia 2: ingenua (valor de ayer)", metricas(y_v, valid["lag_1"].values)))
filas.append(("Referencia 3: ingenua estacional (hace 7 días)", metricas(y_v, valid["lag_7"].values)))
m_raw, pred_raw = ajustar_lineal(train, usar_log=False)
m_log, pred_log = ajustar_lineal(train, usar_log=True)
m_log_full, pred_log_full = ajustar_lineal(train_completo, usar_log=True)
p_raw, p_log, p_log_full = pred_raw(valid), pred_log(valid), pred_log_full(valid)
filas.append(("Regresión lineal (escala original)", metricas(y_v, p_raw)))
filas.append(("Regresión lineal (log del objetivo)", metricas(y_v, p_log)))
filas.append(("Regresión lineal (log), entrenada desde may-2024", metricas(y_v, p_log_full)))
tabla = pd.DataFrame({k: v for k, v in filas}).T.round(3)
tabla.to_csv(RES / "resultados_validacion.csv")
print("\nResultados en VALIDACION (", ini_valid.date(), "a", (ini_prueba - pd.Timedelta(days=1)).date(), "):\n", tabla.to_string())

# metricas en entrenamiento (para detectar sobreajuste)
tr_met = {"Regresión lineal (escala original)": metricas(train["y"].values, pred_raw(train)),
          "Regresión lineal (log del objetivo)": metricas(train["y"].values, pred_log(train))}
print("\nMetricas en entrenamiento:\n", pd.DataFrame(tr_met).T.round(3).to_string())
coef = pd.Series(m_log.coef_, index=cols_x).round(3); print("\nCoeficientes (modelo log):\n", coef.to_string())

# error por dia de la semana y primer dia de validacion (efecto del salto del 30-jul)
err = pd.DataFrame({"y": y_v, "naive": valid["lag_1"].values, "lr_log": p_log}, index=valid.index)
err["ape_naive"] = (err.naive - err.y).abs() / err.y * 100; err["ape_lr"] = (err.lr_log - err.y).abs() / err.y * 100
print("\nPrimeros 3 dias de validacion:\n", err.head(3).round(1).to_string())
sin_primero = err.iloc[1:]
print("\nMAE sin el primer dia (ingenua, LR log):", round((sin_primero.naive - sin_primero.y).abs().mean()),
      round((sin_primero.lr_log - sin_primero.y).abs().mean()))

# 5.1 Grafica de validacion
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(valid.index, y_v, color="black", lw=1.6, label="Real")
ax.plot(valid.index, valid["lag_1"], color="#ff7f0e", lw=1.1, ls="--", label="Ingenua (ayer)")
ax.plot(valid.index, p_log, color="#1f77b4", lw=1.4, label="Regresión lineal (log)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.set_ylabel("Pico diario (jugadores)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
for t in ax.get_xticklabels(): t.set_rotation(20); t.set_ha("right")
fig.tight_layout(); fig.savefig(RES / "fig3_validacion.png", dpi=170); plt.close(fig)

# %% 6. Muestra de la tabla de modelado y resumen de la particion
# muestra tomada del inicio de la validacion (no del periodo reservado)
muestra = valid[["y", "lag_1", "lag_7", "media_7d"]].head(5).round(0).astype(int)
muestra.insert(0, "dia_semana", valid.index[:5].dayofweek)
muestra.to_csv(RES / "muestra_modelado.csv")
print("\nMuestra de la tabla de modelado:\n", muestra.to_string())
info = {"dias_serie": len(serie), "train_filas": len(train), "valid_filas": len(valid), "prueba_filas": len(prueba),
        "fin_train": str(fin_train.date()), "ini_valid": str(ini_valid.date()), "ini_prueba": str(ini_prueba.date())}
pd.Series(info).to_csv(RES / "particion.csv", header=False)
print("\nListo. Resultados en:", RES)
