#!/usr/bin/env python3
# 01_mip_solver_by_state.py — MIP com cobertura mínima por UF (via geobr)

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
import geobr  # pip install geobr

R_KM, R_EARTH  = 300, 6371.0088
R_RAD          = R_KM / R_EARTH
N_HWY, KW_HWY  = 10, 120
N_CITY, KW_CITY= 5, 22
kWh_por_carro  = 20_000  # kWh/dia por carro


# ─────────── ENCONTRA CACHE DINAMICAMENTE ───────────
HERE = Path(__file__).resolve().parent  # .../WithoutFlow/ThirdExperiment
CUR = HERE
while CUR != CUR.parent:
    if (CUR / "cache").is_dir():
        CACHE = CUR / "cache"
        break
    CUR = CUR.parent
else:
    raise FileNotFoundError(f"Não achei nenhuma pasta 'cache' acima de {HERE}")

# ─────────── DEFINE PATHS E PARÂMETROS ───────────
BASE        = CACHE / "parquet"
GRAPHML     = CACHE / "graph_Brazil.graphml"

# ─────────── PREPARA OUTPUT_DIR ───────────
OUTPUT_DIR = BASE / "WithoutFlow" / HERE.name  # .../cache/parquet/WithoutFlow/ThirdExperiment
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# demanda de veículos por UF (número de carros)
demanda_veiculos = {
    "SP":28350, "PR":7817, "SC":6666, "RS":7002,
    "RJ":8065,  "ES":2791, "BA":3962, "SE":669,
    "AL":1518,  "PE":4368, "PB":1038, "RN":2249,
    "CE":2854,  "PI":981,  "TO":378,  "MA":1604,
    "PA":977,   "AP":163,  "RR":218,  "AM":1640,
    "MT":1215,  "GO":2359, "DF":11115,"MS":851,
    "MG":5625,  "RO":667,  "AC":157
}

def main():
    t0_all = time.perf_counter()

    # ─── 1) carrega grafo e dados ──────────────────────────────
    G   = ox.load_graphml(GRAPHML)
    dem = pd.read_parquet(BASE/"demand_by_node.parquet")   # node, kWh_dia
    sup = pd.read_parquet(BASE/"supply_existing.parquet")  # node

    # mapeia lon/lat
    lon = {n:d["x"] for n,d in G.nodes(data=True)}
    lat = {n:d["y"] for n,d in G.nodes(data=True)}
    for df in (dem, sup):
        df["lon"] = df["node"].map(lon)
        df["lat"] = df["node"].map(lat)

    # ─── 2) busca limites de estados ───────────────────────────
    ufs = geobr.read_state(year=2020)[["abbrev_state","geometry"]]
    ufs = ufs.rename(columns={"abbrev_state":"UF"})

    # ─── 3) spatial join para atribuir UF a cada nó ───────────
    for df in (dem, sup):
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.lon, df.lat),
            crs="EPSG:4326"
        )
        joined = gpd.sjoin(gdf, ufs, how="left", predicate="within")
        df["UF"] = joined["UF"].fillna("UNK").values

    # ─── 4) descobre quais nós ainda NÃO estão cobertos ────────
    tree_sup = BallTree(np.deg2rad(sup[["lat","lon"]]), metric="haversine")
    idxs = tree_sup.query_radius(np.deg2rad(dem[["lat","lon"]]), r=R_RAD)
    dem["covered"] = False
    for inds in idxs:
        dem.loc[inds, "covered"] = True
    uncov = dem[~dem["covered"]].reset_index(drop=True)

    print(f"• nós descobertos (sem cobertura existing): {len(uncov)}")

    # ─── 5) vizinhanças entre candidatos ───────────────────────
    coords = np.deg2rad(uncov[["lat","lon"]].values)
    neighbors = BallTree(coords, metric="haversine")\
                   .query_radius(coords, r=R_RAD)

    # ─── 6) monta modelo MIP ──────────────────────────────────
    prob = pulp.LpProblem("EV_by_state", pulp.LpMinimize)
    n_cand = len(uncov)
    x = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")
    a = {}
    for i in range(n_cand):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    # objetivo: mínimo número de estações
    prob += pulp.lpSum(x[i] for i in range(n_cand))

    # todo j deve ter sua demanda atendida
    for j in range(n_cand):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            == uncov.at[j,"kWh_dia"]
        ), f"demand_{j}"

    # capacidade de cada i
    for i in range(n_cand):
        node = uncov.at[i,"node"]
        deg  = G.degree[node]
        Qi   = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"cap_{i}"

    # cobertura mínima por UF
    for uf in sorted(uncov["UF"].dropna().unique()):
        idxs_uf = [j for j,u in enumerate(uncov["UF"]) if u==uf]
        Duf     = demanda_veiculos.get(uf,0) * kWh_por_carro
        prob += (
            pulp.lpSum(a[(i,j)]
                       for j in idxs_uf
                       for i in neighbors[j])
            >= Duf
        ), f"cov_{uf}"

    # ─── 7) resolve ─────────────────────────────────────────────
    print("• iniciando solver…")
    res = prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=600))
    print("Status:", pulp.LpStatus[res])

    # ─── 8) extrai solução ────────────────────────────────────
    chosen = [i for i in range(n_cand) if x[i].value()>0.5]
    out = uncov.loc[chosen, ["node","lat","lon","UF"]].copy()
    out["tipo"] = out["node"].map(lambda n: "CIDADE" if G.degree[n]>4 else "RODOVIA")
    out.to_parquet(OUTPUT_DIR/"sites_new_mip_experiment9.parquet", index=False)

    t_all = time.perf_counter() - t0_all
    print(f"✓ Gerou {len(chosen)} novas estações em {t_all:.1f}s")

if __name__=="__main__":
    main()