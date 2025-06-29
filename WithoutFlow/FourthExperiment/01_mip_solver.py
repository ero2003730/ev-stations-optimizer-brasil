#!/usr/bin/env python3
# 01_mip_solver_by_state_zero_start.py — MIP por UF, começando sem estações prévias

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
import geobr  # pip install geobr

# ─────────── ENCONTRA CACHE DINAMICAMENTE ───────────
HERE = Path(__file__).resolve().parent  # pasta do script: .../WithoutFlow/FourthExperiment
CUR = HERE
while CUR != CUR.parent:
    if (CUR / "cache").is_dir():
        CACHE = CUR / "cache"
        break
    CUR = CUR.parent
else:
    raise FileNotFoundError(f"Não achei nenhuma pasta 'cache' acima de {HERE}")

# ─────────── DEFINE PATHS ───────────
BASE       = CACHE / "parquet"
GRAPHML    = CACHE / "graph_Brazil.graphml"

# cria pasta de saída específica para este experimento
OUTPUT_DIR = BASE / "WithoutFlow" / HERE.name  # .../cache/parquet/WithoutFlow/FourthExperiment
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


R_KM, R_EARTH   = 150, 6371.0088
R_RAD           = R_KM / R_EARTH
N_HWY, KW_HWY   = 10, 120
N_CITY, KW_CITY = 5, 22
kWh_por_carro   = 20_000  # kWh/dia por carro

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

    # 1) carrega grafo e demanda
    G   = ox.load_graphml(GRAPHML)
    dem = pd.read_parquet(BASE/"demand_by_node.parquet")   # node, kWh_dia

    # mapeia lon/lat
    lon = {n:d["x"] for n,d in G.nodes(data=True)}
    lat = {n:d["y"] for n,d in G.nodes(data=True)}
    dem["lon"] = dem["node"].map(lon)
    dem["lat"] = dem["node"].map(lat)

    # 2) atribui UF a cada nó via geobr + spatial join
    ufs = geobr.read_state(year=2020)[["abbrev_state","geometry"]]
    ufs = ufs.rename(columns={"abbrev_state":"UF"})
    gdf_dem = gpd.GeoDataFrame(
        dem, geometry=gpd.points_from_xy(dem.lon, dem.lat), crs="EPSG:4326"
    )
    gdf_dem = gpd.sjoin(gdf_dem, ufs, how="left", predicate="within")
    dem["UF"] = gdf_dem["UF"].fillna("UNK").values

    # 3) TODOS os nós são candidatos (zero start)
    uncov = dem.reset_index(drop=True)
    n_cand = len(uncov)
    print(f"• nós candidatos (zero start): {n_cand}  (100% do total)")

    # 4) vizinhanças entre candidatos
    coords    = np.deg2rad(uncov[["lat","lon"]].values)
    neighbors = BallTree(coords, metric="haversine") \
                   .query_radius(coords, r=R_RAD)

    # 5) monta modelo MIP
    prob = pulp.LpProblem("EV_by_state_zero", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")
    a = {}
    print("• criando variáveis a_{i,j} …")
    for i in range(n_cand):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    # objetivo: minimizar número de estações
    prob += pulp.lpSum(x[i] for i in range(n_cand)), "Minimize_num_stations"

    # 5.1) atender demanda de cada nó j
    print("• adicionando restrições de demanda …")
    for j in range(n_cand):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            == uncov.at[j,"kWh_dia"]
        ), f"demand_{j}"

    # 5.2) capacidade de cada candidato i
    print("• adicionando restrições de capacidade …")
    for i in range(n_cand):
        node = uncov.at[i,"node"]
        deg  = G.degree[node]
        Qi   = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"cap_{i}"

    # 5.3) cobertura mínima por UF
    print("• adicionando cobertura mínima por UF …")
    for uf in sorted(uncov["UF"].dropna().unique()):
        idxs_uf = [j for j,u in enumerate(uncov["UF"]) if u==uf]
        Duf     = demanda_veiculos.get(uf,0) * kWh_por_carro
        prob += (
            pulp.lpSum(a[(i,j)]
                       for j in idxs_uf
                       for i in neighbors[j])
            >= Duf
        ), f"cov_{uf}"

    # 6) resolve
    print("• iniciando solver …")
    res = prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=600))
    print("Status:", pulp.LpStatus[res])

    # 7) extrai e salva solução
    chosen = [i for i in range(n_cand) if x[i].value() > 0.5]
    out = uncov.loc[chosen, ["node","lat","lon","UF"]].copy()
    out["tipo"] = out["node"].map(
        lambda n: "CIDADE" if G.degree[n]>4 else "RODOVIA"
    )
    out.to_parquet(OUTPUT_DIR / "sites_new_mip_experiment10.parquet", index=False)
    print(f"✓ Salvou {len(chosen)} estações em 'sites_new_mip_experiment10.parquet'")

    t_all = time.perf_counter() - t0_all
    print(f"Tempo total: {t_all:.1f} s")

if __name__=="__main__":
    main()