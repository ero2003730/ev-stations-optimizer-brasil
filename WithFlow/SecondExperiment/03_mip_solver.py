#!/usr/bin/env python3
# solve_mip_with_flow_noprev.py
# MIP bicritério: min estações + max VMD coberto (opcional)
# SEM pontos de carregamento prévios

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
from tqdm import tqdm
import multiprocessing

# ─── encontra cache/parquet ───────────────────────────
HERE = Path(__file__).resolve().parent
CUR  = HERE
while CUR != CUR.parent:
    if (CUR/"cache"/"parquet").is_dir():
        BASE = CUR/"cache"/"parquet"
        break
    CUR = CUR.parent
else:
    raise FileNotFoundError(f"Não achei cache/parquet acima de {HERE}")

GRAPHML     = CUR/"cache"/"graph_Brazil.graphml"
OUTPUT_DIR  = BASE/"WithFlow"/"SecondExperiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── parâmetros gerais ────────────────────────────────
R_KM, EARTH_R = 300, 6371.0088
R_RAD         = R_KM / EARTH_R
N_HWY, KW_HWY = 10, 120
N_CITY, KW_CITY = 5, 22
LAMBDA       = 0.9   # peso para #estações vs. fluxo

def main():
    t0_all = time.perf_counter()

    # 1) carrega grafo, demanda e fluxo
    G    = ox.load_graphml(str(GRAPHML))
    dem  = pd.read_parquet(BASE/"demand_by_node.parquet")    # node, kWh_dia
    flux = pd.read_parquet(BASE/"flux_all.parquet")          # lat,lon,vmd,...

    # 2) mapeia lon/lat dos nós de demanda
    lon = {n: d["x"] for n,d in G.nodes(data=True)}
    lat = {n: d["y"] for n,d in G.nodes(data=True)}
    dem["lon"] = dem["node"].map(lon)
    dem["lat"] = dem["node"].map(lat)

    # 3) atribui VMD do ponto de fluxo mais próximo
    tree_flux = BallTree(np.deg2rad(flux[["lat","lon"]]), metric="haversine")
    _, idx = tree_flux.query(np.deg2rad(dem[["lat","lon"]]), k=1)
    dem["vmd"] = flux["vmd"].values[idx.flatten()]

    # 4) todos os nós são candidatos (sem filtro de cobertura prévia)
    uncov = dem.reset_index(drop=True)
    n = len(uncov)
    print(f"• candidatos: {n} nós (sem pontos prévios)")

    # 5) vizinhanças dentro de R_KM
    coords    = np.deg2rad(uncov[["lat","lon"]].values)
    neighbors = BallTree(coords, metric="haversine").query_radius(coords, r=R_RAD)

    # 6) monta MIP
    prob = pulp.LpProblem("EV_with_flow_noprev", pulp.LpMinimize)

    # x[i]=1 se abre estação em i
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    # y[j]=1 se decidimos atender o nó j (para fluxo coberto)
    y = pulp.LpVariable.dicts("y", range(n), cat="Binary")
    # a[i,j] = kWh alocados de i para j
    a = {}
    for i in range(n):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    # 7) objetivo bicritério
    term_sites = pulp.lpSum(x[i] for i in range(n))
    term_flow  = pulp.lpSum(uncov.at[j,"vmd"] * y[j] for j in range(n))
    prob += LAMBDA * term_sites - (1 - LAMBDA) * term_flow, "bicriteria"

    # 8) restrições

    # 8.1) atendimento opcional (pode não atender alguns nós)
    for j in range(n):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            >= uncov.at[j,"kWh_dia"] * y[j]
        ), f"demand_opt_{j}"

    # 8.2) link alocação ↔ estação
    for i in range(n):
        for j in neighbors[i]:
            prob += a[(i,j)] <= uncov.at[j,"kWh_dia"] * x[i], f"link_{i}_{j}"

    # 8.3) capacidade de cada estação
    for i in range(n):
        node = uncov.at[i,"node"]
        deg  = G.degree[node]
        Qi   = (N_CITY * KW_CITY * 24) if deg > 4 else (N_HWY * KW_HWY * 24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"cap_{i}"

    # 9) resolve com CBC (10 min, gap 1%)
    n_threads = multiprocessing.cpu_count()
    solver = pulp.PULP_CBC_CMD(
        msg=True,
        threads=max(1, n_threads-2),
        gapRel=0.01
    )
    res = prob.solve(solver)
    print("Status:", pulp.LpStatus[res])

    # 10) extrai solução e salva
    chosen = [i for i in range(n) if x[i].value() > 0.5]
    out = uncov.loc[chosen, ["node","lat","lon"]].copy()
    out["tipo"] = out["node"].apply(
        lambda nn: "CIDADE" if G.degree[nn] > 4 else "RODOVIA"
    )
    out.to_parquet(
        OUTPUT_DIR/"sites_new_mip_withflow6.parquet",
        index=False
    )
    print(f"✓ Salvou {len(chosen)} estações em {OUTPUT_DIR}")
    print(f"⏱️ Tempo total: {time.perf_counter() - t0_all:.1f} s")

if __name__=="__main__":
    main()