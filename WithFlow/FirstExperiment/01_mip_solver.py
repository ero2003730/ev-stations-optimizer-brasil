#!/usr/bin/env python3
# solve_mip_with_flow_optional.py
# MIP bicritério: min estações + max VMD coberto (opcional)

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
OUTPUT_DIR  = BASE/"WithFlow"/"FirstExperiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── parâmetros gerais ────────────────────────────────
R_KM, EARTH_R = 150, 6371.0088
R_RAD         = R_KM / EARTH_R
N_HWY, KW_HWY = 10, 120
N_CITY, KW_CITY = 5, 22
LAMBDA       = 0.9   # peso para #estações vs. fluxo

def main():
    t0_all = time.perf_counter()

    # 1) carrega grafo, demanda, oferta e fluxo
    G    = ox.load_graphml(str(GRAPHML))
    dem  = pd.read_parquet(BASE/"demand_by_node.parquet")    # node, kWh_dia
    sup  = pd.read_parquet(BASE/"supply_existing.parquet")   # node
    flux = pd.read_parquet(BASE/"flux_all.parquet")          # lat,lon,vmd,...

    # 2) mapeia lon/lat
    lon = {n: d["x"] for n,d in G.nodes(data=True)}
    lat = {n: d["y"] for n,d in G.nodes(data=True)}
    dem["lon"] = dem["node"].map(lon)
    dem["lat"] = dem["node"].map(lat)

    # 3) atribui VMD do ponto de fluxo mais próximo
    tree_flux = BallTree(np.deg2rad(flux[["lat","lon"]]), metric="haversine")
    _, idx = tree_flux.query(np.deg2rad(dem[["lat","lon"]]), k=1)
    dem["vmd"] = flux["vmd"].values[idx.flatten()]

    # 4) remove nós já cobertos por supply existente
    sup["lon"] = sup["node"].map(lon)
    sup["lat"] = sup["node"].map(lat)
    tree_sup = BallTree(np.deg2rad(sup[["lat","lon"]]), metric="haversine")
    covered = tree_sup.query_radius(np.deg2rad(dem[["lat","lon"]]), r=R_RAD)
    dem["covered"] = False
    for inds in covered:
        dem.loc[inds,"covered"] = True
    uncov = dem[~dem["covered"]].reset_index(drop=True)

    n = len(uncov)
    print(f"• candidatos: {n} nós não cobertos")

    # 5) vizinhanças dentro de R_KM
    coords    = np.deg2rad(uncov[["lat","lon"]].values)
    neighbors = BallTree(coords,metric="haversine")\
                  .query_radius(coords, r=R_RAD)

    # 6) monta MIP
    prob = pulp.LpProblem("EV_with_flow_optional", pulp.LpMinimize)

    # x[i]=1 se abre estação em i
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    # y[j]=1 se decidimos atender o nó j (conta para fluxo coberto)
    y = pulp.LpVariable.dicts("y", range(n), cat="Binary")
    # a[i,j] = kWh alocados de i para j
    a = {}
    for i in range(n):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    # objetivo bicritério
    term_sites = pulp.lpSum(x[i] for i in range(n))
    term_flow  = pulp.lpSum(uncov.at[j,"vmd"] * y[j] for j in range(n))
    prob += LAMBDA*term_sites - (1-LAMBDA)*term_flow

    # 7) restrições

    # 7.1) atendimento opcional
    for j in range(n):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            >= uncov.at[j,"kWh_dia"] * y[j]
        ), f"demand_opt_{j}"

    # 7.2) link a-x: só aloca se existe estação
    for i in range(n):
        for j in neighbors[i]:
            prob += a[(i,j)] <= uncov.at[j,"kWh_dia"] * x[i], f"link_{i}_{j}"

    # 7.3) capacidade das estações
    for i in range(n):
        node = uncov.at[i,"node"]
        deg  = G.degree[node]
        Qi   = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"cap_{i}"

    # 8) resolve com CBC (10 min, gap 1%)
    n_threads = multiprocessing.cpu_count()  # 14 no seu caso
    solver = pulp.PULP_CBC_CMD(
        msg=True,
        threads=n_threads - 2, # ou n_threads//2 se quiser economizar CPU/RAM
        gapRel=0.01        # 1% de gap relativo aceitável
    )
    res = prob.solve(solver)
    print("Status:", pulp.LpStatus[res])

    # 9) extrai solução
    chosen = [i for i in range(n) if x[i].value()>0.5]
    out = uncov.loc[chosen, ["node","lat","lon"]].copy()
    out["tipo"] = out["node"].apply(
        lambda nn: "CIDADE" if G.degree[nn]>4 else "RODOVIA"
    )
    out.to_parquet(OUTPUT_DIR/"sites_new_mip_withflow1.parquet", index=False)
    print(f"✓ Salvou {len(chosen)} estações em {OUTPUT_DIR}")

    print(f"⏱️ Tempo total: {time.perf_counter()-t0_all:.1f} s")


if __name__=="__main__":
    main()