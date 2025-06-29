#!/usr/bin/env python3
# solve_mip_progress.py  ── modelo MIP com barras de progresso

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
from tqdm import tqdm   # pip install tqdm

# ─────────────────────────── PARÂMETROS ────────────────────────────
R_KM       = 150
R_RAD      = R_KM / 6371.0088          # km → rad
N_HWY, KW_HWY   = 10, 120
N_CITY, KW_CITY =  5,  22


# ─────────── ENCONTRA CACHE DINAMICAMENTE ───────────
HERE = Path(__file__).resolve().parent
# sobemos até achar uma pasta 'cache' (ou chegar na raiz)
CUR = HERE
while CUR != CUR.parent:
    if (CUR / "cache").is_dir():
        CACHE = CUR / "cache"
        break
    CUR = CUR.parent
else:
    raise FileNotFoundError("Não achei nenhuma pasta 'cache' acima de " + str(HERE))

BASE    = CACHE / "parquet"
GRAPHML = CACHE / "graph_Brazil.graphml"
# cria pasta de saída para esse experimento
OUTPUT_DIR = BASE / "WithoutFlow" / Path(__file__).parent.name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# carrega o grafo
G = ox.load_graphml(str(GRAPHML))

# ─────────────────────────── FUNÇÃO PRINCIPAL ──────────────────────
def main():
    t0_all = time.perf_counter()

    # 1) ------------- CARREGA DADOS --------------------------------
    dem = pd.read_parquet(BASE / "demand_by_node.parquet")   # node, kWh_dia
    sup = pd.read_parquet(BASE / "supply_existing.parquet")  # node …

    lon = {n:d["x"] for n,d in G.nodes(data=True)}
    lat = {n:d["y"] for n,d in G.nodes(data=True)}
    dem["lon"], dem["lat"] = dem["node"].map(lon), dem["node"].map(lat)
    sup["lon"], sup["lat"] = sup["node"].map(lon), sup["node"].map(lat)

    # 2) ------------- COBERTURA EXISTENTE --------------------------
    tree_dem = BallTree(np.deg2rad(dem[["lat","lon"]]), metric="haversine")
    idxs = tree_dem.query_radius(np.deg2rad(sup[["lat","lon"]]), r=R_RAD)
    dem["covered_exist"] = False
    for inds in idxs:
        dem.loc[inds, "covered_exist"] = True

    uncov = dem.loc[~dem.covered_exist].reset_index(drop=True)
    n_cand = len(uncov)
    print(f"• nós descobertos: {n_cand}  ({n_cand/len(dem):.1%} do total)")

    # 3) ------------- VIZINHANÇAS ----------------------------------
    coords = np.deg2rad(uncov[["lat","lon"]].values)
    tree   = BallTree(coords, metric="haversine")
    neighbors = tree.query_radius(coords, r=R_RAD)

    # 4) ------------- MODELO PULP ----------------------------------
    prob = pulp.LpProblem("EV_Charging_Stations", pulp.LpMinimize)

    # x_i  (binário)
    x = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")

    # a_{i,j}  (contínuo)  —— progress-bar na criação
    a = {}
    print("• criando variáveis de alocação …")
    for i in tqdm(range(n_cand)):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    # Objetivo
    prob += pulp.lpSum(x[i] for i in range(n_cand)), "Minimize_sites"

    # Restrições —— com progress-bar
    print("• adicionando restrições …")
    for j in tqdm(range(n_cand), desc="- Demanda"):
        prob += (pulp.lpSum(a[(i,j)] for i in neighbors[j])
                 == uncov.at[j,"kWh_dia"])

    for i in tqdm(range(n_cand), desc="- Capacidade"):
        node_id = uncov.at[i,"node"]
        deg     = G.degree[node_id]
        Qi = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (pulp.lpSum(a[(i,j)] for j in neighbors[i]) <= Qi * x[i])

    # 5) ------------- RESOLVER -------------------------------------
    t0_solve = time.perf_counter()
    solver   = pulp.PULP_CBC_CMD(msg=True)   # log completo do CBC
    res      = prob.solve(solver)
    t_solve  = time.perf_counter() - t0_solve

    # 6) ------------- RESULTADOS -----------------------------------
    new_sites = [i for i in range(n_cand) if x[i].value() > 0.5]
    print("\n────────── RESUMO ──────────")
    print("status  :", pulp.LpStatus[res])
    print("novos   :", len(new_sites))
    print(f"tempo   : {t_solve:.1f} s (solver) | {time.perf_counter()-t0_all:.1f} s total")

    # salva parquet
    out = pd.DataFrame({
        "node":  uncov.loc[new_sites,"node"],
        "lat":   uncov.loc[new_sites,"lat"],
        "lon":   uncov.loc[new_sites,"lon"],
        "tipo": ["CIDADE" if G.degree[n]>4 else "RODOVIA"
                 for n in uncov.loc[new_sites,"node"]]
    })
    out.to_parquet(OUTPUT_DIR / "sites_new_mip_experiment1.parquet", index=False)
    print("✓ sites_new_mip_experiment1.parquet salvo")

# ─────────────────────────── EXECUÇÃO ─────────────────────────────
if __name__ == "__main__":
    main()