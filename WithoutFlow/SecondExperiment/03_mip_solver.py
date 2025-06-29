#!/usr/bin/env python3
# solve_mip_zero_start.py — modelo MIP sem estações pré-existentes

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
from tqdm import tqdm

# ─────────────────────────── PARÂMETROS ────────────────────────────
R_KM, EARTH_R = 300, 6371.0088
R_RAD         = R_KM / EARTH_R          # km → rad
N_HWY, KW_HWY = 10, 120                 # rodovia
N_CITY, KW_CITY = 5, 22                 # cidade

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

def main():
    t0_all = time.perf_counter()

    # 1) Carrega apenas a demanda
    dem = pd.read_parquet(BASE / "demand_by_node.parquet")  # colunas: node, kWh_dia
    # Mapeia lat/lon a partir do grafo
    lon = {n: d["x"] for n, d in G.nodes(data=True)}
    lat = {n: d["y"] for n, d in G.nodes(data=True)}
    dem["lon"] = dem["node"].map(lon)
    dem["lat"] = dem["node"].map(lat)

    # 2) Como não há estações existentes, TODOS os nós estão descobertos
    dem_uncovered = dem.copy().reset_index(drop=True)
    n_cand = len(dem_uncovered)
    print(f"• nós descobertos (todo o conjunto): {n_cand}  (100 % do total)")

    # 3) Pré-cálculo de vizinhanças dentro do raio R
    coords    = np.deg2rad(dem_uncovered[["lat","lon"]].values)
    tree      = BallTree(coords, metric="haversine")
    neighbors = tree.query_radius(coords, r=R_RAD)

    # 4) Montagem do modelo MIP
    prob = pulp.LpProblem("EV_Charging_Stations_ZeroStart", pulp.LpMinimize)

    # variáveis x_i ∈ {0,1} indicando se instalamos estação em i
    x = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")

    # variáveis a_{i,j} contínuas para alocação de demand
    a = {}
    print("• criando variáveis de alocação …")
    for i in tqdm(range(n_cand), desc="vars a_{i,j}"):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    # Função-objetivo: minimizar número de estações
    prob += pulp.lpSum(x[i] for i in range(n_cand)), "Minimize_num_stations"

    # Restrição de atendimento de demanda para cada nó j
    print("• adicionando restrições de demanda …")
    for j in tqdm(range(n_cand), desc="Demand"):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            == dem_uncovered.at[j, "kWh_dia"]
        ), f"Demand_{j}"

    # Restrição de capacidade de cada estação candidata i
    print("• adicionando restrições de capacidade …")
    for i in tqdm(range(n_cand), desc="Capacity"):
        node_id = dem_uncovered.at[i, "node"]
        deg     = G.degree[node_id]
        # Q_i em kWh/dia
        Qi = (N_CITY * KW_CITY * 24) if deg > 4 else (N_HWY * KW_HWY * 24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"Cap_{i}"

    # 5) Resolvemeto
    print("• resolvendo o modelo …")
    solver = pulp.PULP_CBC_CMD(msg=True)
    t0 = time.perf_counter()
    status = prob.solve(solver)
    t_solve = time.perf_counter() - t0

    # 6) Resultados
    chosen = [i for i in range(n_cand) if x[i].value() > 0.5]
    num_new = len(chosen)
    print("\n────────── RESUMO ──────────")
    print("Status       :", pulp.LpStatus[status])
    print("Estações = X :", num_new)
    print(f"Tempo solver : {t_solve:.1f} s")
    print(f"Tempo total  : {time.perf_counter()-t0_all:.1f} s")

    # 7) Monta DataFrame de saída
    df_new = pd.DataFrame({
        "node": dem_uncovered.loc[chosen, "node"],
        "lat" : dem_uncovered.loc[chosen, "lat"],
        "lon" : dem_uncovered.loc[chosen, "lon"],
        "tipo": [
            "CIDADE" if G.degree[n] > 4 else "RODOVIA"
            for n in dem_uncovered.loc[chosen, "node"]
        ]
    })
    df_new.to_parquet(OUTPUT_DIR / "sites_new_mip_experiment6.parquet", index=False)
    print(f"✓ sites_new_mip_experiment6.parquet salvo — {num_new} estações")

if __name__ == "__main__":
    main()