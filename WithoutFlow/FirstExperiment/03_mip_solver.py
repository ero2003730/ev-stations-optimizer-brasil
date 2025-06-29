#!/usr/bin/env python3
# 04_mip_solver_gap.py  –– encerra ao atingir 0,5 % de gap ou 15 min
# CHANGE --> R_KM TO 300km

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
from tqdm import tqdm

# -------------------- parâmetros globais ---------------------------
R_KM, EARTH_R = 300, 6371.0088
R_RAD         = R_KM / EARTH_R             # km → rad
N_HWY, KW_HWY = 10, 120                    # rodovia
N_CITY, KW_CITY = 5, 22                    # cidade

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
    t0 = time.perf_counter()

    # 1) Carrega dados
    dem = pd.read_parquet(BASE / "demand_by_node.parquet")
    sup = pd.read_parquet(BASE / "supply_existing.parquet")

    lon = {n: d["x"] for n, d in G.nodes(data=True)}
    lat = {n: d["y"] for n, d in G.nodes(data=True)}
    dem["lon"], dem["lat"] = dem["node"].map(lon), dem["node"].map(lat)
    sup["lon"], sup["lat"] = sup["node"].map(lon), sup["node"].map(lat)

    # 2) Marca cobertura existente
    tree_dem    = BallTree(np.deg2rad(dem[["lat","lon"]].values), metric="haversine")
    covered_idx = tree_dem.query_radius(np.deg2rad(sup[["lat","lon"]].values), r=R_RAD)
    dem["covered_exist"] = False
    for idx in covered_idx:
        dem.loc[idx, "covered_exist"] = True

    uncov   = dem[~dem.covered_exist].reset_index(drop=True)
    n_cand  = len(uncov)
    print(f"• nós descobertos: {n_cand}  ({n_cand/len(dem):.1%} do total)")

    # 3) Prepara vizinhanças
    coords    = np.deg2rad(uncov[["lat","lon"]].values)
    tree      = BallTree(coords, metric="haversine")
    neighbors = tree.query_radius(coords, r=R_RAD)

    # 4) Monta modelo
    prob = pulp.LpProblem("EV_Charging_Stations", pulp.LpMinimize)
    x    = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")

    a = {}
    print("• criando variáveis de alocação …")
    for i in tqdm(range(n_cand)):
        for j in neighbors[i]:
            a[(i,j)] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0)

    prob += pulp.lpSum(x[i] for i in range(n_cand)), "Minimize_sites"

    print("• adicionando restrições …")
    for j in tqdm(range(n_cand), desc="- Demanda"):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            == uncov.at[j, "kWh_dia"]
        )
    for i in tqdm(range(n_cand), desc="- Capacidade"):
        node_id = uncov.at[i, "node"]
        deg     = G.degree[node_id]
        Qi      = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        )

    # 5) Resolve com critérios de parada
    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=900,                   # 15 min
        options=[
            "ratioGap=0.005",            # gap ≤ 0,5 %
            "maxSecondsSinceCutoff=120"  # sem melhora em 120 s
        ],
    )
    t_solve0 = time.perf_counter()
    status   = prob.solve(solver)
    t_solve  = time.perf_counter() - t_solve0

    # 6) Imprime resultados
    obj_val       = pulp.value(prob.objective)
    new_sites_idx = [i for i in range(n_cand) if x[i].value() > 0.5]

    print("\n────────── RESUMO ──────────")
    print("status     :", pulp.LpStatus[status])
    print(f"objective  : {obj_val:.3f}")
    print(f"tempo      : {t_solve:.1f} s (solver) | {(time.perf_counter()-t0):.1f} s total")
    print("novos      :", len(new_sites_idx))

    # 7) Salva saída
    out = pd.DataFrame({
        "node": uncov.loc[new_sites_idx, "node"],
        "lat" : uncov.loc[new_sites_idx, "lat"],
        "lon" : uncov.loc[new_sites_idx, "lon"],
        "tipo": [
            "CIDADE" if G.degree[n]>4 else "RODOVIA"
            for n in uncov.loc[new_sites_idx, "node"]
        ],
    })
    out.to_parquet(OUTPUT_DIR / "sites_new_mip_experiment3.parquet", index=False)
    print("✓ sites_new_mip_experiment3.parquet salvo")

if __name__ == "__main__":
    main()