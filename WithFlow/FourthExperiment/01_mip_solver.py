#!/usr/bin/env python3
# 01_mip_solver_by_state_with_flow_from_scratch.py
# MIP bicritério: min estações + max VMD coberto,
# com cobertura mínima de fluxo por UF — começando do zero

from pathlib import Path
import time
import multiprocessing

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import osmnx as ox
import pulp
import geobr  # pip install geobr

# ─── parâmetros gerais ────────────────────────────────
R_KM, R_EARTH   = 150, 6371.0088
R_RAD           = R_KM / R_EARTH
N_HWY, KW_HWY   = 10, 120
N_CITY, KW_CITY = 5,  22
kWh_por_carro   = 20_000   # consumo diário médio por veículo
LAMBDA          = 0.9      # peso entre #estações (λ) e fluxo (1−λ)

demanda_veiculos = {
    "SP":28350, "PR":7817, "SC":6666, "RS":7002,
    "RJ":8065,  "ES":2791, "BA":3962, "SE":669,
    "AL":1518,  "PE":4368, "PB":1038, "RN":2249,
    "CE":2854,  "PI":981,  "TO":378,  "MA":1604,
    "PA":977,   "AP":163,  "RR":218,  "AM":1640,
    "MT":1215,  "GO":2359, "DF":11115,"MS":851,
    "MG":5625,  "RO":667,  "AC":157
}

# ─── localiza cache/parquet ───────────────────────────
HERE = Path(__file__).resolve().parent
CUR  = HERE
while CUR != CUR.parent:
    if (CUR/"cache"/"parquet").is_dir():
        BASE = CUR/"cache"/"parquet"
        break
    CUR = CUR.parent
else:
    raise FileNotFoundError(f"Não achei cache/parquet acima de {HERE}")

# grafo está em cache/graph_Brazil.graphml (nível acima de parquet)
GRAPHML    = BASE.parent/"graph_Brazil.graphml"
if not GRAPHML.exists():
    raise FileNotFoundError(f"Grafo não encontrado em {GRAPHML}")
    
OUTPUT_DIR = BASE/"WithFlow"/HERE.name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0_all = time.perf_counter()

    # 1) carrega grafo e dados (ignora oferta existente)
    G    = ox.load_graphml(str(GRAPHML))
    dem  = pd.read_parquet(BASE/"demand_by_node.parquet")
    flux = pd.read_parquet(BASE/"flux_all.parquet")

    # 2) mapeia lon/lat em dem
    lon = {n: d["x"] for n,d in G.nodes(data=True)}
    lat = {n: d["y"] for n,d in G.nodes(data=True)}
    dem["lon"] = dem["node"].map(lon)
    dem["lat"] = dem["node"].map(lat)

    # 3) associa VMD ao nó de demanda mais próximo
    tree_flux = BallTree(np.deg2rad(flux[["lat","lon"]]), metric="haversine")
    _, idx = tree_flux.query(np.deg2rad(dem[["lat","lon"]]), k=1)
    dem["vmd"] = flux["vmd"].values[idx.flatten()]

    # 4) carrega malha de UFs e faz spatial-join para etiquetar cada nó
    ufs = geobr.read_state(year=2020)[["abbrev_state","geometry"]]
    ufs = ufs.rename(columns={"abbrev_state":"UF"})
    # se já tiver CRS, só override; se não, define
    if ufs.crs:
        ufs = ufs.set_crs("EPSG:4326", allow_override=True)
    else:
        ufs = ufs.set_crs("EPSG:4326")
    gdem = gpd.GeoDataFrame(
        dem,
        geometry=gpd.points_from_xy(dem.lon, dem.lat),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(gdem, ufs, how="left", predicate="within")
    dem["UF"] = joined["UF"].fillna("UNK").values

    # 5) todos os nós são candidatos (parte do zero)
    uncov = dem.reset_index(drop=True)
    n_cand = len(uncov)
    print(f"• candidatos totais: {n_cand} nós")

    # 6) vizinhanças dentro de R_KM
    coords    = np.deg2rad(uncov[["lat","lon"]].values)
    neighbors = BallTree(coords, metric="haversine") \
                    .query_radius(coords, r=R_RAD)

    # 7) monta modelo MIP bicritério
    prob = pulp.LpProblem("EV_by_state_with_flow_scratch", pulp.LpMinimize)

    # variáveis de decisão
    x = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")  # instala em i?
    y = pulp.LpVariable.dicts("y", range(n_cand), cat="Binary")  # atende j?
    a = { (i,j): pulp.LpVariable(f"a_{i}_{j}", lowBound=0)
          for i in range(n_cand) for j in neighbors[i] }

    # 7.1) objetivo bicritério
    term_sites = pulp.lpSum(x[i] for i in range(n_cand))
    term_flow  = pulp.lpSum(uncov.at[j,"vmd"] * y[j] for j in range(n_cand))
    prob += LAMBDA * term_sites - (1 - LAMBDA) * term_flow, "bicriteria"

    # 7.2) atendimento opcional
    for j in range(n_cand):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            >= uncov.at[j,"kWh_dia"] * y[j]
        ), f"demand_opt_{j}"

    # 7.3) vínculo alocação–estação
    for i in range(n_cand):
        for j in neighbors[i]:
            prob += a[(i,j)] <= uncov.at[j,"kWh_dia"] * x[i], f"link_{i}_{j}"

    # 7.4) capacidade
    for i in range(n_cand):
        node = uncov.at[i,"node"]
        deg  = G.degree[node]
        Qi   = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"cap_{i}"

    # 7.5) cobertura mínima de fluxo por UF
    for uf, n_cars in demanda_veiculos.items():
        Duf = n_cars * kWh_por_carro
        idxs = [j for j,u in enumerate(uncov["UF"]) if u==uf]
        if not idxs:
            continue
        prob += (
            pulp.lpSum(uncov.at[j,"vmd"] * y[j] for j in idxs)
            >= Duf
        ), f"cov_{uf}"

    # 8) resolve com CBC (10 min, 1% gap)
    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=600,
        gapRel=0.01,
        threads=max(1, multiprocessing.cpu_count()-2)
    )
    status = prob.solve(solver)
    print("Status:", pulp.LpStatus[status])

    # 9) extrai e salva solução
    chosen = [i for i in range(n_cand) if x[i].value() > 0.5]
    out = uncov.loc[chosen, ["node","lat","lon","UF"]].copy()
    out["tipo"] = out["node"].map(lambda n: "CIDADE" if G.degree[n]>4 else "RODOVIA")
    out.to_parquet(OUTPUT_DIR/"sites_new_mip_withflow10.parquet", index=False)

    print(f"✓ Gerado {len(chosen)} estações em {time.perf_counter()-t0_all:.1f}s")


if __name__ == "__main__":
    main()