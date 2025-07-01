#!/usr/bin/env python3
# 01_mip_solver_by_state_scratch_projection.py
# Projeções para 2025, 2030, 2035 e 2040:
# MIP bicritério: min estações + max VMD coberto,
# cobertura mínima por UF, começando do zero

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
R_KM, R_EARTH     = 50, 6371.0088
R_RAD            = R_KM / R_EARTH
N_HWY, KW_HWY    = 10, 120
N_CITY, KW_CITY  = 5,  22
kWh_por_carro    = 20_000   # consumo médio diário por veículo
LAMBDA           = 0.9      # peso entre #estações (λ) e fluxo coberto (1−λ)

# frota EV atual por UF (base 2025)
demanda_veiculos = {
    "SP":28350, "PR":7817, "SC":6666, "RS":7002,
    "RJ":8065,  "ES":2791, "BA":3962, "SE":669,
    "AL":1518,  "PE":4368, "PB":1038, "RN":2249,
    "CE":2854,  "PI":981,  "TO":378,  "MA":1604,
    "PA":977,   "AP":163,  "RR":218,  "AM":1640,
    "MT":1215,  "GO":2359, "DF":11115,"MS":851,
    "MG":5625,  "RO":667,  "AC":157
}

# ─── fatores de projeção (NEEL/Climate Scorecard) ─────────
# base 2025 = 1.0
projection_factors = {
    2025: 1.00,
    2030: 2.85,  # 300 000 / 105 299 veículos 
    2035: 4.75,  # 500 000 / 105 299 veículos 
    2040: 7.03   # extrapolação linear (~740 000 / 105 299)
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

# grafo
GRAPHML       = BASE.parent/"graph_Brazil.graphml"
# saída de projeções sem fluxo prévio
PROJ_BASE_DIR = BASE/"Projection"/"WithoutFlow"

def solve_for_year_scratch(year: int, factor: float):
    print(f"\n▶ Ano {year} (fator {factor:.2f})")
    t0 = time.perf_counter()

    # 1) carrega grafo, demanda e fluxo
    G    = ox.load_graphml(str(GRAPHML))
    dem  = pd.read_parquet(BASE/"demand_by_node.parquet")
    dem["kWh_dia"] = dem["kWh_dia"] * factor
    flux = pd.read_parquet(BASE/"flux_all.parquet")

    # 2) mapeia lon/lat em dem
    lon = {n: d["x"] for n,d in G.nodes(data=True)}
    lat = {n: d["y"] for n,d in G.nodes(data=True)}
    dem["lon"] = dem["node"].map(lon)
    dem["lat"] = dem["node"].map(lat)

    # 3) atribui VMD ao nó de demanda mais próximo
    tree_flux = BallTree(np.deg2rad(flux[["lat","lon"]]), metric="haversine")
    _, idx = tree_flux.query(np.deg2rad(dem[["lat","lon"]]), k=1)
    dem["vmd"] = flux["vmd"].values[idx.flatten()]

    # 4) etiqueta UFs
    ufs = geobr.read_state(year=2020)[["abbrev_state","geometry"]]
    ufs = ufs.rename(columns={"abbrev_state":"UF"}).to_crs("EPSG:4326")
    gdem = gpd.GeoDataFrame(dem, geometry=gpd.points_from_xy(dem.lon, dem.lat), crs="EPSG:4326")
    dem["UF"] = (
        gpd.sjoin(gdem, ufs, how="left", predicate="within")["UF"]
           .fillna("UNK")
           .values
    )

    # 5) parte do zero: todos os nós são candidatos
    uncov = dem.reset_index(drop=True)
    n_cand = len(uncov)
    print(f"  • {n_cand} candidatos totais")

    # 6) vizinhanças de cobertura
    coords    = np.deg2rad(uncov[["lat","lon"]].values)
    neighbors = BallTree(coords, metric="haversine").query_radius(coords, r=R_RAD)

    # 7) monta modelo MIP
    prob = pulp.LpProblem(f"EV_scratch_{year}", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n_cand), cat="Binary")
    y = pulp.LpVariable.dicts("y", range(n_cand), cat="Binary")
    a = {
        (i,j): pulp.LpVariable(f"a_{i}_{j}", lowBound=0)
        for i in range(n_cand) for j in neighbors[i]
    }

    # 7.1) objetivo bicritério
    term_sites = pulp.lpSum(x[i] for i in range(n_cand))
    term_flow  = pulp.lpSum(uncov.at[j,"vmd"] * y[j] for j in range(n_cand))
    prob += LAMBDA * term_sites - (1 - LAMBDA) * term_flow

    # 7.2) atendimento opcional
    for j in range(n_cand):
        prob += (
            pulp.lpSum(a[(i,j)] for i in neighbors[j])
            >= uncov.at[j,"kWh_dia"] * y[j]
        ), f"demand_opt_{j}"

    # 7.3) vínculo fluxo × estação
    for i in range(n_cand):
        for j in neighbors[i]:
            prob += a[(i,j)] <= uncov.at[j,"kWh_dia"] * x[i], f"link_{i}_{j}"

    # 7.4) capacidade por estação
    for i in range(n_cand):
        node = uncov.at[i,"node"]
        deg  = G.degree[node]
        Qi   = (N_CITY*KW_CITY*24) if deg>4 else (N_HWY*KW_HWY*24)
        prob += (
            pulp.lpSum(a[(i,j)] for j in neighbors[i])
            <= Qi * x[i]
        ), f"cap_{i}"

    # 7.5) cobertura mínima por UF (ajustada ao fator)
    dem_ajustada = { uf: int(n*factor) for uf,n in demanda_veiculos.items() }
    for uf, n_cars in dem_ajustada.items():
        Duf = n_cars * kWh_por_carro
        idxs = [j for j,u in enumerate(uncov["UF"]) if u==uf]
        if not idxs: 
            continue
        prob += (
            pulp.lpSum(uncov.at[j,"vmd"] * y[j] for j in idxs)
            >= Duf
        ), f"cov_{uf}_{year}"

    # 8) resolve com CBC
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=600,
        gapRel=0.01,
        threads=max(1, multiprocessing.cpu_count()-2)
    )
    status = prob.solve(solver)
    print(f"  • Status: {pulp.LpStatus[status]}")

    # 9) extrai e salva solução
    chosen = [i for i in range(n_cand) if x[i].value() > 0.5]
    out = uncov.loc[chosen, ["node","lat","lon","UF"]].copy()
    out["tipo"] = out["node"].map(lambda n: "CIDADE" if G.degree[n]>4 else "RODOVIA")

    output_dir = PROJ_BASE_DIR/str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/f"sites_new_mip_scratch_{year}.parquet"
    out.to_parquet(path, index=False)

    print(f"✓ Ano {year}: {len(chosen)} estações geradas em {time.perf_counter()-t0:.1f}s → {path}")

def main():
    for yr, fac in projection_factors.items():
        solve_for_year_scratch(yr, fac)

if __name__ == "__main__":
    main()