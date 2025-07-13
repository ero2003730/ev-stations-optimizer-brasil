# Projeto de Otimização de Infraestrutura de Carregamento de VE

Este repositório contém uma série de experimentos de otimização para definição de pontos de carregamento de veículos elétricos (EV) no Brasil, usando diferentes modelos de programação inteira mista (MIP). Os scripts estão organizados em duas pastas principais:

- **WithoutFlow**: experimentos que consideram apenas cobertura de demanda por energia (kWh/dia).
- **WithFlow**: experimentos que incorporam também a **demanda de fluxo de tráfego** (VMD – veículos médios diários) na malha rodoviária.

---

## 🚀 Pipeline de Dados

Antes de rodar qualquer modelo, todos os dados brutos são tratados e consolidados no **Jupyter Notebook** [`data.ipynb`](./data.ipynb), que gera diversos arquivos no formato Parquet em `cache/parquet/`. Este notebook:

1. **Baixa e consolida a malha viária** do OSM (motorway | trunk | primary) por estado e compõe o grafo nacional:
   - Saída final: `cache/graph_Brazil.graphml`.
2. **Baixa e processa o tráfego ANTT** (volume em praça de pedágio, CSV):
   - Lê via CKAN da ANTT, descompacta (se necessário) e salva em `dados/antt_YYYY.csv`.
   - Normaliza cabeçalhos e gera `cache/cadastro_pracas.parquet`.
3. **Baixa e processa o tráfego DNIT** (VMDA 2023, Excel → CSV → Parquet):
   - Extrai VMD por rodovia (BR) e quilômetro (km), soma VMD_C + VMD_D, produz `cache/dnit_vmda_2023.parquet`.
4. **Une ANTT + DNIT** em um único DataFrame e resolve duplicidades:
   - Mantém o maior VMD para cada par (BR, km).
   - Salva em `cache/parquet/flux_all.parquet`.
5. **Geocodifica cada ponto de fluxo** (`flux_all.parquet`) ao grafo OSM mais próximo:
   - Usa `osmnx.nearest_nodes` para obter o nó OSM de cada registro.
6. **Gera demanda** agregada por nó:
   - Fórmula:  
     ```
     kWh_dia = VMD × EV_SHARE(1%) × AUTONOMIA(400 km) × CONSUMO(15 kWh/100 km)
     ```
   - Salva em `cache/parquet/demand_by_node.parquet` (colunas: `node`, `kWh_dia`).
7. **Gera oferta existente** a partir de pontos OCM (OpenChargeMap):
   - Agrega número de carregadores e potência total por nó.
   - Salva em `cache/parquet/supply_existing.parquet` (colunas: `node`, `n_chargers`, `pot_kW_total`).

---

## 📑 Descrição dos Principais Conjuntos de Dados

### `flux_all.parquet`

| Coluna | Tipo   | Descrição                                                 |
| ------ | ------ | --------------------------------------------------------- |
| `br`   | string | Código da rodovia (ex.: `BR-116`)                         |
| `km`   | float  | Quilômetro ao longo da rodovia                            |
| `vmd`  | int    | Veículos médios diários (Fonte: ANTT + DNIT)              |
| `lat`  | float  | Latitude do ponto de fluxo (WGS84)                        |
| `lon`  | float  | Longitude do ponto de fluxo (WGS84)                       |
| `src`  | string | Fonte original (`"ANTT"` ou `"DNIT"`)                     |

> **Créditos**: dados ANTT via [dados.antt.gov.br](https://dados.antt.gov.br), DNIT VMDA via DNIT (planilha VMDA 2023).

---

### `demand_by_node.parquet`

| Coluna   | Tipo    | Descrição                                                           |
| -------- | ------- | ------------------------------------------------------------------- |
| `node`   | int64   | Identificador do nó OSM mais próximo (grafo OSMnx)                  |
| `kWh_dia`| float64 | Demanda diária em kWh estimada a partir do VMD e parâmetros de EV  |

Calculado a partir de `flux_all.parquet` e do grafo OSM:  
```text
kWh_dia = vmd × 0.01 × (400 km) × (15 kWh/100 km)
```

---

### `supply_existing.parquet`

| Coluna   | Tipo    | Descrição                                                           |
| -------- | ------- | ------------------------------------------------------------------- |
| `node`   | int64   | Identificador do nó OSM           |
| `n_chargers`| int64 | Número de conectores instalados  |
| `pot_kW_total`| float |Potência Total Instalada (kW)  |

Agregado a partir de pontos da OpenChargeMap:
```text
	•	Fonte: https://openchargemap.org
	•	Parâmetro power_kW calculado ou estimado por conector.
```

---

# WithoutFlow

## 1. Variáveis de decisão
- **x[i]** ∈ {0,1}: = 1 se instalamos estação no candidato _i_
- **a[i,j]** ≥ 0: kWh alocados da estação _i_ para atender o nó _j_

---

## 2. Parâmetros principais
- **R_KM**: raio de cobertura (km)  
- **R_RAD** = R_KM / raio da Terra (rad)  
- **Qᵢ**: capacidade (kWh/dia) da estação _i_ (depende do tipo de nó: cidade ou rodovia)  
- **kWh_dia[j]**: demanda diária em kWh no nó _j_

---

## 3. Função objetivo
Minimizar o número total de estações instaladas:
$$
\min \sum_{i} x[i]
$$

---

## 4. Restrições
1. **Demanda**  
   Para todo nó _j_, a soma da energia alocada deve atender exatamente sua demanda:
   $$
   \sum_{i \in \text{vizinhos}(j)} a[i,j] = \text{kWh\_dia}[j]
   $$
2. **Ligação alocação–estação**  
   Só é possível alocar energia de _i_ para _j_ se a estação _i_ estiver instalada:
   $$
   a[i,j] \le \text{kWh\_dia}[j] \;\cdot\; x[i]
   $$
3. **Capacidade**  
   A energia total distribuída pela estação _i_ não pode exceder sua capacidade se estiver instalada:
   $$
   \sum_{j \in \text{vizinhos}(i)} a[i,j] \le Q_i \;\cdot\; x[i]
   $$

---

## 5. Experimentos “WithoutFlow”  
Cobertura mínima **SEM** considerar fluxo de veículos.

| Experimento           | Usa existentes? | Cobertura por UF? | Fluxo? |
| --------------------- | :-------------: | :---------------: | :----: |
| **1. Normal**         | Sim             | Não               | Não    |
| **2. Sem prévios**    | Não             | Não               | Não    |
| **3. Por Estado (UF)**| Sim             | Sim               | Não    |
| **4. Por UF sem prévios** | Não         | Sim               | Não    |

---

## 6. Variações de raio (R_KM)
Em cada experimento, testamos três valores de R_KM:
- **Sub1**: R_KM = 150 km  
- **Sub2**: R_KM = 50 km  
- **Sub3**: R_KM = 300 km  

> *R_KM* define a distância máxima em que uma estação candidata pode atender um nó.

---

## 7. Como rodar
```bash
cd scripts
bash WithoutFlow.sh
```

---

## 8. Resultados

Os resultados estão disponíveis na pasta `WithoutFlow`, no arquivo: `map_visualization.ipynb`

### WithFlow

## 1. Variáveis de decisão
- **x[i]** ∈ {0,1}: 1 se instalamos estação no candidato _i_
- **y[j]** ∈ {0,1}: 1 se decidimos atender o nó _j_ (conta para fluxo coberto)
- **a[i,j]** ≥ 0: kWh alocados da estação _i_ para atender o nó _j_

---

## 2. Parâmetros principais
- **R_KM**: raio de cobertura (km)  
- **R_RAD** = R_KM / raio da Terra (rad)  
- **Qᵢ**: capacidade (kWh/dia) da estação _i_  
- **kWh_dia[j]**: demanda diária em kWh no nó _j_  
- **vmd[j]**: fluxo diário (veículos/dia) no nó _j_  
- **LAMBDA** ∈ [0,1]: peso do trade-off (#estações vs. fluxo coberto)

---

## 3. Função objetivo
Bicritério: minimizar estações e maximizar fluxo coberto  
$$
\min \; LAMBDA \sum_i x[i] \;-\;(1 - LAMBDA)\sum_j vmd[j]\;y[j]
$$

---

## 4. Restrições
1. **Atendimento opcional**  
   Se _y[j]=1_, atende toda demanda kWh_dia[j]:
   $$
   \sum_{i\in \mathrm{vizinhos}(j)} a[i,j] \;\ge\; kWh\_dia[j]\;y[j]
   $$
2. **Ligação alocação–estação**  
   Só aloca de _i_ para _j_ se _x[i]=1_:
   $$
   a[i,j]\;\le\;kWh\_dia[j]\;x[i]
   $$
3. **Capacidade da estação**  
   Total alocado pela estação _i_ não pode exceder Qᵢ se instalada:
   $$
   \sum_{j\in \mathrm{vizinhos}(i)} a[i,j]\;\le\;Q_i\;x[i]
   $$

---

## 5. Experimentos “WithFlow”
Cobertura **CONSIDERANDO** fluxo de veículos (VMD)

| Experimento                 | Usa existentes? | Cobertura por UF? | Fluxo? |
| --------------------------- | :-------------: | :---------------: | :----: |
| **1. Normal**               | Sim             | Não               | Sim    |
| **2. Sem prévios**          | Não             | Não               | Sim    |
| **3. Por Estado (UF)**      | Sim             | Sim               | Sim    |
| **4. Por UF sem prévios**   | Não             | Sim               | Sim    |

---

## 6. Variações de raio (R_KM)
Em cada experimento, testamos:
- **Sub1**: R_KM = 150 km  
- **Sub2**: R_KM =  50 km  
- **Sub3**: R_KM = 300 km  

> *R_KM* define a distância máxima em que uma estação candidata pode atender um nó.

---

## 7. Como rodar
```bash
cd scripts
bash WithFlow.sh
```

## 8. Resultados

Os resultados estão disponíveis na pasta `WithtFlow`, no arquivo: `map_visualization.ipynb`

---

## ➕ Novos Experimentos de Projeção (2025–2040)

Adicionamos agora duas variações de projeção considerando o crescimento da frota EV no Brasil:

### 1. Projeção **WithFlow + Pontos Existentes**
- **O que**: reutiliza os carregadores já instalados como cobertura inicial e, em seguida, resolve um MIP bicritério (minimiza nº de estações novas e maximiza fluxo coberto) ajustando as metas de cobertura por UF segundo a frota projetada.
- **Onde**: scripts em `Projection/WithFlow/`
- **Arquivos gerados**:  
  - `Projection/WithFlow/2025/sites_new_mip_withflow_2025.parquet`  
  - `Projection/WithFlow/2030/sites_new_mip_withflow_2030.parquet`  
  - `Projection/WithFlow/2035/sites_new_mip_withflow_2035.parquet`  
  - `Projection/WithFlow/2040/sites_new_mip_withflow_2040.parquet`  
- **Visualização**:  
  - Jupyter cell `map_visualization.py` plota, lado a lado, os quatro mapas comparando pontos existentes (azul) e projetados (vermelho).

### 2. Projeção **Scratch + Sem Pontos Existentes**
- **O que**: parte do zero (ignora oferta atual) e resolve MIP bicritério, com metas ajustadas pela frota projetada por UF.
- **Onde**: scripts em `Projection/WithoutFlow/`
- **Arquivos gerados**:  
  - `Projection/WithoutFlow/2025/sites_new_mip_scratch_2025.parquet`  
  - `Projection/WithoutFlow/2030/sites_new_mip_scratch_2030.parquet`  
  - `Projection/WithoutFlow/2035/sites_new_mip_scratch_2035.parquet`  
  - `Projection/WithoutFlow/2040/sites_new_mip_scratch_2040.parquet`  
- **Visualização**:  
  - Jupyter cell `map_visualization.py` plota os quatro mapas apenas com os pontos projetados (vermelho), em um grid 2×2.

---

### Como rodar as projeções

```bash
# Projeções com pontos existentes (WithFlow)
cd Scripts
./Projections.sh
```
---

### Como visualizar

1. Abra o notebook `Projection/map_visualization.ipynb`  
2. Execute as duas células:  
   - **Map Visualization With Points**  
   - **Map Visualization Without Points**  

## 🔗 Referências e Créditos

	•	OSMnx / OpenStreetMap: osmnx para baixar e processar a malha viária.
	•	ANTT: CKAN API para volume de tráfego em praças de pedágio.
	•	DNIT: Planilha VMDA 2023 para fluxos de tráfego em rodovias.
	•	Climate Scorecard: *The Growth of Electric Vehicles in Brazil: a sustainable revolution*.
	•	OpenChargeMap: banco de dados de estações de carregamento existentes.
	•	Geobr: limites estaduais (UF) para cobertura mínima por estado.
