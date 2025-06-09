"""
Calcule le circuit eulérien et l'exporte.
"""

import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import time


# ----------------------------------------------------------------------
def _pair_odd_nodes(G: nx.MultiDiGraph, odd_nodes: List[int]) -> List[Tuple[int, int]]:
    """Trouve la meilleure manière d'équilibrer le graphe grâce à Dijkstra et l'algorithme Hongrois"""
    # On sépare les nœuds impairs selon leur excédent
    plus  = [n for n in odd_nodes for _ in range(G.out_degree(n) - G.in_degree(n))
             if G.out_degree(n) > G.in_degree(n)]
    minus = [n for n in odd_nodes for _ in range(G.in_degree(n) - G.out_degree(n))
             if G.out_degree(n) < G.in_degree(n)]
    assert len(plus) == len(minus), "[ERROR] Graph is not balanced"
    

    # On calcule les distances avec Dijkstra et crée la matrice de couts
    print("Création de la matrice de coûts...")
    matrix_start = time.time()
    BIG = 10**12  # coût pour les paires inaccessibles
    Gu = G.to_undirected(as_view=True)
    def _row(u):
        dist_u = nx.single_source_dijkstra_path_length(Gu, u, weight="length")
        return [dist_u.get(v, BIG) for v in plus]
    
    with ThreadPoolExecutor() as exe:
        cost_matrix = list(exe.map(_row, minus))
    
    matrix_time = time.time() - matrix_start
    print(f"Matrice créée ({len(minus)} lignes, {len(plus)} colonnes) en {matrix_time:.2f} secondes")

    # On trouve la meilleure option (couples de noeuds à joindre)
    print("Calcul des meilleurs chemins...")
    path_start = time.time()
    
    # La fonction appelle l'algorithme Hongrois
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs = [(minus[i], plus[j]) for i, j in zip(row_ind, col_ind)]
    
    path_time = time.time() - path_start
    print(f"Meilleurs chemins trouvés ({len(pairs)} paires) en {path_time:.2f} secondes")
    
    return pairs


def _add_matching_edges(G: nx.MultiDiGraph, pairs: List[Tuple[int, int]]):
    """Crée les nouvelles aretes et les ajoute au graph"""
    Gu = G.to_undirected(as_view=True)
    print("Ajout des arêtes")
    start_time = time.time() 
    # Statistiques sur les arêtes ajoutées
    edges_added = 0
    total_length = 0
    
    for u, v in pairs:
        # (Chemin le plus court pour relier les deux noeuds)
        # La fonction appelle un Dijkstra bidirectionnel 
        path = nx.shortest_path(Gu, u, v, weight="length")
        # On ajoute ce chemin au graph 
        for a, b in zip(path, path[1:]):
            if G.has_edge(a, b):
                data = G[a][b][0]
                new_data = data.copy()
            elif G.has_edge(b, a):
                data = G[b][a][0]
                new_data = data.copy()
                # Géométrie inversée pour respecter l'orientation
                if "geometry" in new_data:
                    new_data["geometry"] = LineString(new_data["geometry"].coords[::-1])
            else:
                continue
            new_data["duplicate"] = True
            G.add_edge(a, b, **new_data)
            edges_added += 1
            total_length += new_data["length"]
    
    if edges_added > 0:
        avg_length = total_length / edges_added / 1000  # Conversion en km
        total_length_km = total_length / 1000
        print(f"{edges_added} arêtes créées, {total_length_km:.1f} km d'arêtes en {time.time() - start_time:.2f} secondes")


def find_eulerian(G: nx.MultiDiGraph) -> List[Tuple[int, int, int]]:
    """Return an eulerian circuit"""
    print("Recherche du circuit eulérien...")

    # Balance degrees
    odd = [n for n in G.nodes if G.out_degree(n) != G.in_degree(n)]
    if odd:
        print(f"Graphe non équilibré. {len(odd)} noeuds déséquilibrés")
        pairs = _pair_odd_nodes(G, odd)
        _add_matching_edges(G, pairs)

    # Ensure eulerian
    assert all(G.out_degree(n) == G.in_degree(n) for n in G.nodes), "[ERROR] Graph not balanced"
    print("Graphe équilibré, extraction du circuit eulérien...")
    circuit_start = time.time()
    
    # Extract eulerian circuit
    circuit = list(nx.eulerian_circuit(G, keys=True))
    
    circuit_time = time.time() - circuit_start
    print(f"Circuit eulérien extrait ({len(circuit)} arêtes) en {circuit_time:.2f} secondes")
    
    return circuit


def circuit_to_geodataframe(circuit: List[Tuple[int, int, int]], G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    # Convert circuit (u,v,key) to GeoDataFrame
    res_cir = []
    for u, v, k in circuit:
        data = G[u][v][k]
        geom = data.get("geometry")
        if geom is None:
            # build straight line if no geometry stored
            geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        length = data["length"]
        res_cir.append({"u": u, "v": v, "key": k, "length": length, "geometry": geom})
    return gpd.GeoDataFrame(res_cir, crs=G.graph["crs"])


def export_circuit(circuit_gdf: gpd.GeoDataFrame, out_path: Path):
    #Save circuit as GeoJSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    circuit_gdf.to_file(out_path, driver="GeoJSON")


def validate_eulerian_circuit(circuit: List[Tuple[int, int, int]], G: nx.MultiDiGraph) -> bool:
    print("Validation du circuit eulérien...")
    start_time = time.time()
    if not circuit:
        print("[ERROR] Le circuit est vide")
        return False
    
    # Vérification que le circuit est fermé
    if circuit[0][0] != circuit[-1][1]:
        print(f"[ERROR] Le circuit n'est pas fermé: débute à {circuit[0][0]} et termine à {circuit[-1][1]}")
        return False
    
    # Création d'un ensemble pour vérifier que chaque arête est parcourue exactement une fois
    edges_in_circuit = set()
    for u, v, k in circuit:
        if (u, v, k) in edges_in_circuit:
            print(f"[ERROR] L'arête ({u}, {v}, {k}) est parcourue plusieurs fois")
            return False
        edges_in_circuit.add((u, v, k))
    
    # Vérification que toutes les arêtes du graphe original sont dans le circuit
    original_edges = set((u, v, k) for u, v, k in G.edges(keys=True) 
                        if not G[u][v][k].get("duplicate", False))
    missing_edges = original_edges - edges_in_circuit
    if missing_edges:
        print(f"[ERROR] {len(missing_edges)} arêtes originales ne sont pas parcourues")
        return False
    
    validation_time = time.time() - start_time
    print(f"[SUCCESS] Le circuit est un cycle eulérien valide (validation en {validation_time:.2f} secondes)")
    return True


# ----------------------------------------------------------------------
def main():
    print("Démarrage du calcul du circuit eulérien...")
    
    print("Chargement du graphe...")
    load_start = time.time()
    G: nx.MultiDiGraph = ox.load_graphml("assets/graph/graphml/graphe_montreal.graphml")
    load_time = time.time() - load_start
    print(f"Graphe chargé ({G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes) en {load_time:.2f} secondes")
    
    # Trouve le circit Eulerien
    circuit = find_eulerian(G)
    
    # Vérifie la validité du circuit
    is_valid = validate_eulerian_circuit(circuit, G)
    if not is_valid:
        print("[WARNING] Le circuit eulérien calculé n'est pas valide!")
    
    circuit_gdf = circuit_to_geodataframe(circuit, G)
    total_len = circuit_gdf["length"].sum()
    total_km = total_len / 1000
    print(f"Taille du circuit: {total_km:.1f} km")
    total_cost = 100 + (total_km * 0.01)
    print(f"Soit un cout total de: {total_cost:.2f} €")
    export_circuit(circuit_gdf, Path("assets/graph/geojson/graphe_montreal_eulerian.geojson"))
    ox.save_graphml(G, filepath="assets/graph/graphml/graphe_montreal_eulerian.graphml")


if __name__ == "__main__":
    main()
