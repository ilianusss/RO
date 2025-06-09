"""
Planification de flotte de drones
"""

import math
import random
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point
from sklearn.cluster import KMeans

from eulerian import find_eulerian


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _eucl_dist_matrix(pts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice des distances euclidiennes entre chaque point
    et chaque centre.
    """
    # pts : tableau de forme (N_arêtes, 2)
    # centers : tableau de forme (N_bases, 2)
    return np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)


def _edge_centroids_lengths(G: nx.MultiDiGraph):
    """
    Extrait pour chaque arête de G :
    - son identifiant (u, v, k)
    - son centre
    - sa longueur en km
    """
    edges = []
    coords = []
    lengths_km = []

    for u, v, k, data in G.edges(keys=True, data=True):
        # On récupère la geométrie si elle existe 
        geom = data.get("geometry")
        if geom is not None:
            mid = geom.interpolate(0.5, normalized=True).coords[0]
        else:
            # Si pas de géométrie, on la calcule 
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            mid = ((x1 + x2) / 2, (y1 + y2) / 2)

        edges.append((u, v, k))
        coords.append(mid)
        lengths_km.append(data["length"] / 1000.0)

    return edges, np.array(coords), lengths_km


def _recompute_centers(coords: np.ndarray, clusters: list, old_centers: np.ndarray) -> np.ndarray:
    """
    A partir des aretes du cluster, on replace le centre au mieux 
    Si un cluster n'a pas d'arêtes, conserve l'ancienne position ou (0,0).
    """
    new_centers = []
    for i, idxs in enumerate(clusters):
        if idxs:
            # Moyenne des coordonnées des aretes dans le cluster
            new_centers.append(coords[idxs].mean(axis=0))
        else:
            # Aucun élément => conserve l'ancien centre si possible, sinon (0,0)
            if i < len(old_centers):
                new_centers.append(old_centers[i])
            else:
                new_centers.append(np.zeros(2))
    return np.array(new_centers)


def _capacitated_assignment(edges, coords, lengths_km, centers, B: int):
    """
    Affectation des aretes vers les centres
    On calcule un seuil cap_each basé sur la longueur totale / B, arrondi à 105 km
    On affecte chaque arete (de la plus longue a la plus courte) au centre
    le plus proche possible sans dépasser ce seuil
    """
    # Seuil de km pour chaque base
    cap_each = math.ceil((sum(lengths_km) / B) / 105.0) * 105.0

    # Matrice distances entre les aretes et les centres
    dmat = _eucl_dist_matrix(coords, centers)

    # Tri des indices d'arêtes par longueur décroissante
    idx_by_len = sorted(range(len(edges)), key=lambda i: lengths_km[i], reverse=True)

    cluster_sum = [0.0] * B
    assignment = [-1] * len(edges)

    for i in idx_by_len:
        # Tente d'affecter à la base la plus proche sans dépasser cap_each
        for c in np.argsort(dmat[i]):
            if cluster_sum[c] + lengths_km[i] <= cap_each:
                assignment[i] = int(c)
                cluster_sum[c] += lengths_km[i]
                break
        # Si aucune base n'a pu prendre l'arête, l'envoyer à la plus proche quoi qu'il arrive
        if assignment[i] == -1:
            closest = int(np.argmin(dmat[i]))
            assignment[i] = closest
            cluster_sum[closest] += lengths_km[i]

    # Regrouper les indices par cluster
    clusters = [[] for _ in range(B)]
    for i, c in enumerate(assignment):
        clusters[c].append(i)

    return clusters


def _split_sequence_with_return(lengths_km: list, full_seq: list, center: tuple, cap_total=105.0):
    """
    Découpe une séquence d'arêtes (full_seq) en tournées pour un drone,
    en s'assurant qu'avec le trajet aller-retour à la base, on ne dépasse pas cap_total.
    Retourne une liste de listes (tours), chaque tour étant une sous-séquence d'arêtes.
    """
    tours = []
    n = len(full_seq)
    i = 0

    while i < n:
        # Calculer la distance aller-retour minimale pour l'arête de départ
        midpoint_i = full_seq[i]["geometry"].interpolate(0.5, normalized=True).coords[0]
        dist_base = (
            ox.distance.great_circle(midpoint_i[1], midpoint_i[0], center[1], center[0])
            / 1000.0
        )
        # Distance disponible pour le circuit, sans dépasser cap_total
        allowable = cap_total - 2.0 * dist_base
        if allowable < 0:
            allowable = 0.0

        acc = 0.0
        j = i
        # Empiler tant que l'on peut ajouter la longueur de l'arête suivante
        while j < n and acc + lengths_km[j] <= allowable + 1e-9:
            acc += lengths_km[j]
            j += 1

        # Au moins une arête par tournée
        if j == i:
            j = i + 1

        tours.append(full_seq[i:j])
        i = j

    return tours


def _balanced_split(lengths_km_all: list, edges_idx: list):
    """
    Quand un cluster produit trop de tournées (K_b > 25),
    on découpe ses indices d'arêtes en deux sous-ensembles équilibrés.
    On attribue tour à tour l'arête la plus longue à la partie la moins chargée.
    """
    # Liste de tuples (index_arête, longueur_km)
    items = [(i, lengths_km_all[i]) for i in edges_idx]
    items.sort(key=lambda x: x[1], reverse=True)

    part1, part2 = [], []
    sum1 = sum2 = 0.0
    for idx, ln in items:
        if sum1 <= sum2:
            part1.append(idx)
            sum1 += ln
        else:
            part2.append(idx)
            sum2 += ln

    return part1, part2


# ────────────────────────────────────────────────────────────────────────────────
# Traitement des clusters (CPP + découpe)
# ────────────────────────────────────────────────────────────────────────────────

# Cache global pour éviter de recomputations coûteuses
_cache = {}


def _process_cluster(args):
    """
    Pour un sous-ensemble d'arêtes (edges_idx) et un centre donné :
    1) Construire le sous-graphe induit par ces arêtes
    2) Pour chaque composante faiblement connexe, résoudre le problème du tour eulérien (CPP)
       grâce à la fonction find_eulerian(sub)
    3) Découper le circuit en tournées équilibrées (_split_sequence_with_return)
    4) Si trop de tournées (K_b > 25), on split récursivement (_balanced_split)
    Renvoie un tuple : (tours, K_b, distance_totale_km, distance_duplication_km)
    """
    edges, G_original, edges_idx, lengths_all, center = args
    key = tuple(sorted(edges_idx))

    # Vérifier cache
    if key in _cache:
        return _cache[key]

    # 1) Construire le sous-graphe induit
    G_sub = nx.MultiDiGraph()
    G_sub.graph.update(G_original.graph)
    G_sub.add_nodes_from(G_original.nodes(data=True))
    for i in edges_idx:
        u, v, k = edges[i]
        G_sub.add_edge(u, v, **G_original[u][v][k])

    # 2) Extraire circuits eulériens et constituer full_seq
    full_seq = []
    lens_km_seq = []
    dup_km = 0.0

    for comp in nx.weakly_connected_components(G_sub):
        sub = G_sub.subgraph(comp).copy()
        circuit = find_eulerian(sub)  # liste de tuples (u,v,k)
        for u, v, k in circuit:
            data = sub[u][v][k]
            ln_km = data["length"] / 1000.0
            lens_km_seq.append(ln_km)
            if data.get("duplicate", False):
                dup_km += ln_km

            geom = data.get("geometry")
            if geom is None:
                # Si pas de géométrie, créer un segment entre nœuds
                x1, y1 = sub.nodes[u]["x"], sub.nodes[u]["y"]
                x2, y2 = sub.nodes[v]["x"], sub.nodes[v]["y"]
                geom = LineString([(x1, y1), (x2, y2)])

            full_seq.append({
                "u": u,
                "v": v,
                "geometry": geom,
                "length_m": data["length"],
                "duplicate": data.get("duplicate", False)
            })

    # Si aucun circuit (cluster vide), on retourne zéros
    if not full_seq:
        result = ([], 0, 0.0, 0.0)
        _cache[key] = result
        return result

    total_dist_km = sum(lens_km_seq)

    # 3) Découpe en tournées tenant compte du retour à la base
    tours = _split_sequence_with_return(lens_km_seq, full_seq, center, cap_total=105.0)
    K_b = len(tours)

    # 4) Si trop de tournées, on split le cluster en deux et combine récursivement
    if K_b > 25:
        part1, part2 = _balanced_split(lengths_all, edges_idx)
        res1 = _process_cluster((edges, G_original, part1, lengths_all, center))
        res2 = _process_cluster((edges, G_original, part2, lengths_all, center))
        # Combinaison des résultats
        combined = (
            res1[0] + res2[0],  # tours combinés
            res1[1] + res2[1],  # nombre total de tournées
            res1[2] + res2[2],  # distance totale
            res1[3] + res2[3]   # distance duplications
        )
        _cache[key] = combined
        return combined

    result = (tours, K_b, total_dist_km, dup_km)
    _cache[key] = result
    return result


# ────────────────────────────────────────────────────────────────────────────────
# Validate the solution
# ────────────────────────────────────────────────────────────────────────────────

def _validate_solution(plan: list, centers: np.ndarray) -> bool:
    """
    Vérifie que chaque tournée respecte la contrainte :
    distance aller-retour + circuit ≤ 105 km.
    plan : liste de listes de tournées (chaque tournée est liste d'arêtes)
    centers : array de shape (B,2) donnant la position de chaque base
    """
    for i_base, tours in enumerate(plan):
        bx, by = centers[i_base]
        for tour in tours:
            if not tour:
                continue
            # Calcul du point de départ (centroïde de la première arête)
            midpoint0 = tour[0]["geometry"].interpolate(0.5, normalized=True).coords[0]
            # Distance aller-retour minimale
            dist_base = math.hypot(midpoint0[0] - bx, midpoint0[1] - by) / 1000.0
            # Longueur du circuit en km
            km_circuit = sum(e["length_m"] for e in tour) / 1000.0
            total_km = 2.0 * dist_base + km_circuit
            # Si on dépasse la capacité (tolérance 1e-6 km)
            if total_km - 105.0 > 1e-6:
                return False
    return True


# ────────────────────────────────────────────────────────────────────────────────
# Enforce connectivity
# ────────────────────────────────────────────────────────────────────────────────

def _enforce_connectivity(clusters: list, edges: list, coords: np.ndarray, centers: np.ndarray) -> list:
    """
    Pour chaque cluster, on identifie les composantes faiblement connexes “orphelines”.
    Chaque sous-composante non-connectée est réaffectée à la base la plus proche
    en distance euclidienne de son centroïde.
    Renvoie la nouvelle liste de clusters.
    """
    B = len(clusters)
    new_clusters = [[] for _ in range(B)]
    # Dictionnaire pour retrouver l'indice d'une arête à partir de (u,v,k)
    edge2idx = {edges[i]: i for i in range(len(edges))}

    for c_id, lst in enumerate(clusters):
        if not lst:
            continue

        # Construire sous-graphe induit
        H = nx.MultiDiGraph()
        for i in lst:
            u, v, k = edges[i]
            H.add_edge(u, v, key=k)

        # Parcourir chaque composante faiblement connexe de H
        for comp in nx.weakly_connected_components(H):
            comp_edges = []
            # Extraire toutes les arêtes dont l'un des nœuds est dans comp
            for u, v, k in H.edges(keys=True):
                if u in comp or v in comp:
                    comp_edges.append(edge2idx[(u, v, k)])

            if not comp_edges:
                continue

            if len(comp_edges) == len(lst):
                # C'est la composante principale, on la garde dans le même cluster
                new_clusters[c_id].extend(comp_edges)
            else:
                # C'est une "miette" : calcul du centroïde et réaffectation
                pts = coords[comp_edges].mean(axis=0)
                target = np.argmin(np.linalg.norm(centers - pts, axis=1))
                new_clusters[target].extend(comp_edges)

    return new_clusters


# ────────────────────────────────────────────────────────────────────────────────
# Main algorithm
# ────────────────────────────────────────────────────────────────────────────────

def plan_drone_fleet(G: nx.MultiDiGraph):
    """
    Entrée :
    - G : graphe orienté (MultiDiGraph) dont chaque arête possède une longueur et éventuellement une géométrie.
    Sortie :
    - best_centers : coordonnées des bases (array de shape (B,2))
    - best_plan : liste de listes de tournées pour chaque base
    ------------------------------------------------------------
    Étapes :
    1) Extraction des centroïdes et longueurs des arêtes
    2) Pour 5 essais avec une seed aléatoire à chaque fois, et pour B variant de 11 à 4 :
       a) K-Means++ (50 essais pour l'inertie la plus faible)
       b) Affectation capacitaire (_capacitated_assignment)
       c) Enforce connectivité (_enforce_connectivity)
       d) Recalcul des centres (_recompute_centers)
       e) Calcul des tournées en parallèle (_process_cluster)
       f) Évaluation du coût et validation (_validate_solution)
       g) Sauvegarde si meilleur coût (avec en-tête “Global seed” et “KMeans seed”)
    """
    # 1) Extraction des informations d'arêtes
    edges, coords, lengths_km = _edge_centroids_lengths(G)

    best_cost = math.inf
    best_centers = None
    best_plan = None
    best_global_seed = None
    best_kmeans_state = None

    # Répertoire de sortie pour les résultats intermédiaires
    output_root = Path("assets/drone_fleet")
    output_root.mkdir(parents=True, exist_ok=True)

    # On fait 5 essais avec une seed aléatoire par essai
    for _ in range(5):
        # Générer une seed "globale" aléatoire (0 ≤ seed < 1e6)
        seed = random.randint(0, int(1e6))
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n--- Début seed {seed} ---")

        b_max = 11
        # B_target descend de 11 à 5
        for B_target in range(b_max, 4, -1):
            print(f"  → Essai avec B={B_target} bases (seed {seed})")
            t0 = time.time()

            # a) K-Means++ initial (50 essais pour l'inertia la plus faible)
            inertia_best = math.inf
            centers = None
            chosen_kstate = None

            for _ in range(50):
                kstate = random.randint(0, int(1e6))
                km = KMeans(
                    n_clusters=B_target,
                    init="k-means++",
                    n_init=1,
                    random_state=kstate
                ).fit(coords)

                if km.inertia_ < inertia_best:
                    inertia_best = km.inertia_
                    centers = km.cluster_centers_.copy()
                    chosen_kstate = kstate

            print(f"    · K-Means initial (50 itérations) terminé (inertia={inertia_best:.2f})")

            # b) Affectation capacitaire des arêtes aux B_target centres
            clusters = _capacitated_assignment(edges, coords, lengths_km, centers, B_target)
            print("    · Affectation capacitaire terminée")

            # c) Enforce connectivité : retrait des “miettes” et réaffectations
            clusters = _enforce_connectivity(clusters, edges, coords, centers)
            print("    · Connectivité assurée (clusters faiblement connexes)")

            # d) Recalculer les centres après réaffectation
            centers = _recompute_centers(coords, clusters, centers)

            # e) Calcul des tournées pour chaque cluster en parallèle
            args = [
                (edges, G, clusters[i], lengths_km, centers[i])
                for i in range(len(clusters))
            ]
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(_process_cluster, args))

            print("    · Traitement des clusters terminé")

            # Calcul du coût global : K_total * 100 € + 0.01 € * distance_totale
            total_K = sum(res[1] for res in results)
            total_dist = sum(res[2] for res in results)
            cost_glob = total_K * 100.0 + 0.01 * total_dist
            t1 = time.time()
            print(
                f"      ≫ coût={cost_glob:.2f} € "
                f"(meilleur={best_cost:.2f} €) en {t1 - t0:.2f}s"
            )

            # f) Si meilleur coût et solution valide, on sauvegarde les résultats
            if cost_glob < best_cost and _validate_solution([res[0] for res in results], centers):
                best_cost = cost_glob
                best_centers = centers.copy()
                best_plan = [res[0] for res in results]
                best_global_seed = seed
                best_kmeans_state = chosen_kstate

                # Nettoyer l'ancien dossier de sortie
                shutil.rmtree(output_root, ignore_errors=True)
                output_root.mkdir(parents=True, exist_ok=True)
                dir_save = output_root

                # 6) Sauvegarde des bases en GeoJSON
                pts = [
                    {"base": i + 1, "geometry": Point(xy[0], xy[1])}
                    for i, xy in enumerate(best_centers)
                ]
                gpd.GeoDataFrame(pts, crs=G.graph["crs"]).to_file(
                    dir_save / "bases.geojson", driver="GeoJSON"
                )

                # 7) Détail des tournées + GeoJSON par drone
                detail_lines = []
                tot_drones = 0
                tot_distance = 0.0  # Somme des circuits (km)

                # En-têtes de seed
                detail_lines.append(f"Global seed = {best_global_seed}.")
                detail_lines.append(f"KMeans seed = {best_kmeans_state}.")
                detail_lines.append("")  # ligne vide

                for i_base, tours in enumerate(best_plan, start=1):
                    nb_dr = len(tours)
                    tot_drones += nb_dr

                    bx, by = best_centers[i_base - 1]
                    base_dir = dir_save / f"base_{i_base}_{nb_dr}_drones"
                    base_dir.mkdir(parents=True, exist_ok=True)

                    # Distance totale parcours (sans trajet aller-retour)
                    total_circuit = sum(
                        sum(e["length_m"] for e in t) for t in tours
                    ) / 1000.0

                    # Calcul de la distance aller-retour pour chaque tour
                    total_travel = 0.0
                    for tour in tours:
                        if tour:
                            mpt = tour[0]["geometry"].interpolate(0.5, normalized=True).coords[0]
                            d = ox.distance.great_circle(mpt[1], mpt[0], by, bx) / 1000.0
                            total_travel += 2.0 * d

                    base_cost = nb_dr * 100.0 + 0.01 * (total_circuit + total_travel)
                    detail_lines.append(
                        f"→ Base {i_base} (coord : {bx:.1f},{by:.1f}) "
                        f"{total_circuit:.1f} km (dont {total_travel:.1f} km de trajet) "
                        f"→ coût {base_cost:.2f} € :"
                    )
                    tot_distance += total_circuit

                    for j, tour in enumerate(tours, start=1):
                        if tour:
                            mpt = tour[0]["geometry"].interpolate(0.5, normalized=True).coords[0]
                            travel = 2.0 * (
                                ox.distance.great_circle(mpt[1], mpt[0], by, bx) / 1000.0
                            )
                        else:
                            travel = 0.0

                        circuit = sum(e["length_m"] for e in tour) / 1000.0
                        total_km = travel + circuit

                        detail_lines.append(
                            f"  • Drone {j} : {total_km:.1f} km "
                            f"({travel:.1f} km de trajet) → coût {100.0 + 0.01 * total_km:.2f} €"
                        )

                        # Construire GeoDataFrame pour chaque drone (même si vide)
                        rows = []
                        for e in tour:
                            rows.append({
                                "u": e["u"],
                                "v": e["v"],
                                "length": e["length_m"],
                                "geometry": e["geometry"],
                                "duplicate": e["duplicate"]
                            })

                        if rows:
                            gpd.GeoDataFrame(rows, crs=G.graph["crs"]).to_file(
                                base_dir / f"drone_{j}.geojson", driver="GeoJSON"
                            )
                        else:
                            empty = gpd.GeoDataFrame(
                                [], columns=["u", "v", "length", "geometry", "duplicate"],
                                crs=G.graph["crs"]
                            )
                            empty.to_file(
                                base_dir / f"drone_{j}.geojson", driver="GeoJSON"
                            )

                # Lignes finales après un saut de ligne
                detail_lines.append("")
                detail_lines.append(f"Total bases = {len(best_plan)}")
                detail_lines.append(f"Total drones = {tot_drones}")
                detail_lines.append(f"Total distance = {tot_distance:.1f} km.")
                detail_lines.append(f"Total cost = {best_cost:.2f} €")

                # Écriture du fichier detail.txt
                (dir_save / "detail.txt").write_text(
                    "\n".join(detail_lines), encoding="utf8"
                )

    return best_centers, best_plan


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    G = ox.load_graphml("assets/graph/graphml/graphe_montreal_eulerian.graphml")
    centers, plan = plan_drone_fleet(G)

    if centers is None or plan is None:
        print("Aucune solution valide n'a été trouvée.")
    else:
        print("\nPlan final sauvegardé")


if __name__ == "__main__":
    main()
