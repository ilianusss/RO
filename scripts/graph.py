"""
Télécharge (ou charge) le graphe de Montréal et le nettoie
"""

import argparse
from pathlib import Path
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Configuration de OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True

GRAPH_FPATH = Path("assets/graph/graphml/graphe_montreal.graphml")


# ----------------------------------------------------------------------
def load_or_download_graph() -> nx.MultiDiGraph:
    if GRAPH_FPATH.exists():
        print(f"Chargement depuis {GRAPH_FPATH}")
        return ox.load_graphml(str(GRAPH_FPATH))

    print("Téléchargement depuis OpenStreetMap…")
    G = ox.graph_from_place(
        "Montréal, Québec, Canada",
        network_type="drive_service",
        truncate_by_edge=True
    )
    GRAPH_FPATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Sauvegarde dans {GRAPH_FPATH}")
    ox.save_graphml(G, filepath=str(GRAPH_FPATH))
    return G



def preprocess_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    # 1. Convertit les coordonnées GPS en mètres
    G = ox.project_graph(G)

    # 2. Fusionne les chaînes de nœuds de degré 2 (routes sans intersection intermédiaire)
    if not G.graph.get("simplified"):
            G = ox.simplify_graph(G)

    # 3. Regroupe les nœuds très proches (< 15 mètres), qui représentent la même intersection réelle
    try:
        G = ox.consolidate_intersections(G, rebuild_graph=True, tolerance=15)
    except Exception:
        pass

    # 4. Ajout de la vitesse max ainsi que le temps pour parcourir l'arête 
    G = ox.add_edge_speeds(G, fallback=50)
    G = ox.add_edge_travel_times(G)

    return G



def save_graph_to_geojson(G: nx.MultiDiGraph, filepath: Path):
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    
    edges_gdf = edges_gdf.to_crs(epsg=4326)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    edges_gdf.to_file(str(filepath), driver="GeoJSON")
    print(f"Graphe sauvegardé au format GeoJSON dans {filepath}")



def quick_plot(
    G: nx.MultiDiGraph,
    fname: str,
    arrows: bool,
    dpi: int
):
    print("Génération de la figure…")

    if not arrows:
        fig, ax = ox.plot_graph(
            G,
            node_size=0,
            edge_linewidth=0.4,
            edge_color="gray",
            bgcolor="white",
            show=False,
            close=False
        )
    else:
        pos = {n: (data["x"], data["y"]) for n, data in G.nodes(data=True)}
        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_nodes(G, pos, node_size=1, ax=ax)
        nx.draw_networkx_edges(
            G, pos,
            edge_color="lightblue",
            width=0.5,
            arrows=False,
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos,
            edge_color="lightgray",
            width=0.3,
            arrows=True,
            arrowstyle="->",
            arrowsize=10,
            ax=ax
        )

        ax.set_title("Réseau routier de Montréal (orienté)")
        ax.axis("off")
        plt.tight_layout()
        # plt.savefig("graphe_montreal_oriente.png", dpi=dpi)

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    print(f"Figure enregistrée dans {fname} (dpi={dpi})")


# ----------------------------------------------------------------------
def main(build_only: bool):
    G_raw = load_or_download_graph()
    print(f"Brut : {G_raw.number_of_nodes():,} noeuds, {G_raw.number_of_edges():,} arêtes")

    G = preprocess_graph(G_raw)
    print(f"Nettoyé : {G.number_of_nodes():,} noeuds et {G.number_of_edges():,} arêtes")
    
    save_graph_to_geojson(G, Path("assets/graph/geojson/graphe_montreal.geojson"))

    if not build_only:
        dpi=1000
        quick_plot(G, fname="assets/graph/img/graphe_montreal.png", arrows=False, dpi=dpi)
        quick_plot(G, fname="assets/graph/img/graphe_montreal_oriente.png", arrows=True, dpi=dpi)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nofig", action="store_true")
    args = parser.parse_args()

    main(build_only=args.nofig)
