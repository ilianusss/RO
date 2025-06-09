"""
Génère une animation du circuit eulérien sur le réseau routier de Montréal
"""

import os
import networkx as nx
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.collections import LineCollection
from pathlib import Path


def load_data():
    """Charge le graphe et le circuit eulérien"""
    # Charger le graphe
    G = ox.load_graphml("assets/graph/graphml/graphe_montreal.graphml")
    
    # Charger le circuit
    circuit_path = Path("assets/graph/geojson/graphe_montreal_eulerian.geojson")
    if not circuit_path.exists():
        raise FileNotFoundError(f"Le fichier {circuit_path} n'existe pas. Exécutez d'abord eulerian.py")
    
    circuit_gdf = gpd.read_file(circuit_path)
    
    # S'assurer que le CRS est défini
    if circuit_gdf.crs is None:
        # Utiliser le CRS du graphe si disponible
        if 'crs' in G.graph:
            circuit_gdf.set_crs(G.graph['crs'], inplace=True)
        else:
            # Par défaut, utiliser WGS84
            circuit_gdf.set_crs("EPSG:4326", inplace=True)
    
    print(f"CRS du circuit: {circuit_gdf.crs}")
    
    return G, circuit_gdf


def prepare_background(G, fig, ax, bgcolor="white"):
    """Dessine le fond de carte (réseau routier)"""
    # Projeter le graphe pour avoir les coordonnées x,y
    G_proj = ox.project_graph(G)
    
    # Extraire les nœuds et les arêtes
    nodes, edges = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)
    
    # Dessiner les arêtes en gris clair
    ax.set_facecolor(bgcolor)
    edges.plot(ax=ax, color="lightgray", linewidth=0.5, alpha=0.7, zorder=1)
    
    # Ajuster les limites
    margin = 0.02
    west, south, east, north = edges.total_bounds
    ax.set_xlim([west - margin, east + margin])
    ax.set_ylim([south - margin, north + margin])
    
    # Supprimer les axes
    ax.set_axis_off()
    
    # Forcer le rafraîchissement de la figure
    plt.draw()
    
    return nodes, edges


def animate_circuit(G, circuit_gdf, output_path="assets/animations", 
                   fps=30, duration=20, dpi=300, format="mp4", debug=True):
    """
    Crée une animation du circuit eulérien
    
    Args:
        G: Le graphe du réseau routier
        circuit_gdf: GeoDataFrame du circuit eulérien
        output_path: Dossier de sortie pour l'animation
        fps: Images par seconde
        duration: Durée de l'animation en secondes
        dpi: Résolution de l'animation
        format: Format de sortie (mp4, gif)
    """
    # Créer le dossier de sortie s'il n'existe pas
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Dessiner le fond de carte
    nodes, edges = prepare_background(G, fig, ax)
    
    # Projeter le circuit pour qu'il corresponde au fond de carte
    print("Projection du circuit pour correspondre au fond de carte...")
    # Assurons-nous que le circuit est dans le même CRS que le graphe projeté
    G_proj = ox.project_graph(G)
    if 'crs' in G_proj.graph:
        target_crs = G_proj.graph['crs']
        print(f"Projection vers CRS: {target_crs}")
        circuit_gdf = circuit_gdf.to_crs(target_crs)
    
    # Préparer les segments du circuit pour l'animation
    segments = []
    for _, row in circuit_gdf.iterrows():
        geom = row['geometry']
        if hasattr(geom, 'coords'):
            # Convertir les coordonnées en liste de tuples (x, y)
            coords = list(geom.coords)
            if coords:  # S'assurer que la géométrie n'est pas vide
                segments.append(coords)
    
    print(f"Nombre de segments chargés: {len(segments)}")
    if not segments:
        raise ValueError("Aucun segment valide trouvé dans le circuit")
    
    # Calculer le nombre total de points
    total_points = sum(len(segment) for segment in segments)
    
    # Calculer combien de segments à afficher par frame
    total_frames = fps * duration
    points_per_frame = max(1, total_points // total_frames)
    
    # Collection de lignes pour l'animation
    line_collection = LineCollection([], color='red', linewidth=2.5, zorder=10, alpha=0.8)
    ax.add_collection(line_collection)
    
    # Pour débogage, afficher quelques segments pour vérifier qu'ils sont visibles
    if debug and segments:
        print("Affichage de segments de débogage...")
        debug_segments = segments[:min(10, len(segments))]
        debug_collection = LineCollection(debug_segments, color='blue', linewidth=3, zorder=11)
        ax.add_collection(debug_collection)
        plt.draw()
    
    # Titre
    title = ax.set_title("Circuit Eulérien de Montréal", fontsize=15)
    
    # Fonction d'initialisation pour l'animation
    def init():
        line_collection.set_segments([])
        return line_collection,
    
    # Fonction d'animation
    def update(frame):
        # Pour le premier frame, imprimer des infos de débogage
        if frame == 0 and debug:
            print(f"Total frames: {total_frames}, points par frame: {points_per_frame}")
            print(f"Total points: {total_points}, segments: {len(segments)}")
            if segments:
                print(f"Premier segment: {segments[0][:2]}... (longueur: {len(segments[0])})")
        
        # Calculer jusqu'à quel segment on doit afficher
        current_points = frame * points_per_frame
        
        # Collecter les segments à afficher
        displayed_segments = []
        points_count = 0
        
        for segment in segments:
            if points_count + len(segment) <= current_points:
                # Ajouter le segment entier
                displayed_segments.append(segment)
                points_count += len(segment)
            else:
                # Ajouter une partie du segment
                points_to_add = current_points - points_count
                if points_to_add > 0 and points_to_add < len(segment):
                    displayed_segments.append(segment[:points_to_add])
                break
        
        # Mettre à jour la collection de lignes
        line_collection.set_segments(displayed_segments)
        
        # Mettre à jour le titre avec la progression
        progress = min(100, int(100 * current_points / total_points))
        title.set_text(f"Circuit Eulérien de Montréal ({progress}%)")
        
        # Afficher des infos de débogage tous les 30 frames
        if debug and frame % 30 == 0:
            print(f"Frame {frame}/{total_frames}: {len(displayed_segments)} segments affichés ({progress}%)")
        
        # Forcer le rafraîchissement de la figure à chaque frame
        fig.canvas.draw_idle()
        
        return line_collection,
    
    # Créer l'animation
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, blit=False, interval=1000/fps
    )
    
    # Sauvegarder l'animation
    if format.lower() == "mp4":
        try:
            # Utiliser FFMpegWriter avec des paramètres plus robustes
            writer = animation.FFMpegWriter(
                fps=fps, 
                metadata=dict(artist='Eulerian Circuit Animation'),
                bitrate=5000  # Augmenter le bitrate pour une meilleure qualité
            )
            output_file = os.path.join(output_path, "circuit_eulerien.mp4")
            print(f"Sauvegarde de l'animation en MP4 ({total_frames} frames)...")
            anim.save(output_file, writer=writer, dpi=dpi)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde en MP4: {e}")
            print("Tentative de sauvegarde en GIF...")
            output_file = os.path.join(output_path, "circuit_eulerien.gif")
            anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    else:
        output_file = os.path.join(output_path, "circuit_eulerien.gif")
        print(f"Sauvegarde de l'animation en GIF ({total_frames} frames)...")
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    
    print(f"Animation sauvegardée dans {output_file}")
    return output_file


def main():
    print("Chargement des données...")
    G, circuit_gdf = load_data()
    
    # Vérifier que le circuit contient des données
    if len(circuit_gdf) == 0:
        print("ERREUR: Le circuit eulérien est vide!")
        return
    
    print(f"Circuit chargé: {len(circuit_gdf)} segments")
    
    # Afficher quelques informations sur le circuit pour débogage
    print("Exemple de géométries:")
    for i, (_, row) in enumerate(circuit_gdf.head(3).iterrows()):
        geom = row['geometry']
        if hasattr(geom, 'coords'):
            coords = list(geom.coords)
            print(f"  Segment {i}: {len(coords)} points, début: {coords[0]}, fin: {coords[-1]}")
    
    print("Création de l'animation...")
    # Ajuster ces paramètres selon vos besoins
    animate_circuit(
        G, circuit_gdf, 
        output_path="assets/animations",
        fps=30,            # Images par seconde
        duration=30,       # Durée en secondes
        dpi=200,           # Résolution (plus basse pour accélérer le rendu)
        format="mp4",      # Format: mp4 ou gif
        debug=True         # Activer le mode débogage
    )


if __name__ == "__main__":
    main()