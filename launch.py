import sys
import subprocess
import time
import argparse
from datetime import datetime

# Log function
def log(message):
    print(f"[{datetime.now()}] {message}")



log("STARTING PIPELINE")
start_time = time.time()


# Cartography
log("[START] CARTOGRAPHY EXTRACTION")
task_start = time.time()
subprocess.run([sys.executable, "scripts/graph.py"])
#subprocess.run([sys.executable, "scripts/graph.py", "--nofig"])
task_end = time.time()
log(f"[DONE] CARTOGRAPHY EXTRACTION in {(task_end - task_start):.2f} seconds")

# Eulerian
log("[START] EULERIAN CIRCUIT CALCULATION")
task_start = time.time()
subprocess.run([sys.executable, "scripts/eulerian.py"])
task_end = time.time()
log(f"[DONE] EULERIAN CIRCUIT CALCULATION in {(task_end - task_start):.2f} seconds")

# Animation
log("[START] ANIMATION GENERATION")
task_start = time.time()
subprocess.run([sys.executable, "scripts/animate.py"])
task_end = time.time()
log(f"[DONE] ANIMATION GENERATION in {(task_end - task_start):.2f} seconds")


# Drone
log("[START] DRONE SETUP CALCULATION")
task_start = time.time()
subprocess.run([sys.executable, "scripts/drone.py"])
task_end = time.time()
log(f"[DONE] DRONE SETUP CALCULATION in {(task_end - task_start):.2f} seconds")


# End
end_time = time.time()
log(f"PIPELINE COMPLETED in {(end_time - start_time):.2f} seconds")
