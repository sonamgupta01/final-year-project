#!/usr/bin/env python3
"""
External Dataset Hotspot Detection + Predictive Traffic Split Routing
(FINAL CLEAN VERSION)
"""

import pandas as pd

# =========================================
# LOAD DATA
# =========================================

def load_data():
    df = pd.read_csv("Temp1A.txt", sep=r"\s+", header=None)
    df.columns = ["time", "source", "destination"]
    print("✓ Loaded external dataset:", len(df))
    return df


# =========================================
# HOTSPOT DETECTION
# =========================================

def detect_hotspots(df, top_k=3):
    return df["source"].value_counts().index.tolist()[:top_k]


# =========================================
# NORMAL XY ROUTING
# =========================================

def get_xy_path(src, dst, mesh=8):
    path = [src]

    sr, sc = src // mesh, src % mesh
    dr, dc = dst // mesh, dst % mesh

    while sr < dr:
        src += mesh
        sr += 1
        path.append(src)

    while sr > dr:
        src -= mesh
        sr -= 1
        path.append(src)

    while sc < dc:
        src += 1
        sc += 1
        path.append(src)

    while sc > dc:
        src -= 1
        sc -= 1
        path.append(src)

    return path


# =========================================
# SAFE ROUTING (NO HOTSPOT, NO LOOP)
# =========================================

def get_xy_path_avoid_hotspot(src, dst, hotspots, mesh=8, max_steps=50):

    path = [src]
    visited = set([src])

    sr, sc = src // mesh, src % mesh
    dr, dc = dst // mesh, dst % mesh

    steps = 0

    while (sr, sc) != (dr, dc):

        steps += 1
        if steps > max_steps:
            break

        moved = False

        # row move
        if sr < dr:
            nxt = src + mesh
            if nxt not in hotspots and nxt not in visited:
                src = nxt
                sr += 1
                path.append(src)
                visited.add(src)
                moved = True

        elif sr > dr:
            nxt = src - mesh
            if nxt not in hotspots and nxt not in visited:
                src = nxt
                sr -= 1
                path.append(src)
                visited.add(src)
                moved = True

        # column move
        if not moved:
            if sc < dc:
                nxt = src + 1
                if nxt not in hotspots and nxt not in visited:
                    src = nxt
                    sc += 1
                    path.append(src)
                    visited.add(src)
                    moved = True

            elif sc > dc:
                nxt = src - 1
                if nxt not in hotspots and nxt not in visited:
                    src = nxt
                    sc -= 1
                    path.append(src)
                    visited.add(src)
                    moved = True

        # fallback
        if not moved:
            for n in get_neighbors(src):
                if n not in hotspots and n not in visited:
                    src = n
                    sr, sc = src // mesh, src % mesh
                    path.append(src)
                    visited.add(src)
                    moved = True
                    break

        if not moved:
            break

    return path


# =========================================
# NEIGHBORS
# =========================================

def get_neighbors(node, mesh=8):
    r, c = node // mesh, node % mesh

    n = []
    if r > 0: n.append(node - mesh)
    if r < mesh-1: n.append(node + mesh)
    if c > 0: n.append(node - 1)
    if c < mesh-1: n.append(node + 1)

    return n


# =========================================
# REROUTING PLAN
# =========================================

def show_rerouting_plan(hotspots):
    print("\n➡️ Rerouting Plan:")

    for node in hotspots:
        neighbors = get_neighbors(node)
        safe = [n for n in neighbors if n not in hotspots]

        if len(safe) >= 2:
            print(f"- Node {node} → {safe[0]} , {safe[1]} (30% each)")
        elif len(safe) == 1:
            print(f"- Node {node} → {safe[0]} (limited)")
        else:
            print(f"- Node {node} → No safe neighbor")


# =========================================
# 🔥 CORE SPLIT ROUTING LOGIC
# =========================================

def show_split_routing(src, dst, hotspots):

    original = get_xy_path(src, dst)

    print(f"\nSource: {src} → Dest: {dst}")
    print("Original Path :", " → ".join(map(str, original)))
    # -------- LATENCY --------
    orig_len = len(original) #addeded
    print(f"Original Path Length : {orig_len} hops") #added

    print("\n🔀 Traffic Split Routing:")

    # find first hotspot in path
    hotspot_index = -1
    for i in range(len(original)):
        if original[i] in hotspots:
            hotspot_index = i
            break

    if hotspot_index == -1:
        print("No rerouting needed")
        return

    hotspot_node = original[hotspot_index]

    # ================================
    # CASE 1: hotspot at source
    # ================================
    if hotspot_index == 0:

        neighbors = get_neighbors(hotspot_node)
        safe_prev = [n for n in neighbors if n not in hotspots]

        if len(safe_prev) == 0:
            print("No safe rerouting possible")
            return

        decision_node = safe_prev[0]

        print(f"\nAt Virtual Source {decision_node} (HOTSPOT node: {hotspot_node}):")

    # ================================
    # CASE 2: hotspot in path
    # ================================
    else:
        decision_node = original[hotspot_index - 1]
        print(f"\nBefore Hotspot {hotspot_node}, rerouting at Node {decision_node}:")

    neighbors = get_neighbors(decision_node)

    safe_neighbors = [
        n for n in neighbors
        if n not in hotspots and n != hotspot_node
    ]

    if len(safe_neighbors) == 0:
        print("No alternate path available")
        return

    elif len(safe_neighbors) == 1:
        n1 = safe_neighbors[0]

        path1 = [decision_node] + get_xy_path_avoid_hotspot(n1, dst, hotspots)

        print(f" ├── {n1} (60%) →", " → ".join(map(str, path1)))
        print(f" └── original (40%) →", " → ".join(map(str, original)))
        return

    # -------- SMART SELECTION --------
    neighbor_score = {}

    for n in safe_neighbors:
        # simple logic: choose node closer to destination
        nr, nc = n // 8, n % 8
        dr, dc = dst // 8, dst % 8

        distance = abs(nr - dr) + abs(nc - dc)
        neighbor_score[n] = distance

    # sort by least distance (better path)
    sorted_neighbors = sorted(neighbor_score, key=neighbor_score.get)

    n1 = sorted_neighbors[0]
    n2 = sorted_neighbors[1] if len(sorted_neighbors) > 1 else sorted_neighbors[0]

    print(f"\nChosen Neighbors (Smart Selection): {n1}, {n2}")   #added



    path1 = [decision_node] + get_xy_path_avoid_hotspot(n1, dst, hotspots)
    path2 = [decision_node] + get_xy_path_avoid_hotspot(n2, dst, hotspots)

    print(f" ├── {n1} (30%) →", " → ".join(map(str, path1)))
    print(f" ├── {n2} (30%) →", " → ".join(map(str, path2)))
    print(f" └── original (40%) →", " → ".join(map(str, original)))

    alt1_len = len(path1)  #added
    alt2_len = len(path2)  #added

    print(f"   ↳ Path1 Length : {alt1_len} hops")   #added
    print(f"   ↳ Path2 Length : {alt2_len} hops")   #added

    # -------- SUCCESS CHECK --------
    success1 = "YES" if path1[-1] == dst else "NO"  #added
    success2 = "YES" if path2[-1] == dst else "NO"   #added

    print(f"   ✔ Path1 Reached Dest? : {success1}")   #added
    print(f"   ✔ Path2 Reached Dest? : {success2}")    #added

    # -------- CONGESTION REDUCTION --------
    best_len = min(alt1_len, alt2_len)
    reduction = ((orig_len - best_len) / orig_len) * 100
    print(f"   🔻 Congestion Reduced by : {reduction:.2f}%")

# =========================================
# SHOW PACKETS
# =========================================

def show_packet_examples(df, hotspots, max_per_node=3):

    print("\n📦 Packet Routing Details (Per Hotspot Node):")

    for node in hotspots:

        print("\n" + "="*60)
        print(f"🔥 Hotspot Node: {node}")
        print("="*60)

        packets = df[df["source"] == node]

        if len(packets) == 0:
            print("No packets found")
            continue

        count = 0

        for _, row in packets.iterrows():

            src = int(row["source"])
            dst = int(row["destination"])

            show_split_routing(src, dst, hotspots)

            count += 1

            if count >= max_per_node:
                print(f"\n... showing first {max_per_node} packets only")
                break


# =========================================
# MAIN
# =========================================

def run():

    df = load_data()

    hotspots = detect_hotspots(df)

    print("\nDetected Hotspot Nodes:", hotspots)

    show_rerouting_plan(hotspots)

    show_packet_examples(df, hotspots)


if __name__ == "__main__":
    run()