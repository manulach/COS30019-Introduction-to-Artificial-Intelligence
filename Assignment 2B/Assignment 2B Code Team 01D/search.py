"""
COS30019 Assignment 2A: Tree-Based Search
Route Finding Problem
Team ID: 01D
Team Members: Saniru, Haresh, Manula


Algorithms implemented:
    DFS  - Depth First Search (uninformed)
    BFS  - Breadth First Search (uninformed)
    GBFS - Greedy Best First Search (informed)
    AS   - A* Search (informed)
    CUS1 - Uniform Cost Search (uninformed custom)
    CUS2 - Iterative Deepening A* / IDA* (informed custom)

Usage:
    python search.py <filename> <method>

Output format (goal reachable):
    filename method
    goal number_of_nodes
    path

Output format (no goal reachable):
    filename method
    No goal is reachable; number_of_nodes
"""

import sys
import math
import heapq
from collections import deque


# FILE PARSING

def parse_file(filename):
    """
    Break down the problem file into nodes, edges, origin and destinations.

    Returns:
    nodes  : dict  {node_id: (x, y)}
    edges  : dict  {from_node: [(to_node, cost), ...]}
    origin : int
    destinations: list[int]
    """
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    section = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == 'Nodes:':
                section = 'nodes'
            elif line == 'Edges:':
                section = 'edges'
            elif line == 'Origin:':
                section = 'origin'
            elif line == 'Destinations:':
                section = 'destinations'

            elif section == 'nodes':
                # Format: "1: (4,1)"
                node_id, coords = line.split(':')
                node_id = int(node_id.strip())
                x, y = coords.strip().strip('()').split(',')
                nodes[node_id] = (int(x), int(y))

            elif section == 'edges':
                # Format: "(2,1): 4"
                edge_part, cost_part = line.split(':')
                from_node, to_node = edge_part.strip().strip('()').split(',')
                from_node, to_node = int(from_node), int(to_node)
                cost = int(cost_part.strip())
                if from_node not in edges:
                    edges[from_node] = []
                edges[from_node].append((to_node, cost))

            elif section == 'origin':
                origin = int(line)

            elif section == 'destinations':
                destinations = [int(d.strip()) for d in line.split(';')]

    return nodes, edges, origin, destinations


# HELPERS

def euclidean(node, nodes):
    return nodes[node]


def heuristic(node, destinations, nodes):
    """
    Straight-line (Euclidean) distance from node to the nearest destination.
    Admissible heuristic for informed search methods.
    """
    x1, y1 = nodes[node]
    return min(
        math.sqrt((nodes[d][0] - x1) ** 2 + (nodes[d][1] - y1) ** 2)
        for d in destinations
    )


def get_neighbors_sorted(node, edges):
    """
    Return neighbours of a node sorted by node id (ascending).
    Satisfies the tie-breaking rule: expand smaller-numbered nodes first
    when all else is equal.
    """
    if node not in edges:
        return []
    return sorted(edges[node], key=lambda pair: pair[0])


def format_path(path):
    """Format a list of node ids as '2 -> 3 -> 5'."""
    return ' -> '.join(str(n) for n in path)


# ALGORITHM 1 – DFS (Depth First Search, uninformed)

def dfs(nodes, edges, origin, destinations):
    """
    Depth-First Search with Explicit Stack (Last-In-First-Out).
    Tie-Breaking Rule: Neighbours are added to the stack in descending order.
    The smallest-numbered node is always at the top of the stack.
    nodes_created: Number of nodes generated in the search process.
    """
    dest_set = set(destinations)
    stack = [(origin, [origin])]   # (current_node, path_so_far)
    visited = set()
    nodes_created = 1              # origin is the first created node

    while stack:
        current, path = stack.pop()

        if current in visited:
            continue
        visited.add(current)

        if current in dest_set:
            return current, nodes_created, path

        neighbours = get_neighbors_sorted(current, edges)
        # Push in REVERSE order = smallest node id ends up on top of stack
        for neighbour, _ in reversed(neighbours):
            if neighbour not in visited:
                nodes_created += 1
                stack.append((neighbour, path + [neighbour]))

    return None, nodes_created, []


# ALGORITHM 2 – BFS (Breadth First Search, uninformed)

def bfs(nodes, edges, origin, destinations):
    """
    BFS Algorithm employing FIFO queue
    Marking nodes as visited once put in the queue to avoid repetition.
    Tie breaking rule states neighbors must be placed into the queue in increasing order of node id.
    nodes_created: total number of nodes placed in the queue.
    """
    dest_set = set(destinations)
    queue = deque([(origin, [origin])])
    visited = {origin}
    nodes_created = 1

    while queue:
        current, path = queue.popleft()

        if current in dest_set:
            return current, nodes_created, path

        for neighbour, _ in get_neighbors_sorted(current, edges):
            if neighbour not in visited:
                visited.add(neighbour)
                nodes_created += 1
                queue.append((neighbour, path + [neighbour]))

    return None, nodes_created, []


# ALGORITHM 3 – GBFS (Greedy Best First Search, informed)

def gbfs(nodes, edges, origin, destinations):
    """
    Greedy Best-First Search with Min-Heap.
    Priority: h(n), distance to nearest destination.
    Tiebreaker: (h, id, num_inserts)
    = smallest ID wins, followed by earliest insert.
    nodes_created: tracks all nodes inserted into heap.
    """
    dest_set = set(destinations)
    counter = 0
    h0 = heuristic(origin, destinations, nodes)
    # heap entry: (h, node_id, counter, path)
    heap = [(h0, origin, counter, [origin])]
    visited = set()
    nodes_created = 1

    while heap:
        h, current, _, path = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        if current in dest_set:
            return current, nodes_created, path

        for neighbour, _ in get_neighbors_sorted(current, edges):
            if neighbour not in visited:
                counter += 1
                nodes_created += 1
                h_n = heuristic(neighbour, destinations, nodes)
                heapq.heappush(heap, (h_n, neighbour, counter, path + [neighbour]))

    return None, nodes_created, []


# ALGORITHM 4 – AS (A* Search, informed)

def astar(nodes, edges, origin, destinations):
    """
    A* Search utilizing min-heap.
    Priority function: f(n) = g(n) + h(n),
    where g(n) is the cost so far,
    h(n) is an admissible heuristic for the straight-line distance to closest goal.
    Ties broken by: (f, node_id, insertion_count)
    nodes_created: keeps track of all nodes inserted into
    """
    dest_set = set(destinations)
    counter = 0
    h0 = heuristic(origin, destinations, nodes)
    # heap entry: (f, g, node_id, counter, path)
    heap = [(h0, 0, origin, counter, [origin])]
    best_g = {}          # best known g-cost to each node
    nodes_created = 1

    while heap:
        f, g, current, _, path = heapq.heappop(heap)

        # Skip if we have already found a cheaper path to this node
        if current in best_g and best_g[current] <= g:
            continue
        best_g[current] = g

        if current in dest_set:
            return current, nodes_created, path

        for neighbour, edge_cost in get_neighbors_sorted(current, edges):
            new_g = g + edge_cost
            if neighbour not in best_g or best_g[neighbour] > new_g:
                counter += 1
                nodes_created += 1
                h_n = heuristic(neighbour, destinations, nodes)
                heapq.heappush(
                    heap,
                    (new_g + h_n, new_g, neighbour, counter, path + [neighbour])
                )

    return None, nodes_created, []


# ALGORITHM 5 – CUS1: Uniform Cost Search (uninformed custom)

def ucs(nodes, edges, origin, destinations):
    """
    Uniform Cost Search (CUS1) – Uninformed Custom Algorithm
    Expansion of nodes based on the total cost incurred while going from the start state,
    without any heuristic.
    Selected because it is an extension of BFS for dealing with weighted graphs.
    Priority: g(n) (total cost till now)
    Tied nodes: (g, node_id, insertion_counter)
    nodes_created: Counts every node that was added to the priority queue.
    """
    dest_set = set(destinations)
    counter = 0
    heap = [(0, origin, counter, [origin])]   # (g, node_id, counter, path)
    best_g = {}
    nodes_created = 1

    while heap:
        g, current, _, path = heapq.heappop(heap)

        if current in best_g and best_g[current] <= g:
            continue
        best_g[current] = g

        if current in dest_set:
            return current, nodes_created, path

        for neighbour, edge_cost in get_neighbors_sorted(current, edges):
            new_g = g + edge_cost
            if neighbour not in best_g or best_g[neighbour] > new_g:
                counter += 1
                nodes_created += 1
                heapq.heappush(
                    heap,
                    (new_g, neighbour, counter, path + [neighbour])
                )

    return None, nodes_created, []


# ALGORITHM 6 – CUS2: IDA* (Iterative Deepening A*, informed custom)

def ida_star(nodes, edges, origin, destinations):
    """
    Iterative deepening A* (CUS2) – informed custom strategy.
    Space efficiency of DFS and optimal solution of A* algorithm. It uses an increasing cost limit in each step, determined by admissible h(n)
    equal to Euclidean distance to the nearest goal.
    It finds optimal cost solution and takes O(d) space complexity (d = depth).
    nodes_created: number of states generated by the algorithm.
    """
    dest_set = set(destinations)
    nodes_created = [1]   # mutable counter shared across recursive calls

    def search(path, g, bound):
        """
        Returns (t, result):
            t      = minimum f-value that exceeded bound (used as next bound)
            result = solution path if found, else None
        """
        current = path[-1]
        f = g + heuristic(current, destinations, nodes)

        if f > bound:
            return f, None

        if current in dest_set:
            return -1, list(path)

        minimum = float('inf')

        for neighbour, edge_cost in get_neighbors_sorted(current, edges):
            if neighbour not in path:          # avoid cycles in current path
                nodes_created[0] += 1
                path.append(neighbour)
                t, result = search(path, g + edge_cost, bound)
                if result is not None:
                    return -1, result
                if t < minimum:
                    minimum = t
                path.pop()

        return minimum, None

    bound = heuristic(origin, destinations, nodes)
    path = [origin]

    while True:
        t, result = search(path, 0, bound)
        if result is not None:
            return result[-1], nodes_created[0], result
        if t == float('inf'):
            return None, nodes_created[0], []
        bound = t   # raise the threshold and retry


# MAIN

METHODS = {
    'DFS':  dfs,
    'BFS':  bfs,
    'GBFS': gbfs,
    'AS':   astar,
    'CUS1': ucs,
    'CUS2': ida_star,
}


def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method   = sys.argv[2].upper()

    if method not in METHODS:
        print(f"Unknown method '{method}'. Valid methods: {', '.join(METHODS)}")
        sys.exit(1)

    nodes, edges, origin, destinations = parse_file(filename)

    goal, num_nodes, path = METHODS[method](nodes, edges, origin, destinations)

    print(f"{filename} {method}")
    if goal is not None:
        print(f"{goal} {num_nodes}")
        print(format_path(path))
    else:
        print(f"No goal is reachable; {num_nodes}")


if __name__ == '__main__':
    main()
