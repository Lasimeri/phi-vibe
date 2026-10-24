/*
 * dstar.h: D* Lite (Koenig & Likhachev, AAAI 2002, the optimised version, Fig. 4) over the song graph: the
 * navigation mesh is the vibe-neighbour graph, the agent stands on the song now playing, and the target is a virtual
 * goal node reached from every song. Searching backward from the goal lets the start move (one song per step) and
 * edge costs change (the network learning, the target vibe drifting, songs blocked after they play) while the search
 * repairs only what the change touches.
 *
 * Costs:  song a -> song b (b a graph neighbour of a):  alpha * angle(a, b) + node(b)   (infinite when b is blocked)
 *         song b -> goal:                                goal(b)                         (infinite when b is blocked)
 * angle() is the angle between the unit vibe vectors: a metric, so h(start, s) = alpha * angle(start, s) is a
 * consistent heuristic (every other cost term is >= 0).
 */
#ifndef DSTAR_H
#define DSTAR_H

#include <stdint.h>

typedef struct dstar dstar;

typedef struct {
	long expanded;        /* vertices popped by ComputeShortestPath */
	long updated;         /* UpdateVertex calls */
	long heap_ops;
} ds_stats;

/* n songs with unit vectors E (n x d); adjacency in CSR form: song a's neighbours are adj[off[a] .. off[a+1]-1] */
/* alpha: the angle weight now and its floor for good (the heuristic uses the floor) */
dstar *ds_new(int n, const float *E, int d, const int *off, const int *adj, double alpha);
/* a new angle weight, clamped to the floor; every edge changes */
void ds_set_alpha(dstar *D, double alpha);
void ds_free(dstar *D);

/* queue cost changes; applied by the next ds_plan */
void ds_set_node(dstar *D, int b, double cost);
void ds_set_goal(dstar *D, int b, double cost);
void ds_set_blocked(dstar *D, int b, int blocked);
double ds_node(const dstar *D, int b);
double ds_goal(const dstar *D, int b);
int ds_blocked(const dstar *D, int b);

/* moves the agent to song s (the first call places it) */
void ds_move(dstar *D, int s);

/* repairs the search and writes the planned songs after the start (at most max) into path; returns how many, 0 when
 * the goal is unreachable. *cost gets the plan's total cost. */
int ds_plan(dstar *D, int *path, int max, double *cost);

void ds_get_stats(const dstar *D, ds_stats *s);

/* reference: plain Dijkstra from the start over the same costs; the cost to the goal (for the gate) */
double ds_dijkstra(const dstar *D, long *expanded);

/* the edge cost a -> b as the planner sees it (b == n: the goal) */
double ds_edge(const dstar *D, int a, int b);

#endif
