/*
 * dstar.c: see dstar.h. D* Lite as in Koenig & Likhachev (AAAI 2002), Fig. 4 (the optimised version), searching
 * backward from the goal. One simplification: after a batch of cost changes every affected vertex's rhs is
 * recomputed from all its successors (the paper's per-edge shortcut gives the same rhs; this form is easier to keep
 * right with many edges changing at once).
 *
 * The heuristic is h(start, s) = alpha_h * angle(start, s) with alpha_h <= alpha (the edges' angle weight), and 0 for
 * the goal; angle() obeys the triangle inequality, so h stays consistent for every start, including jumps.
 */
#include "dstar.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { double k1, k2; } key;

struct dstar {
	int n;                    /* songs; vertex n is the goal */
	int d;
	const float *E;
	int *off, *adj;           /* successors (songs only; every song also reaches the goal) */
	float *ang;               /* angle of each successor edge */
	int *poff, *pred;         /* predecessors among songs (the goal's predecessors are all songs) */
	double alpha, alpha_h;
	double *node, *goal;      /* applied costs */
	unsigned char *blocked;
	double *node_new, *goal_new;
	unsigned char *blocked_new;
	int *dirty;               /* songs whose costs changed since the last plan */
	unsigned char *is_dirty;
	int ndirty;
	int all_dirty;            /* alpha changed: every edge */
	double *g, *rhs;
	double *h;                /* h(start, s) for the current start */
	int start, last;
	double km;
	/* the priority queue: a binary heap of vertices with their keys */
	int *heap, *pos, hn;
	key *hk;
	ds_stats st;
};

static double angle(const dstar *D, int a, int b)
{
	const float *x = D->E + (size_t)a * D->d, *y = D->E + (size_t)b * D->d;
	double s = 0;
	for (int k = 0; k < D->d; k++) s += (double)x[k] * y[k];
	if (s > 1) s = 1;
	if (s < -1) s = -1;
	return acos(s);
}

static int kless(key a, key b) { return a.k1 < b.k1 || (a.k1 == b.k1 && a.k2 < b.k2); }

static void hswap(dstar *D, int i, int j)
{
	int vi = D->heap[i], vj = D->heap[j];
	key t = D->hk[i];
	D->heap[i] = vj; D->heap[j] = vi;
	D->hk[i] = D->hk[j]; D->hk[j] = t;
	D->pos[vj] = i; D->pos[vi] = j;
}

static void hup(dstar *D, int i)
{
	while (i > 0) {
		int p = (i - 1) / 2;
		if (!kless(D->hk[i], D->hk[p])) break;
		hswap(D, i, p);
		i = p;
	}
}

static void hdown(dstar *D, int i)
{
	for (;;) {
		int l = 2 * i + 1, r = l + 1, m = i;
		if (l < D->hn && kless(D->hk[l], D->hk[m])) m = l;
		if (r < D->hn && kless(D->hk[r], D->hk[m])) m = r;
		if (m == i) break;
		hswap(D, i, m);
		i = m;
	}
}

static void hset(dstar *D, int v, key k)   /* insert, or update the key */
{
	D->st.heap_ops++;
	int i = D->pos[v];
	if (i < 0) {
		i = D->hn++;
		D->heap[i] = v;
		D->pos[v] = i;
		D->hk[i] = k;
		hup(D, i);
		return;
	}
	key old = D->hk[i];
	D->hk[i] = k;
	if (kless(k, old)) hup(D, i); else hdown(D, i);
}

static void hremove(dstar *D, int v)
{
	int i = D->pos[v];
	if (i < 0) return;
	D->st.heap_ops++;
	int last = --D->hn;
	if (i != last) {
		hswap(D, i, last);
		D->pos[v] = -1;
		hup(D, i);
		hdown(D, i);
	} else D->pos[v] = -1;
}

static key topkey(const dstar *D) { return D->hn ? D->hk[0] : (key){ INFINITY, INFINITY }; }

static double cost_song(const dstar *D, int e /* index into adj */)
{
	const int b = D->adj[e];
	if (D->blocked[b]) return INFINITY;
	return D->alpha * D->ang[e] + D->node[b];
}

static double cost_goal(const dstar *D, int a) { return D->blocked[a] ? INFINITY : D->goal[a]; }

double ds_edge(const dstar *D, int a, int b)
{
	if (b == D->n) return cost_goal(D, a);
	for (int e = D->off[a]; e < D->off[a + 1]; e++) if (D->adj[e] == b) return cost_song(D, e);
	return INFINITY;
}

static key calc(const dstar *D, int s)
{
	const double m = D->g[s] < D->rhs[s] ? D->g[s] : D->rhs[s];
	return (key){ m + D->h[s] + D->km, m };
}

static void update_vertex(dstar *D, int u)
{
	D->st.updated++;
	if (D->g[u] != D->rhs[u]) hset(D, u, calc(D, u));
	else if (D->pos[u] >= 0) hremove(D, u);
}

/* rhs(u) = min over successors of c(u, s') + g(s') */
static double best_rhs(const dstar *D, int u)
{
	double b = cost_goal(D, u) + D->g[D->n];
	for (int e = D->off[u]; e < D->off[u + 1]; e++) {
		const double c = cost_song(D, e) + D->g[D->adj[e]];
		if (c < b) b = c;
	}
	return b;
}

static void compute(dstar *D)
{
	const int s = D->start;
	while (kless(topkey(D), calc(D, s)) || D->rhs[s] > D->g[s]) {
		const int u = D->heap[0];
		const key kold = D->hk[0], knew = calc(D, u);
		D->st.expanded++;
		if (kless(kold, knew)) { hset(D, u, knew); continue; }
		if (D->g[u] > D->rhs[u]) {
			D->g[u] = D->rhs[u];
			hremove(D, u);
			if (u == D->n) {
				for (int p = 0; p < D->n; p++) {
					const double c = cost_goal(D, p) + D->g[u];
					if (c < D->rhs[p]) D->rhs[p] = c;
					update_vertex(D, p);
				}
			} else {
				for (int q = D->poff[u]; q < D->poff[u + 1]; q++) {
					const int p = D->pred[q];
					D->rhs[p] = best_rhs(D, p);
					update_vertex(D, p);
				}
			}
		} else {
			D->g[u] = INFINITY;
			if (u != D->n) { D->rhs[u] = best_rhs(D, u); }
			update_vertex(D, u);
			if (u == D->n) {
				for (int p = 0; p < D->n; p++) { D->rhs[p] = best_rhs(D, p); update_vertex(D, p); }
			} else {
				for (int q = D->poff[u]; q < D->poff[u + 1]; q++) {
					const int p = D->pred[q];
					D->rhs[p] = best_rhs(D, p);
					update_vertex(D, p);
				}
			}
		}
	}
}

dstar *ds_new(int n, const float *E, int d, const int *off, const int *adj, double alpha)
{
	dstar *D = calloc(1, sizeof *D);
	D->n = n;
	D->d = d;
	D->E = E;
	const int m = off[n];
	D->off = malloc((size_t)(n + 1) * sizeof(int));
	D->adj = malloc((size_t)m * sizeof(int));
	D->ang = malloc((size_t)m * sizeof(float));
	memcpy(D->off, off, (size_t)(n + 1) * sizeof(int));
	memcpy(D->adj, adj, (size_t)m * sizeof(int));
	for (int a = 0; a < n; a++)
		for (int e = off[a]; e < off[a + 1]; e++) D->ang[e] = (float)angle(D, a, adj[e]);
	/* predecessors: the reverse of adj */
	D->poff = calloc((size_t)(n + 1), sizeof(int));
	D->pred = malloc((size_t)m * sizeof(int));
	for (int e = 0; e < m; e++) D->poff[adj[e] + 1]++;
	for (int i = 0; i < n; i++) D->poff[i + 1] += D->poff[i];
	int *fill = calloc((size_t)n, sizeof(int));
	for (int a = 0; a < n; a++)
		for (int e = off[a]; e < off[a + 1]; e++) { const int b = adj[e]; D->pred[D->poff[b] + fill[b]++] = a; }
	free(fill);
	D->alpha = alpha;
	D->alpha_h = alpha;
	const size_t V = (size_t)n + 1;
	D->node = calloc(V, sizeof(double)); D->goal = calloc(V, sizeof(double)); D->blocked = calloc(V, 1);
	D->node_new = calloc(V, sizeof(double)); D->goal_new = calloc(V, sizeof(double)); D->blocked_new = calloc(V, 1);
	D->dirty = malloc(V * sizeof(int)); D->is_dirty = calloc(V, 1);
	D->g = malloc(V * sizeof(double)); D->rhs = malloc(V * sizeof(double)); D->h = calloc(V, sizeof(double));
	D->heap = malloc(V * sizeof(int)); D->pos = malloc(V * sizeof(int)); D->hk = malloc(V * sizeof(key));
	for (size_t i = 0; i < V; i++) { D->g[i] = D->rhs[i] = INFINITY; D->pos[i] = -1; }
	D->start = D->last = -1;
	return D;
}

void ds_free(dstar *D)
{
	if (!D) return;
	free(D->off); free(D->adj); free(D->ang); free(D->poff); free(D->pred);
	free(D->node); free(D->goal); free(D->blocked); free(D->node_new); free(D->goal_new); free(D->blocked_new);
	free(D->dirty); free(D->is_dirty); free(D->g); free(D->rhs); free(D->h); free(D->heap); free(D->pos); free(D->hk);
	free(D);
}

static void mark(dstar *D, int b) { if (!D->is_dirty[b]) { D->is_dirty[b] = 1; D->dirty[D->ndirty++] = b; } }

void ds_set_node(dstar *D, int b, double c) { if (c < 0) c = 0; if (D->node_new[b] != c) { D->node_new[b] = c; mark(D, b); } }
void ds_set_goal(dstar *D, int b, double c) { if (c < 0) c = 0; if (D->goal_new[b] != c) { D->goal_new[b] = c; mark(D, b); } }
void ds_set_blocked(dstar *D, int b, int on) { on = !!on; if (D->blocked_new[b] != on) { D->blocked_new[b] = (unsigned char)on; mark(D, b); } }
double ds_node(const dstar *D, int b) { return D->node_new[b]; }
double ds_goal(const dstar *D, int b) { return D->goal_new[b]; }
int ds_blocked(const dstar *D, int b) { return D->blocked_new[b]; }

void ds_set_alpha(dstar *D, double alpha)
{
	if (alpha < D->alpha_h) alpha = D->alpha_h;   /* the heuristic's weight is the floor */
	if (alpha != D->alpha) { D->alpha = alpha; D->all_dirty = 1; }
}

void ds_move(dstar *D, int s)
{
	if (D->start < 0) {   /* first placement: Initialize() */
		D->start = D->last = s;
		D->km = 0;
		for (int v = 0; v < D->n; v++) D->h[v] = D->alpha_h * angle(D, s, v);
		D->h[D->n] = 0;
		D->rhs[D->n] = 0;
		hset(D, D->n, calc(D, D->n));
		return;
	}
	if (s == D->start) return;
	D->km += D->alpha_h * angle(D, D->last, s);
	D->last = s;
	D->start = s;
	for (int v = 0; v < D->n; v++) D->h[v] = D->alpha_h * angle(D, s, v);
}

static void apply_changes(dstar *D)
{
	if (!D->ndirty && !D->all_dirty) return;
	/* commit the new costs, then repair every vertex whose outgoing edges changed */
	for (int i = 0; i < D->ndirty; i++) {
		const int b = D->dirty[i];
		D->node[b] = D->node_new[b];
		D->goal[b] = D->goal_new[b];
		D->blocked[b] = D->blocked_new[b];
	}
	unsigned char *touch = calloc((size_t)D->n, 1);
	if (D->all_dirty) memset(touch, 1, (size_t)D->n);
	for (int i = 0; i < D->ndirty; i++) {
		const int b = D->dirty[i];
		touch[b] = 1;   /* its goal edge */
		for (int q = D->poff[b]; q < D->poff[b + 1]; q++) touch[D->pred[q]] = 1;   /* edges into b */
		D->is_dirty[b] = 0;
	}
	for (int u = 0; u < D->n; u++)
		if (touch[u]) { D->rhs[u] = best_rhs(D, u); update_vertex(D, u); }
	free(touch);
	D->ndirty = 0;
	D->all_dirty = 0;
}

int ds_plan(dstar *D, int *path, int max, double *cost)
{
	apply_changes(D);
	compute(D);
	*cost = D->g[D->start] < D->rhs[D->start] ? D->g[D->start] : D->rhs[D->start];
	if (!isfinite(*cost)) return 0;
	/* follow argmin c(s, s') + g(s') from the start to the goal */
	int n = 0, s = D->start;
	while (n < max) {
		int best = -1;
		double bv = cost_goal(D, s) + D->g[D->n];
		for (int e = D->off[s]; e < D->off[s + 1]; e++) {
			const double c = cost_song(D, e) + D->g[D->adj[e]];
			if (c < bv) { bv = c; best = D->adj[e]; }
		}
		if (best < 0 || !isfinite(bv)) break;   /* the goal edge is best: the plan ends here */
		path[n++] = best;
		s = best;
	}
	return n;
}

void ds_get_stats(const dstar *D, ds_stats *s) { *s = D->st; }

double ds_dijkstra(const dstar *D, long *expanded)
{
	const int V = D->n + 1;
	double *dist = malloc((size_t)V * sizeof(double));
	unsigned char *done = calloc((size_t)V, 1);
	const size_t cap = (size_t)D->off[D->n] + 2 * (size_t)V + 16;   /* one push per improving relaxation at most */
	int *heap = malloc(cap * sizeof(int));
	double *hd = malloc(cap * sizeof(double));
	int hn = 0;
	long ex = 0;
	for (int i = 0; i < V; i++) dist[i] = INFINITY;
	dist[D->start] = 0;
	/* a lazy binary heap: stale entries are skipped */
#define PUSH(v, x) do { int i_ = hn++; heap[i_] = v; hd[i_] = x; while (i_ > 0 && hd[(i_ - 1) / 2] > hd[i_]) { int p_ = (i_ - 1) / 2; int tv = heap[p_]; double tx = hd[p_]; heap[p_] = heap[i_]; hd[p_] = hd[i_]; heap[i_] = tv; hd[i_] = tx; i_ = p_; } } while (0)
	PUSH(D->start, 0.0);
	while (hn) {
		const int u = heap[0];
		const double du = hd[0];
		hn--;
		heap[0] = heap[hn]; hd[0] = hd[hn];
		for (int i = 0;;) {
			int l = 2 * i + 1, r = l + 1, m = i;
			if (l < hn && hd[l] < hd[m]) m = l;
			if (r < hn && hd[r] < hd[m]) m = r;
			if (m == i) break;
			int tv = heap[m]; double tx = hd[m]; heap[m] = heap[i]; hd[m] = hd[i]; heap[i] = tv; hd[i] = tx; i = m;
		}
		if (done[u] || du > dist[u]) continue;
		done[u] = 1;
		ex++;
		if (u == D->n) break;
		const double cg = cost_goal(D, u);
		if (du + cg < dist[D->n]) { dist[D->n] = du + cg; PUSH(D->n, dist[D->n]); }
		for (int e = D->off[u]; e < D->off[u + 1]; e++) {
			const int b = D->adj[e];
			const double c = du + cost_song(D, e);
			if (c < dist[b]) { dist[b] = c; PUSH(b, c); }
		}
	}
#undef PUSH
	const double r = dist[D->n];
	free(dist); free(done); free(heap); free(hd);
	if (expanded) *expanded = ex;
	return r;
}
