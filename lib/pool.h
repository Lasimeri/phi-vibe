/* Copied from Lasimeri/phi-vision lib/pool.h at 8704e46 (same author, MIT); pool_set_spin added here. */
/* pool.h: a persistent thread pool. pool_run hands out tasks 0..n-1 through an atomic counter; the caller works too and
 * returns when every task is done and every worker has left the run. Workers spin briefly between runs, then sleep. */
#ifndef POOL_H
#define POOL_H

typedef void (*pool_fn)(void *arg, int task);
typedef struct pool pool;

/* threads includes the caller; cpus (may be NULL) pins worker i to cpus[i] (the caller is not moved) */
pool *pool_open(int threads, const int *cpus);
void pool_run(pool *p, pool_fn fn, void *arg, int ntasks);
int pool_threads(const pool *p);
/* how long idle workers spin (polling with a pause) before they sleep; 2 ms by default. Waking sleepers costs
 * milliseconds on the card, and pool_run waits for every worker. */
void pool_set_spin(pool *p, double seconds);
void pool_close(pool *p);

#endif
