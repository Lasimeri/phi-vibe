/* Copied from Lasimeri/phi-vision lib/pool.c at 8704e46 (same author, MIT); changed here: idle workers spin for a set
 * time (pool_set_spin) with the KNC DELAY instruction between polls, instead of a fixed poll count. */
/* pool.c: see pool.h. */
#define _GNU_SOURCE
#include "pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>

#include <time.h>

#define SPIN_CHECK 1024   /* polls between clock reads while spinning */

/* one poll's pause: on the card DELAY (KNC's replacement for PAUSE) gives the core's other threads the issue slots */
static inline void relax(void)
{
#ifdef __KNC__
	__asm__ volatile("delay %0" : : "r"(16u));
#endif
}

static double mono(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

struct pool {
	int nworkers;
	pthread_t *tid;
	pthread_mutex_t m;
	pthread_cond_t c;
	int sleepers;
	double spin_s;   /* how long an idle worker spins before it sleeps */
	int quit;
	/* the current run */
	pool_fn fn;
	void *arg;
	int ntasks;
	int gen;        /* raised by pool_run after fn/arg/ntasks are set */
	int next;       /* the next task to hand out */
	int done;       /* tasks finished */
	int arrived;    /* workers that finished their part of this run */
};

static void work(pool *p, pool_fn fn, void *arg, int ntasks)
{
	int t;
	while ((t = __atomic_fetch_add(&p->next, 1, __ATOMIC_ACQ_REL)) < ntasks) {
		fn(arg, t);
		__atomic_fetch_add(&p->done, 1, __ATOMIC_RELEASE);
	}
}

typedef struct { pool *p; int cpu; } start;

static void *worker(void *v)
{
	start *s = v;
	pool *p = s->p;
	if (s->cpu >= 0) {
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(s->cpu, &set);
		pthread_setaffinity_np(pthread_self(), sizeof set, &set);
	}
	free(s);
	int seen = 0;
	for (;;) {
		int g;
		long spins = 0;
		double t0 = 0;
		while ((g = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE)) == seen && !__atomic_load_n(&p->quit, __ATOMIC_ACQUIRE)) {
			relax();
			if (++spins % SPIN_CHECK)
				continue;
			if (spins == SPIN_CHECK) t0 = mono();
			if (mono() - t0 < p->spin_s)
				continue;
			pthread_mutex_lock(&p->m);
			p->sleepers++;
			while (__atomic_load_n(&p->gen, __ATOMIC_ACQUIRE) == seen && !p->quit)
				pthread_cond_wait(&p->c, &p->m);
			p->sleepers--;
			pthread_mutex_unlock(&p->m);
			spins = 0;
		}
		if (__atomic_load_n(&p->quit, __ATOMIC_ACQUIRE))
			return NULL;
		seen = g;
		work(p, p->fn, p->arg, p->ntasks);
		__atomic_fetch_add(&p->arrived, 1, __ATOMIC_RELEASE);
	}
}

pool *pool_open(int threads, const int *cpus)
{
	pool *p = calloc(1, sizeof *p);
	if (!p) return NULL;
	p->nworkers = threads > 1 ? threads - 1 : 0;
	p->spin_s = 0.002;
	p->tid = calloc((size_t)(p->nworkers ? p->nworkers : 1), sizeof *p->tid);
	pthread_mutex_init(&p->m, NULL);
	pthread_cond_init(&p->c, NULL);
	for (int i = 0; i < p->nworkers; i++) {
		start *s = malloc(sizeof *s);
		s->p = p;
		s->cpu = cpus ? cpus[i] : -1;
		pthread_create(&p->tid[i], NULL, worker, s);
	}
	return p;
}

void pool_run(pool *p, pool_fn fn, void *arg, int ntasks)
{
	p->fn = fn;
	p->arg = arg;
	p->ntasks = ntasks;
	__atomic_store_n(&p->next, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&p->done, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&p->arrived, 0, __ATOMIC_RELAXED);
	__atomic_fetch_add(&p->gen, 1, __ATOMIC_RELEASE);
	if (__atomic_load_n(&p->sleepers, __ATOMIC_ACQUIRE)) {
		pthread_mutex_lock(&p->m);
		pthread_cond_broadcast(&p->c);
		pthread_mutex_unlock(&p->m);
	}
	work(p, fn, arg, ntasks);
	while (__atomic_load_n(&p->done, __ATOMIC_ACQUIRE) < ntasks || __atomic_load_n(&p->arrived, __ATOMIC_ACQUIRE) < p->nworkers)
		relax();
}

int pool_threads(const pool *p) { return p->nworkers + 1; }

void pool_set_spin(pool *p, double seconds) { p->spin_s = seconds; }

void pool_close(pool *p)
{
	if (!p) return;
	pthread_mutex_lock(&p->m);
	__atomic_store_n(&p->quit, 1, __ATOMIC_RELEASE);
	pthread_cond_broadcast(&p->c);
	pthread_mutex_unlock(&p->m);
	for (int i = 0; i < p->nworkers; i++) pthread_join(p->tid[i], NULL);
	pthread_mutex_destroy(&p->m);
	pthread_cond_destroy(&p->c);
	free(p->tid);
	free(p);
}
