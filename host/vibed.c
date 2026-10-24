/*
 * vibed: watches VLC, learns from every play, and queues the next song.
 *
 * usage: vibed [-i] [-n] [-b MPRIS_NAME] [-d STATE_DIR]
 *   -i  import the play history in the vlc-discord journal first (only when there is no event log yet)
 *   -n  watch, learn and plan, but never touch VLC's playlist
 *   -b  only this player (an exact bus name such as org.mpris.MediaPlayer2.vlc.instance1234; default: the first VLC)
 *   -d  keep the play log, the network and the status there instead of ~/.local/share/phivibe
 *
 * It reads VLC over MPRIS every second, and at once when VLC announces a change. A play ends when VLC moves to
 * another track or stops; the play is logged
 * with the fraction of the song played and how it ended, measured, not judged:
 *   natural  VLC reached the end (the last position seen within one poll of the length)
 *   next     the track VLC had lined up after it started instead (Next pressed)
 *   pick     any other track started (you chose it)
 *   stop     playback stopped mid-song;  quit  VLC went away
 * After each play the preference network (host/pref.h) takes a few Adam steps on every logged play, its products
 * on the card. When a track starts, the planner (host/dstar.h) runs from it over the vibe graph with the network's
 * costs, and the plan's first song goes into VLC right after the current track, unless you have queued tracks
 * after it yourself or VLC shuffles (then nothing is added). A pick of ours that is still waiting is taken back when the plan
 * changes or when you add tracks; our older picks that still follow the current track are taken out.
 *
 * A skip (Next before the end) is learnt from at once, and then the plan is repaired: from the last song you let
 * play, with the skipped one blocked. When the repaired plan starts with another song than the pick VLC just
 * started, VLC plays the new song at once, inserted after the stale pick; the stale pick stays in the playlist and its
 * fraction of a second is not logged. vibed never removes the track VLC is playing, and never replaces a track you
 * lined up yourself.
 *
 * Files in ~/.local/share/phivibe: events.tsv (the play log), pref.bin (the network), status.txt (now, the plan,
 * the weights); emb.bin/emb.tsv from vibe-train. The card's GEMM daemon (vgd) must be running; without it vibed
 * keeps logging and retries every 30 s.
 */
#define _GNU_SOURCE
#include "dstar.h"
#include "emb.h"
#include "mpris.h"
#include "pref.h"
#include "vg.h"

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define NEAR_END 1.5      /* seconds: a play that stopped this close to its end reached it (one poll plus slack) */
#define HIST 8
#define STEPS 16          /* Adam steps after each play */
#define BATCH 64
#define LR 2e-3f
#define DECAY 1e-4f

typedef struct {
	double t0, t1, length, played;
	char end[8], source[8];
	int song, queued;     /* emb indexes, -1 when not in the library */
} play;

static emb E;
static vg *card;
static pref *P;
static dstar *DS;
static int *goff, *gadj;
static play *plays;
static int nplays, capplays;
static char dir[4096], path_events[4200], path_pref[4200], path_status[4200];
static const char *root;
static volatile sig_atomic_t quit;
static double last_card_try;
static double last_loss[3] = { NAN, NAN, NAN };
static int no_touch;
static double *lengths;   /* seconds, per song, from the extraction index */
static int *rep;          /* copies of one recording (same audio in two folders) share rep[]: see dup_groups */

static void say(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "vibed: ");
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
}

static double unix_now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

static const char *rel_of(int song) { return song >= 0 ? E.rel[song] : "-"; }

/* a file:// url -> the song index (-1 outside the library or not embedded) */
static int song_of_url(const char *url)
{
	char p[8192];
	if (mp_path(url, p, sizeof p)) return -1;
	const size_t n = strlen(root);
	if (strncmp(p, root, n) || p[n] != '/') return -1;
	return emb_find(&E, p + n + 1);
}

/* ---- the play log ---- */

static void add_play(const play *pl)
{
	if (nplays == capplays) { capplays = capplays ? 2 * capplays : 256; plays = realloc(plays, (size_t)capplays * sizeof *plays); }
	plays[nplays++] = *pl;
}

static void log_play(const play *pl)
{
	FILE *f = fopen(path_events, "a");
	if (!f) { say("%s: %s", path_events, strerror(errno)); return; }
	fprintf(f, "%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\n", pl->t0, pl->t1, pl->length, pl->played, pl->end, pl->source,
	        rel_of(pl->queued), rel_of(pl->song));
	fclose(f);
}

static void load_plays(void)
{
	FILE *f = fopen(path_events, "r");
	if (!f) return;
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	while ((len = getline(&line, &cap, f)) > 0) {
		if (line[len - 1] == '\n') line[--len] = 0;
		char *fld[8], *s = line;
		int nf = 0;
		while (nf < 7) { char *t = strchr(s, '\t'); if (!t) break; *t = 0; fld[nf++] = s; s = t + 1; }
		if (nf != 7) continue;
		fld[7] = s;
		play pl = { atof(fld[0]), atof(fld[1]), atof(fld[2]), atof(fld[3]), "", "", emb_find(&E, fld[7]), strcmp(fld[6], "-") ? emb_find(&E, fld[6]) : -1 };
		snprintf(pl.end, sizeof pl.end, "%s", fld[4]);
		snprintf(pl.source, sizeof pl.source, "%s", fld[5]);
		add_play(&pl);
	}
	free(line);
	fclose(f);
}
/* the song a vlc-discord title names: "Artist - Title" from the file's tags when it has them, else the file name
 * without its extension. The exact file name first (copies in two folders are the same audio: the first), then a
 * file name containing the title part, narrowed by the first artist's name in the path when several do. -1: none. */
static int match_title(char **stem, const char *title)
{
	for (int i = 0; i < E.n; i++) if (!strcmp(stem[i], title)) return i;
	const char *sep = strstr(title, " - ");
	if (!sep) return -1;
	const char *part = sep + 3;
	char artist[512];
	snprintf(artist, sizeof artist, "%.*s", (int)(sep - title), title);
	artist[strcspn(artist, ",&")] = 0;
	for (char *e = artist + strlen(artist); e > artist && e[-1] == ' '; e--) e[-1] = 0;
	int first = -1, hits = 0, best = -1;
	for (int i = 0; i < E.n; i++) {
		if (!strcasestr(stem[i], part)) continue;
		if (first < 0) first = i;
		hits++;
		if (best < 0 && *artist && strcasestr(E.rel[i], artist)) best = i;
	}
	return hits == 1 ? first : best;
}

/* the vlc-discord journal: "now playing: TITLE" when a track starts (and again on some refreshes), "cleared
 * (Paused)" / "cleared (Stopped)" when it pauses or stops, the unit's own Stopping line when it shuts down. A play
 * lasts from its first line to the next other event; paused stretches are not counted as heard. */
static void import_journal(void)
{
	char **stem = malloc((size_t)E.n * sizeof(char *));
	for (int i = 0; i < E.n; i++) {
		const char *b = strrchr(E.rel[i], '/');
		stem[i] = strdup(b ? b + 1 : E.rel[i]);
		char *dot = strrchr(stem[i], '.');
		if (dot) *dot = 0;
	}
	FILE *j = popen("journalctl --user -u vlc-discord --no-pager -o short-unix 2>/dev/null", "r");
	if (!j) { say("journalctl: %s", strerror(errno)); return; }
	char *line = NULL, cur[4096] = "";
	size_t cap = 0;
	double t0 = 0, seg = 0, heard = 0, pause_t = 0;
	int open = 0, paused = 0, imported = 0, unknown = 0;
#define CLOSE(tend)                                                                                         \
	do {                                                                                                \
		const int song_ = match_title(stem, cur);                                                   \
		if (song_ >= 0) {                                                                           \
			play pl_ = { t0, (tend), lengths[song_], heard, "import", "import", song_, -1 };    \
			add_play(&pl_);                                                                     \
			imported++;                                                                         \
		} else { unknown++; say("journal: no library file for \"%s\"", cur); }                      \
		cur[0] = 0; open = paused = 0; heard = 0;                                                   \
	} while (0)
	while (getline(&line, &cap, j) > 0) {
		line[strcspn(line, "\n")] = 0;
		const double t = atof(line);
		const char *np = strstr(line, "]: now playing: "), *cl = strstr(line, "]: cleared (");
		const int stopping = !np && !cl && strstr(line, "Stopping ");
		if (np) {
			const char *title = np + 16;
			if (open && !strcmp(title, cur)) { if (paused) { paused = 0; seg = t; } continue; }
			if (open) { if (!paused) heard += t - seg; CLOSE(paused ? pause_t : t); }
			snprintf(cur, sizeof cur, "%s", title);
			open = 1;
			t0 = seg = t;
		} else if ((cl || stopping) && open) {
			if (!paused) heard += t - seg;
			if (cl && strstr(cl, "(Paused)")) { paused = 1; pause_t = t; }
			else CLOSE(paused ? pause_t : t);
		}
	}
	if (open && paused) CLOSE(pause_t);   /* a play still running is not finished: left out */
#undef CLOSE
	free(line);
	pclose(j);
	for (int i = 0; i < E.n; i++) free(stem[i]);
	free(stem);
	say("journal: %d plays imported, %d not matched to a library file", imported, unknown);
}

/* song lengths from the extraction index (index.tsv: rec size mtime dur_ms ncrops starts rel) */
static void load_lengths(void)
{
	lengths = calloc((size_t)E.n, sizeof(double));
	char p[4096];
	snprintf(p, sizeof p, "%s/.cache/phivibe/lib/index.tsv", getenv("HOME"));
	FILE *f = fopen(p, "r");
	if (!f) return;
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	while ((len = getline(&line, &cap, f)) > 0) {
		if (line[len - 1] == '\n') line[--len] = 0;
		char *fld[7], *s = line;
		int nf = 0;
		while (nf < 6) { char *t = strchr(s, '\t'); if (!t) break; *t = 0; fld[nf++] = s; s = t + 1; }
		if (nf != 6) continue;
		const int i = emb_find(&E, s);
		if (i >= 0) lengths[i] = atof(fld[3]) / 1000.0;
	}
	free(line);
	fclose(f);
}

static double frac_of(const play *pl)
{
	const double L = pl->length > 0 ? pl->length : pl->song >= 0 ? lengths[pl->song] : 0;
	if (L <= 0) return -1;
	const double f = pl->played / L;
	return f < 0 ? 0 : f > 1 ? 1 : f;
}

/* ---- contexts and examples from the log ---- */

/* the context when play k started (the song then playing: k's), with the plays before it */
static int context_at(int k, float *x)
{
	if (k < 0 || plays[k].song < 0) return -1;
	const float *he[HIST];
	float hf[HIST];
	int nh = 0;
	for (int j = k - 1; j >= 0 && nh < HIST; j--) {
		if (plays[j].song < 0) continue;
		const double f = frac_of(&plays[j]);
		if (f < 0) continue;
		he[nh] = E.mean + (size_t)plays[j].song * E.d;
		hf[nh++] = (float)f;
	}
	pref_context(x, E.mean + (size_t)plays[k].song * E.d, he, hf, nh, plays[k].t0);
	return 0;
}

/* the recency inputs of song b as of the start of play k (k == nplays: now, at time t) */
static void recency(int b, int k, double t, float *r)
{
	r[0] = r[1] = 0;
	for (int j = k - 1; j >= 0; j--)
		if (plays[j].song >= 0 && rep[plays[j].song] == rep[b]) {   /* a copy of b played: b played */
			r[0] = (float)(1.0 / (1 + (k - 1 - j)));
			r[1] = (float)exp(-((t - plays[j].t1) / 3600.0) / 12.0);
			return;
		}
}

static uint64_t rng = 0x1234abcdull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

/* training on the whole log: fraction examples (play k's fraction, decided at the start of play k-1), goal
 * examples (from play k-1's context to play k's vibe, weighted by k's fraction), choices (play k was your pick over
 * our queued song) */
static void learn(int steps)
{
	if (!card || nplays < 2) return;
	const double t0 = unix_now();
	pref_example *fr = malloc((size_t)nplays * sizeof *fr);
	float (*gx)[PREF_X] = malloc((size_t)nplays * sizeof *gx);
	pref_goal_ex *go = malloc((size_t)nplays * sizeof *go);
	pref_choice *ch = malloc((size_t)nplays * sizeof *ch);
	int nfr = 0, ngo = 0, nch = 0;
	for (int k = 1; k < nplays; k++) {
		const play *a = &plays[k - 1], *b = &plays[k];
		const double f = frac_of(b);
		if (a->song < 0 || b->song < 0 || f < 0) continue;
		if (context_at(k - 1, gx[k]) < 0) continue;
		fr[nfr] = (pref_example){ .song = b->song, .target = (float)f };
		memcpy(fr[nfr].x, gx[k], sizeof gx[k]);
		recency(b->song, k, a->t0, fr[nfr].r);
		nfr++;
		go[ngo++] = (pref_goal_ex){ gx[k], E.mean + (size_t)a->song * E.d, E.mean + (size_t)b->song * E.d, (float)f };
		if (!strcmp(a->end, "pick") && b->queued >= 0 && b->queued != b->song) {
			pref_choice c = { gx[k], E.mean + (size_t)a->song * E.d, a->song, b->song, b->queued, { 0, 0 }, { 0, 0 } };
			recency(b->song, k, a->t0, c.rq);
			recency(b->queued, k, a->t0, c.rp);
			ch[nch++] = c;
		}
	}
	/* note: a play's "pick" end is on the play before the chosen one; the chosen play carries the queued song */
	if (!nfr) goto out;
	pref_example *bf = malloc(BATCH * sizeof *bf);
	pref_goal_ex *bg = malloc(BATCH * sizeof *bg);
	pref_choice *bc = malloc(BATCH * sizeof *bc);
	for (int s = 0; s < steps; s++) {
		/* the newest of each kind always, the rest drawn from the whole log */
		const int nb = nfr < BATCH ? nfr : BATCH, ng = ngo < BATCH ? ngo : BATCH, nc = nch < BATCH ? nch : BATCH;
		for (int i = 0; i < nb; i++) bf[i] = nfr <= BATCH ? fr[i] : i == 0 ? fr[nfr - 1] : fr[xr() % (uint64_t)nfr];
		for (int i = 0; i < ng; i++) bg[i] = ngo <= BATCH ? go[i] : i == 0 ? go[ngo - 1] : go[xr() % (uint64_t)ngo];
		for (int i = 0; i < nc; i++) bc[i] = nch <= BATCH ? ch[i] : i == 0 ? ch[nch - 1] : ch[xr() % (uint64_t)nch];
		if (pref_grad(P, bf, nb, bg, ng, bc, nc, &last_loss[0], &last_loss[1], &last_loss[2])) {
			say("card error during training; reconnecting later");
			vg_close(card);
			card = NULL;
			break;
		}
		pref_adam(P, LR, DECAY);
		if (pref_upload(P)) { vg_close(card); card = NULL; break; }
	}
	free(bf); free(bg); free(bc);
	if (card) {
		pref_save(P, path_pref);
		say("learned from %d plays in %.0f ms (%d steps; %d fraction, %d goal, %d choice examples): losses %.4f %.4f %.4f; alpha %.3f beta %.3f kappa %.3f",
		    nplays, 1e3 * (unix_now() - t0), steps, nfr, ngo, nch, last_loss[0], last_loss[1], last_loss[2], pref_alpha(P), pref_beta(P), pref_kappa(P));
	}
out:
	free(fr); free(gx); free(go); free(ch);
}

/* ---- the card ---- */

static int ensure_card(void)
{
	if (card) return 1;
	const double t = unix_now();
	if (t - last_card_try < 30) return 0;
	last_card_try = t;
	card = vg_open(NULL, 0, 3.0);
	if (!card) return 0;
	if (pref_bind(P, card, E.mean, E.n)) { say("the card's arena is too small"); vg_close(card); card = NULL; return 0; }
	if (!DS) {
		if (graph_build_csls(card, E.mean, E.n, E.d, 16, &goff, &gadj)) { vg_close(card); card = NULL; return 0; }
		DS = ds_new(E.n, E.mean, E.d, goff, gadj, PREF_ALPHA_FLOOR);
		const int with = dup_groups(card, E.mean, E.n, E.d, 0.9999, rep);
		say("copies of one recording: %d songs have one (vibe cosine >= 0.9999)", with);
		say("vibe graph: %d songs, %d edges", E.n, goff[E.n]);
	}
	say("card connected (arena %llu MiB)", (unsigned long long)(vg_arena(card) >> 20));
	return 1;
}

/* ---- VLC ---- */

static mpris *M;
static struct {
	int open;
	char id[128], url[8192], next_id[128];
	play pl;
	double last_pos;
} cur;
/* our picks: the one waiting (pend), and every track we added (to tell them from yours) */
typedef struct { char id[128]; int song; char uri[8192]; } pick;
static pick pend = { "", -1, "" };
#define OURS 64
static pick ours[OURS];
static int nours;
/* after a skip: the re-planned track we told VLC to play, the stale pick it replaces, how long we wait for VLC */
static struct { int active; char id[128], uri[8192]; pick stale; double until; } sw;
#define BLOCKS 64
static int blocked[BLOCKS], nblocked;
static int plan[32], nplan;
static double plan_cost;
static int skips_replanned, skips_kept;
static int shuffling;   /* VLC's Shuffle, read each tick */
static double stop_since;   /* when VLC was first seen stopped (it reports Stopped for an instant while switching tracks) */

static void write_status(void)
{
	char tmp[4300];
	snprintf(tmp, sizeof tmp, "%s.tmp", path_status);
	FILE *f = fopen(tmp, "w");
	if (!f) return;
	time_t t = time(NULL);
	char ts[64];
	strftime(ts, sizeof ts, "%F %T %Z", localtime(&t));
	fprintf(f, "updated   %s\n", ts);
	fprintf(f, "card      %s\n", card ? "connected" : "not connected (retrying every 30 s)");
	fprintf(f, "playing   %s\n", cur.open ? rel_of(cur.pl.song) : "-");
	fprintf(f, "queued    %s%s\n", pend.song >= 0 ? rel_of(pend.song) : "-", no_touch ? " (watch only: not added to VLC)" : "");
	fprintf(f, "plan      cost %.3f:", plan_cost);
	for (int i = 0; i < nplan; i++) fprintf(f, "\n  %d. %s", i + 1, rel_of(plan[i]));
	fprintf(f, "\nweights   alpha %.4f  beta %.4f  kappa %.4f  (network step %ld)\n", pref_alpha(P), pref_beta(P), pref_kappa(P), P->step);
	if (isnan(last_loss[0])) fprintf(f, "losses    - (no training since vibed started)\n");
	else fprintf(f, "losses    fraction %.4f  goal %.4f  choice %.4f\n", last_loss[0], last_loss[1], last_loss[2]);
	fprintf(f, "plays     %d logged\n", nplays);
	fprintf(f, "skips     %d re-planned to another song, %d kept the queued one (since vibed started)\n", skips_replanned, skips_kept);
	fclose(f);
	rename(tmp, path_status);
}

/* the picks survive a restart of vibed: the pending file holds the waiting pick on its first line (empty when none),
 * then every other track we added, one "id \t url" per line */
static char path_pending[4200];
static void save_pending(void)
{
	FILE *f = fopen(path_pending, "w");
	if (!f) return;
	fprintf(f, "%s%s%s\n", pend.id, pend.id[0] ? "\t" : "", pend.id[0] ? pend.uri : "");
	for (int i = 0; i < nours; i++) if (strcmp(ours[i].id, pend.id)) fprintf(f, "%s\t%s\n", ours[i].id, ours[i].uri);
	fclose(f);
}

static void remember(const pick *p)
{
	if (!p->id[0]) return;
	for (int i = 0; i < nours; i++) if (!strcmp(ours[i].id, p->id)) { ours[i] = *p; return; }
	if (nours == OURS) { memmove(ours, ours + 1, sizeof ours[0] * (OURS - 1)); nours--; }
	ours[nours++] = *p;
}

static int is_ours(const char *id)
{
	for (int i = 0; i < nours; i++) if (!strcmp(ours[i].id, id)) return i;
	return -1;
}

static void forget(const char *id)
{
	const int i = is_ours(id);
	if (i < 0) return;
	memmove(ours + i, ours + i + 1, sizeof ours[0] * (size_t)(nours - i - 1));
	nours--;
}

static void load_pending(void)
{
	FILE *f = fopen(path_pending, "r");
	char line[8400];
	if (!f) return;
	for (int first = 1; fgets(line, sizeof line, f); first = 0) {
		line[strcspn(line, "\n")] = 0;
		char *tab = strchr(line, '\t');
		if (!tab) continue;
		*tab = 0;
		pick p;
		snprintf(p.id, sizeof p.id, "%.127s", line);
		snprintf(p.uri, sizeof p.uri, "%s", tab + 1);
		p.song = song_of_url(p.uri);
		if (first) pend = p;
		remember(&p);
	}
	fclose(f);
}

/* removes a track we added, only while that id still holds our url (VLC numbers its items afresh after it
 * restarts), and never the track VLC is playing; 1 when removed */
static int remove_ours(const char *id, const char *uri, const char *why)
{
	char u[8192];
	int done = 0;
	mp_state now;
	const int playing_it = mp_read(M, &now) == 1 && !strcmp(now.trackid, id);
	if (playing_it) say("not taking back %s: VLC is playing it", rel_of(song_of_url(uri)));
	else if (!no_touch && id[0] && mp_track_url(M, id, u, sizeof u) == 0 && !strcmp(u, uri) && mp_remove(M, id) == 0) {
		say("took back %s (%s)", rel_of(song_of_url(uri)), why);
		done = 1;
	}
	forget(id);
	return done;
}

static void take_back_pick(const char *why)
{
	if (pend.id[0]) remove_ours(pend.id, pend.uri, why);
	pend.id[0] = 0;
	pend.song = -1;
	save_pending();
}

static void close_play(const char *end)
{
	if (!cur.open) return;
	cur.pl.t1 = unix_now();
	cur.pl.played = cur.last_pos;
	snprintf(cur.pl.end, sizeof cur.pl.end, "%s", end);
	add_play(&cur.pl);
	log_play(&cur.pl);
	const double f = frac_of(&cur.pl);
	say("played %s: %.0f of %.0f s (%.3f), %s, %s", rel_of(cur.pl.song), cur.pl.played, cur.pl.length, f, end, cur.pl.source);
	cur.open = 0;
	if (ensure_card()) learn(STEPS);
}

/* the network's view of every song in context x, then D* Lite from song s with the songs in blk[] blocked:
 * plan[] and plan_cost. 0, or -1 when the card failed. */
static int plan_from(int s, const float *x, const int *blk, int nblk)
{
	nplan = 0;
	plan_cost = NAN;
	float *r = malloc((size_t)E.n * 2 * sizeof(float)), *f = malloc((size_t)E.n * 4), *gc = malloc((size_t)E.n * 4), g[PREF_D];
	const double t = unix_now();
	int rc = 0;
	/* every song's recency now, from the finished plays */
	for (int b = 0; b < E.n; b++) r[2 * b] = r[2 * b + 1] = 0;
	for (int j = nplays - 1, ago = 1; j >= 0; j--, ago++) {
		const int b = plays[j].song >= 0 ? rep[plays[j].song] : -1;   /* the recording: every copy gets its recency below */
		if (b < 0 || r[2 * b] != 0) continue;
		r[2 * b] = (float)(1.0 / (1 + ago));
		r[2 * b + 1] = (float)exp(-((t - plays[j].t1) / 3600.0) / 12.0);
	}
	for (int b = 0; b < E.n; b++) if (rep[b] != b) { r[2 * b] = r[2 * rep[b]]; r[2 * b + 1] = r[2 * rep[b] + 1]; }
	if (pref_score(P, x, E.mean + (size_t)s * E.d, r, f, g, gc)) { vg_close(card); card = NULL; rc = -1; goto out; }
	ds_set_alpha(DS, pref_alpha(P));
	const double kap = pref_kappa(P), bet = pref_beta(P);
	for (int b = 0; b < E.n; b++) {
		ds_set_node(DS, b, kap * (1 - f[b]));
		ds_set_goal(DS, b, bet * acos(gc[b] > 1 ? 1 : gc[b] < -1 ? -1 : gc[b]));
	}
	for (int i = 0; i < nblocked; i++) ds_set_blocked(DS, blocked[i], 0);
	nblocked = 0;
	/* a blocked song with every copy of its recording */
	for (int i = 0; i < nblk; i++) {
		if (blk[i] < 0) continue;
		for (int b = 0; b < E.n && nblocked < BLOCKS; b++)
			if (rep[b] == rep[blk[i]]) { ds_set_blocked(DS, b, 1); blocked[nblocked++] = b; }
	}
	ds_move(DS, s);
	nplan = ds_plan(DS, plan, 32, &plan_cost);
	if (nplan > 0) say("plan from %s: %d songs, cost %.3f; next %s (predicted fraction %.3f)", rel_of(s), nplan, plan_cost, rel_of(plan[0]), f[plan[0]]);
	else say("no plan from %s", rel_of(s));
out:
	free(r); free(f); free(gc);
	return rc;
}

static void uri_of(int song, char *uri, size_t n)
{
	char p[8192];
	snprintf(p, sizeof p, "%s/%s", root, E.rel[song]);
	mp_uri(p, uri, n);
}

/* the plan's first song right after the current track, unless you have tracks lined up after it. Our own earlier
 * picks after it are stale (planned from another song): they go. */
static void queue_after_current(void)
{
	if (no_touch) { pend.song = nplan > 0 ? plan[0] : -1; return; }
	if (shuffling) { take_back_pick("VLC shuffles"); say("VLC shuffles: not adding"); return; }
	char ids[256][128];
	const int n = mp_tracks(M, ids, 256);   /* exact right after a track change */
	int at = -1, user_after = 0;
	for (int i = 0; i < n; i++) if (!strcmp(ids[i], cur.id)) at = i;
	char stale_id[64][128], stale_uri[64][8192];
	int nstale = 0;
	for (int i = at + 1; at >= 0 && i < n; i++) {
		const int o = is_ours(ids[i]);
		if (o >= 0 && nstale < 64) { snprintf(stale_id[nstale], 128, "%s", ours[o].id); snprintf(stale_uri[nstale], 8192, "%s", ours[o].uri); nstale++; }
		else if (o < 0) user_after = 1;
	}
	for (int i = 0; i < nstale; i++) remove_ours(stale_id[i], stale_uri[i], strcmp(stale_id[i], pend.id) ? "an older pick after this track" : "re-planned");
	if (pend.id[0] && is_ours(pend.id) < 0) { pend.id[0] = 0; pend.song = -1; }
	take_back_pick("re-planned");
	if (user_after) { say("tracks you lined up follow this one: not adding"); save_pending(); return; }
	if (nplan == 0) { save_pending(); return; }
	uri_of(plan[0], pend.uri, sizeof pend.uri);
	if (mp_add_after(M, pend.uri, cur.id, 0, pend.id, sizeof pend.id) == 0) {
		pend.song = plan[0];
		remember(&pend);
		say("queued %s after %s (track %s)", rel_of(pend.song), rel_of(cur.pl.song), pend.id[0] ? pend.id : "?");
	}
	save_pending();
}

static void decide(void)
{
	const int s = cur.pl.song;
	nplan = 0;
	if (s >= 0 && ensure_card()) {
		float x[PREF_X];
		add_play(&cur.pl);   /* the context reads the open play as the current song, then it is dropped again */
		context_at(nplays - 1, x);
		nplays--;
		/* the one hard rule: not the song playing now */
		if (plan_from(s, x, &s, 1) == 0) queue_after_current();
	}
	write_status();
}

/*
 * You skipped: the play just closed ended "next" and VLC started the track it had lined up. When that track is one of
 * our picks, it was planned before the skip; the network has now learnt from the skip (close_play), so plan again
 * from the last song you let play (the latest play that did not end in a skip; the skipped song itself when there
 * is none), with every song skipped since blocked, the way a navigating agent repairs its path around newly found
 * obstacles. When the new plan starts elsewhere, VLC plays the new song at once (the stale pick stays in the
 * playlist, before it). Returns 1
 * when VLC was switched.
 */
static int replan_after_skip(const mp_state *s)
{
	const int k = nplays - 1;
	if (no_touch || shuffling || k < 0 || !card || plays[k].song < 0) return 0;
	const int started_ours = (pend.id[0] && !strcmp(pend.id, s->trackid)) || is_ours(s->trackid) >= 0;
	if (!started_ours) return 0;   /* VLC moved on to a track you lined up yourself */
	int from = k - 1;
	while (from >= 0 && !strcmp(plays[from].end, "next")) from--;   /* back over a run of skips */
	if (from < 0 || plays[from].song < 0) from = k;
	float x[PREF_X];
	if (context_at(from, x)) return 0;
	const double t0 = unix_now();
	const int skipped = plays[k].song, start = plays[from].song, now_song = song_of_url(s->url);
	int blk[BLOCKS], nb = 0;
	blk[nb++] = start;
	for (int j = from + 1; j <= k && nb < BLOCKS; j++) if (plays[j].song >= 0 && plays[j].song != start) blk[nb++] = plays[j].song;
	if (plan_from(start, x, blk, nb) || nplan == 0) return 0;
	if (plan[0] == now_song) {
		skips_kept++;
		say("after the skip of %s the plan still starts with %s: keeping it", rel_of(skipped), rel_of(now_song));
		return 0;
	}
	pick next = { "", plan[0], "" };
	uri_of(plan[0], next.uri, sizeof next.uri);
	if (mp_add_after(M, next.uri, s->trackid, 1, next.id, sizeof next.id)) return 0;
	sw.active = 1;
	snprintf(sw.id, sizeof sw.id, "%s", next.id);
	snprintf(sw.uri, sizeof sw.uri, "%s", next.uri);
	sw.stale = (pick){ "", now_song, "" };
	snprintf(sw.stale.id, sizeof sw.stale.id, "%s", s->trackid);
	snprintf(sw.stale.uri, sizeof sw.stale.uri, "%s", s->url);
	sw.until = unix_now() + 5;
	pend = next;
	remember(&pend);
	save_pending();
	skips_replanned++;
	say("skip of %s: playing %s instead of %s (re-planned from %s in %.0f ms after learning)", rel_of(skipped), rel_of(plan[0]),
	    rel_of(now_song), rel_of(start), 1e3 * (unix_now() - t0));
	return 1;
}

static void open_play(const mp_state *s)
{
	memset(&cur.pl, 0, sizeof cur.pl);
	cur.open = 1;
	snprintf(cur.id, sizeof cur.id, "%s", s->trackid);
	snprintf(cur.url, sizeof cur.url, "%s", s->url);
	cur.pl.t0 = unix_now();
	cur.pl.length = s->length_us / 1e6;
	cur.pl.song = song_of_url(s->url);
	const int ours_now = (pend.id[0] && !strcmp(pend.id, s->trackid)) || is_ours(s->trackid) >= 0;
	snprintf(cur.pl.source, sizeof cur.pl.source, "%s", ours_now ? "vibe" : "user");
	static int seen_start;   /* the first track after vibed starts was not chosen against a pick of this run */
	cur.pl.queued = ours_now || !seen_start ? -1 : pend.song;   /* our waiting pick when you chose this one instead */
	seen_start = 1;
	if (ours_now) { forget(s->trackid); if (!strcmp(pend.id, s->trackid)) { pend.id[0] = 0; pend.song = -1; } save_pending(); }
	cur.last_pos = s->pos_us / 1e6;
	/* the item VLC lines up after this one (its play order is fresh right after a track change) */
	char ids[256][128];
	const int n = mp_tracks(M, ids, 256);
	cur.next_id[0] = 0;
	for (int i = 0; i + 1 < n; i++) if (!strcmp(ids[i], s->trackid)) snprintf(cur.next_id, sizeof cur.next_id, "%s", ids[i + 1]);
	say("started %s (%s, %.0f s)", cur.pl.song >= 0 ? rel_of(cur.pl.song) : s->url, cur.pl.source, cur.pl.length);
}

static void tick(void)
{
	mp_state s;
	const int r = mp_read(M, &s);
	if (r <= 0) {
		if (cur.open) close_play("quit");
		pend.id[0] = 0;
		pend.song = -1;
		nours = 0;
		sw.active = 0;
		return;
	}
	shuffling = s.shuffle;
	/* tracks you add while our pick waits: yours go first */
	mp_added add[16];
	const int na = mp_take_added(M, add, 16);
	for (int i = 0; i < na; i++)
		if (pend.id[0] && is_ours(add[i].trackid) < 0 && strcmp(add[i].url, pend.uri)) { take_back_pick("you added a track"); break; }
	/* after a skip we told VLC to play the re-planned song: wait for it (5 s at most), then drop the stale pick */
	if (sw.active) {
		const int there = (sw.id[0] && !strcmp(s.trackid, sw.id)) || !strcmp(s.url, sw.uri);
		if (!there && unix_now() < sw.until) return;
		sw.active = 0;
		/* the pick VLC started on your Next stays in the playlist (before the new song, so VLC does not come back to
		 * it); it is just no longer ours to take back */
		if (there) forget(sw.stale.id);
		else say("VLC did not switch to the re-planned song within 5 s");
		cur.id[0] = 0;
	}
	const int playing = !strcmp(s.status, "Playing"), paused = !strcmp(s.status, "Paused");
	if (!s.trackid[0] || (!playing && !paused)) {
		if (cur.open) {
			const double t = unix_now();
			if (stop_since == 0) stop_since = t;
			if (t - stop_since < 1.0) return;   /* a stop counts once it has lasted a second */
			close_play(cur.pl.length - cur.last_pos <= NEAR_END ? "natural" : "stop");
		}
		stop_since = 0;
		cur.id[0] = 0;
		return;
	}
	stop_since = 0;
	if (strcmp(s.trackid, cur.id) || strcmp(s.url, cur.url) || !cur.open) {
		int skipped = 0;
		if (cur.open) {
			const int lined_up = (cur.next_id[0] && !strcmp(s.trackid, cur.next_id)) || (pend.id[0] && !strcmp(s.trackid, pend.id)) ||
			                     is_ours(s.trackid) >= 0;
			const char *end = cur.pl.length - cur.last_pos <= NEAR_END ? "natural" : lined_up ? "next" : "pick";
			skipped = !strcmp(end, "next");
			close_play(end);
		}
		if (!playing) { /* a new track shown while paused: wait until it plays */ snprintf(cur.id, sizeof cur.id, "%s", s.trackid); snprintf(cur.url, sizeof cur.url, "%s", s.url); return; }
		if (skipped && replan_after_skip(&s)) { cur.id[0] = 0; return; }
		open_play(&s);
		decide();
		return;
	}
	cur.last_pos = s.pos_us / 1e6;
	if (s.length_us > 0) cur.pl.length = s.length_us / 1e6;
}

static void on_signal(int sig) { (void)sig; quit = 1; }

int main(int argc, char **argv)
{
	int do_import = 0, opt;
	const char *busname = NULL, *state = NULL;
	while ((opt = getopt(argc, argv, "inb:d:")) != -1) {
		if (opt == 'i') do_import = 1;
		else if (opt == 'n') no_touch = 1;
		else if (opt == 'b') busname = optarg;
		else if (opt == 'd') state = optarg;
		else { fprintf(stderr, "usage: vibed [-i] [-n] [-b MPRIS_NAME] [-d STATE_DIR]\n"); return 2; }
	}
	setvbuf(stderr, NULL, _IOLBF, 0);
	if (state) snprintf(dir, sizeof dir, "%s", state);
	else snprintf(dir, sizeof dir, "%s/.local/share/phivibe", getenv("HOME"));
	mkdir(dir, 0755);
	snprintf(path_events, sizeof path_events, "%s/events.tsv", dir);
	snprintf(path_pref, sizeof path_pref, "%s/pref.bin", dir);
	snprintf(path_status, sizeof path_status, "%s/status.txt", dir);
	snprintf(path_pending, sizeof path_pending, "%s/pending", dir);
	root = vibe_root();
	if (emb_load(&E, NULL)) return 1;
	load_lengths();
	rep = malloc((size_t)E.n * sizeof(int));   /* each song its own recording until dup_groups runs on the card */
	for (int i = 0; i < E.n; i++) rep[i] = i;
	P = pref_load(path_pref);
	if (!P) { P = pref_new(0xfeed); say("new preference network (%ld parameters)", pref_nparams()); }
	else say("preference network loaded: step %ld", P->step);
	struct stat sb;
	const int had_log = stat(path_events, &sb) == 0;
	load_plays();
	if (!had_log && do_import) {
		import_journal();
		for (int i = 0; i < nplays; i++) log_play(&plays[i]);
	}
	for (int i = 0; i < nplays; i++) if (plays[i].length <= 0 && plays[i].song >= 0) plays[i].length = lengths[plays[i].song];
	say("%d songs embedded (encoder step %lld), library %s, %d plays logged, state %s%s", E.n, (long long)E.step, root, nplays, dir,
	    no_touch ? ", watch only" : "");
	M = mp_open();
	if (!M) { say("no session bus"); return 1; }
	if (busname) { mp_set_name(M, busname); say("player %s only", busname); }
	load_pending();
	signal(SIGTERM, on_signal);
	signal(SIGINT, on_signal);
	if (ensure_card() && !had_log && nplays > 1) learn(200);
	write_status();
	double last = 0;
	while (!quit) {
		/* a tick at least every second, and at once (at most 10 a second) when a player announces a change */
		const double t = unix_now();
		if (t - last >= 0.1) { tick(); last = t; }
		if (!card && ensure_card()) write_status();
		mp_wait(M, 1.0);
	}
	/* a play still open when vibed stops is not finished: it is not logged (VLC quitting is: see tick) */
	if (P) pref_save(P, path_pref);
	mp_close(M);
	if (card) vg_close(card);
	say("stopped");
	return 0;
}
