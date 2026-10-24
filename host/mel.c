/*
 * vibe-mel: the music library as log-mel crops, the vibe encoder's input.
 *
 * usage: vibe-mel [-r ROOT] [-o DIR] [-j THREADS] [-x TOPDIR]... [-n MAX] [-v]
 * Defaults: the library root (see lib/vibe.h), ~/.cache/phivibe/lib, 3 threads, the top folders listed in ~/.config/phivibe/exclude.
 *
 * Each audio file gives up to CROPS crops of FRAMES frames x NMEL bands (22050 Hz mono, Hann 2048, hop 1024, mel
 * 30 Hz..11025 Hz, power in dB relative to a full-scale sine, -100..+10 dB in 255 steps), one crop centred in each of
 * n equal segments of the song (n = duration / crop length, at most CROPS), so crops never overlap. Songs are decoded
 * front to back; a gap of more than SEEK_GAP to the next crop is skipped with a seek (long mixes read ~6 s per crop,
 * not the whole file).
 *
 * Output (resumable: a file already in the index with the same size and mtime is skipped):
 *   mel.bin    record k at k * REC bytes: CROPS x FRAMES x NMEL u8, crops past ncrops zero
 *   index.tsv  k, size, mtime, duration ms, ncrops, crop starts ms (comma list), path relative to ROOT
 *   errors.tsv size, mtime, message, path (not retried while size and mtime stay the same)
 *
 * Run it at idle I/O priority (the library sits on the disk VLC plays from): ionice -c3 nice -n19, or the systemd
 * unit's IOSchedulingClass=idle.
 */
#define _GNU_SOURCE
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fftw3.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../lib/vibe.h"

#define SEEK_GAP (30 * MEL_SR)   /* samples: a longer gap to the next crop is skipped by seeking */

typedef struct { char *rel; int64_t size, mtime; } item;

static const char *root;
static char outdir[4096];
static item *items;
static int nitems, capitems;
static int next_item;            /* the next item to hand out */
static int64_t next_rec;         /* the next record number */
static int fd_bin = -1;
static FILE *f_index, *f_err;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static int verbose;
static int done_ok, done_err;
static double t_start;

/* the mel filterbank: band b covers bins first[b] .. first[b] + len[b] - 1 */
static int mel_first[MEL_BANDS], mel_len[MEL_BANDS];
static float *mel_w[MEL_BANDS];
static float hann[MEL_NFFT];

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static void say(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "vibe-mel: ");
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
}

static double hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel2hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

static void mel_init(void)
{
	const double lo = hz2mel(MEL_FMIN), hi = hz2mel(MEL_SR / 2.0);
	double edge[MEL_BANDS + 2];
	for (int i = 0; i < MEL_BANDS + 2; i++) edge[i] = mel2hz(lo + (hi - lo) * i / (MEL_BANDS + 1));
	const double binhz = (double)MEL_SR / MEL_NFFT;
	for (int b = 0; b < MEL_BANDS; b++) {
		const double f0 = edge[b], f1 = edge[b + 1], f2 = edge[b + 2];
		int k0 = (int)ceil(f0 / binhz), k2 = (int)floor(f2 / binhz);
		if (k2 > MEL_NFFT / 2) k2 = MEL_NFFT / 2;
		if (k2 < k0) k2 = k0;
		mel_first[b] = k0;
		mel_len[b] = k2 - k0 + 1;
		mel_w[b] = calloc((size_t)mel_len[b], sizeof(float));
		const double area = 2.0 / (f2 - f0);   /* each triangle sums to the same energy (Slaney) */
		int any = 0;
		for (int k = k0; k <= k2; k++) {
			const double f = k * binhz;
			double w = f <= f1 ? (f - f0) / (f1 - f0) : (f2 - f) / (f2 - f1);
			if (w < 0) w = 0;
			mel_w[b][k - k0] = (float)(w * area * binhz);
			any |= w > 0;
		}
		if (!any) {   /* a band narrower than a bin (low bands): the nearest bin, whole */
			int k = (int)lround(f1 / binhz);
			mel_first[b] = k;
			mel_len[b] = 1;
			mel_w[b][0] = 1.0f;
		}
	}
	for (int i = 0; i < MEL_NFFT; i++) hann[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / MEL_NFFT));
}

typedef struct {
	float *in;
	fftwf_complex *out;
	fftwf_plan plan;
	float *pcm;        /* the crop being filled: MEL_CROP_LEN samples */
	float *tmp;        /* resampler output */
	int tmpcap;
} worker;

/* one crop of samples -> FRAMES x NMEL u8 */
static void crop_mel(worker *w, const float *pcm, uint8_t *dst)
{
	/* a full-scale sine through the Hann window peaks at |X| = NFFT / 4: that power is 0 dB */
	const float norm = 1.0f / ((MEL_NFFT / 4.0f) * (MEL_NFFT / 4.0f));
	for (int f = 0; f < MEL_FRAMES; f++) {
		const float *x = pcm + (size_t)f * MEL_HOP;
		for (int i = 0; i < MEL_NFFT; i++) w->in[i] = x[i] * hann[i];
		fftwf_execute(w->plan);
		for (int b = 0; b < MEL_BANDS; b++) {
			double e = 0;
			for (int j = 0; j < mel_len[b]; j++) {
				const fftwf_complex *c = &w->out[mel_first[b] + j];
				e += mel_w[b][j] * ((double)(*c)[0] * (*c)[0] + (double)(*c)[1] * (*c)[1]);
			}
			double db = 10.0 * log10(e * norm + 1e-12);
			int q = (int)lround((db - MEL_DB_MIN) * 255.0 / (MEL_DB_MAX - MEL_DB_MIN));
			dst[(size_t)f * MEL_BANDS + b] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
		}
	}
}

typedef struct {
	int n, i;                        /* crops planned, the crop being filled */
	int64_t start[MEL_CROPS];        /* sample index of each crop (the actual one once filled) */
	int64_t fill;                    /* samples of crop i filled */
	int64_t cur;                     /* sample index of the next resampled sample */
	int want_seek;
	uint8_t *rec;
} plan;

/* feeds resampled samples; returns 1 when every crop is done */
static int consume(worker *w, plan *p, const float *s, int n)
{
	for (int j = 0; j < n && p->i < p->n; j++, p->cur++) {
		if (p->fill == 0) {
			if (p->cur < p->start[p->i]) continue;
			p->start[p->i] = p->cur;   /* a seek can land late: the crop starts where the samples are */
		}
		w->pcm[p->fill++] = s[j];
		if (p->fill == MEL_CROP_LEN) {
			crop_mel(w, w->pcm, p->rec + (size_t)p->i * MEL_CROP_BYTES);
			p->i++;
			p->fill = 0;
			if (p->i < p->n && p->start[p->i] > p->cur + SEEK_GAP) { p->cur++; p->want_seek = 1; return 0; }
		}
	}
	return p->i >= p->n;
}

static int resample_into(worker *w, plan *p, SwrContext *swr, const AVFrame *fr)
{
	int need = swr_get_out_samples(swr, fr ? fr->nb_samples : 0);
	if (need <= 0) return p->i >= p->n;
	if (need > w->tmpcap) {
		w->tmpcap = need + 4096;
		w->tmp = realloc(w->tmp, (size_t)w->tmpcap * sizeof(float));
	}
	uint8_t *out = (uint8_t *)w->tmp;
	int got = swr_convert(swr, &out, w->tmpcap, fr ? (const uint8_t *const *)fr->extended_data : NULL, fr ? fr->nb_samples : 0);
	if (got <= 0) return p->i >= p->n;
	return consume(w, p, w->tmp, got);
}

/* decodes the crops of one file into rec; 0, or -1 with the reason in err */
static int extract(worker *w, const char *path, uint8_t *rec, double *dur, int *ncrops, int64_t *starts, char *err, size_t errlen)
{
	AVFormatContext *fc = NULL;
	AVCodecContext *cc = NULL;
	SwrContext *swr = NULL;
	AVPacket *pkt = av_packet_alloc();
	AVFrame *fr = av_frame_alloc();
	AVChannelLayout in_layout = { 0 }, mono = AV_CHANNEL_LAYOUT_MONO;
	int rc = -1, e;
	plan p;
	memset(&p, 0, sizeof p);
	p.rec = rec;

	if ((e = avformat_open_input(&fc, path, NULL, NULL)) < 0) { snprintf(err, errlen, "open: %s", av_err2str(e)); goto out; }
	if ((e = avformat_find_stream_info(fc, NULL)) < 0) { snprintf(err, errlen, "stream info: %s", av_err2str(e)); goto out; }
	const AVCodec *codec = NULL;
	int si = av_find_best_stream(fc, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
	if (si < 0 || !codec) { snprintf(err, errlen, "no audio stream"); goto out; }
	AVStream *st = fc->streams[si];
	for (unsigned k = 0; k < fc->nb_streams; k++) if ((int)k != si) fc->streams[k]->discard = AVDISCARD_ALL;
	double D = fc->duration > 0 ? fc->duration / (double)AV_TIME_BASE : st->duration > 0 ? st->duration * av_q2d(st->time_base) : 0;
	if (D < 1.0) { snprintf(err, errlen, "duration %.3f s", D); goto out; }
	*dur = D;
	const int64_t Ns = (int64_t)(D * MEL_SR);
	p.n = (int)(Ns / MEL_CROP_LEN);
	if (p.n > MEL_CROPS) p.n = MEL_CROPS;
	if (p.n < 1) p.n = 1;
	const int64_t seg = Ns / p.n;
	for (int i = 0; i < p.n; i++) p.start[i] = i * seg + (seg > MEL_CROP_LEN ? (seg - MEL_CROP_LEN) / 2 : 0);

	cc = avcodec_alloc_context3(codec);
	if (!cc || avcodec_parameters_to_context(cc, st->codecpar) < 0) { snprintf(err, errlen, "codec context"); goto out; }
	cc->pkt_timebase = st->time_base;
	if ((e = avcodec_open2(cc, codec, NULL)) < 0) { snprintf(err, errlen, "codec open: %s", av_err2str(e)); goto out; }
	if (cc->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC || !av_channel_layout_check(&cc->ch_layout))
		av_channel_layout_default(&in_layout, cc->ch_layout.nb_channels > 0 ? cc->ch_layout.nb_channels : 2);
	else
		av_channel_layout_copy(&in_layout, &cc->ch_layout);
	if (swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_FLT, MEL_SR, &in_layout, cc->sample_fmt, cc->sample_rate, 0, NULL) < 0 ||
	    swr_init(swr) < 0) { snprintf(err, errlen, "resampler %s %d Hz", av_get_sample_fmt_name(cc->sample_fmt), cc->sample_rate); goto out; }

	const int64_t t0 = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;
	int sync = 0, done = 0, eof = 0;
	while (!done && !eof) {
		if (p.want_seek) {
			p.want_seek = 0;
			const double t = (p.start[p.i] - MEL_SR) / (double)MEL_SR;
			const int64_t ts = t0 + (int64_t)(t / av_q2d(st->time_base));
			if (av_seek_frame(fc, si, ts, AVSEEK_FLAG_BACKWARD) >= 0) {
				avcodec_flush_buffers(cc);
				swr_close(swr);
				swr_init(swr);
				sync = 1;
			}
		}
		e = av_read_frame(fc, pkt);
		if (e < 0) { eof = 1; avcodec_send_packet(cc, NULL); }
		else if (pkt->stream_index != si) { av_packet_unref(pkt); continue; }
		else { avcodec_send_packet(cc, pkt); av_packet_unref(pkt); }
		while (!done && !p.want_seek && avcodec_receive_frame(cc, fr) >= 0) {
			if (sync) {
				const int64_t ts = fr->best_effort_timestamp != AV_NOPTS_VALUE ? fr->best_effort_timestamp : fr->pts;
				if (ts != AV_NOPTS_VALUE) p.cur = (int64_t)llround((ts - t0) * av_q2d(st->time_base) * MEL_SR);
				sync = 0;
			}
			done = resample_into(w, &p, swr, fr);
			av_frame_unref(fr);
		}
	}
	if (!done) resample_into(w, &p, swr, NULL);   /* the resampler's last samples */
	/* a crop cut short by the end of the file: kept zero-padded when at least half of it is there */
	if (p.i < p.n && p.fill >= MEL_CROP_LEN / 2) {
		memset(w->pcm + p.fill, 0, (size_t)(MEL_CROP_LEN - p.fill) * sizeof(float));
		crop_mel(w, w->pcm, rec + (size_t)p.i * MEL_CROP_BYTES);
		p.i++;
	}
	if (p.i == 0) { snprintf(err, errlen, "no samples decoded"); goto out; }
	*ncrops = p.i;
	for (int i = 0; i < p.i; i++) starts[i] = p.start[i];
	rc = 0;
out:
	av_channel_layout_uninit(&in_layout);
	swr_free(&swr);
	avcodec_free_context(&cc);
	avformat_close_input(&fc);
	av_frame_free(&fr);
	av_packet_free(&pkt);
	return rc;
}

static void *run(void *v)
{
	(void)v;
	worker w = { 0 };
	pthread_mutex_lock(&m);   /* FFTW planning is not thread safe */
	w.in = fftwf_alloc_real(MEL_NFFT);
	w.out = fftwf_alloc_complex(MEL_NFFT / 2 + 1);
	w.plan = fftwf_plan_dft_r2c_1d(MEL_NFFT, w.in, w.out, FFTW_MEASURE);
	pthread_mutex_unlock(&m);
	w.pcm = malloc(MEL_CROP_LEN * sizeof(float));
	uint8_t *rec = malloc(MEL_REC);
	char path[8192], err[512];
	for (;;) {
		pthread_mutex_lock(&m);
		int k = next_item < nitems ? next_item++ : -1;
		pthread_mutex_unlock(&m);
		if (k < 0) break;
		item *it = &items[k];
		snprintf(path, sizeof path, "%s/%s", root, it->rel);
		memset(rec, 0, MEL_REC);
		double dur = 0, t = now();
		int n = 0;
		int64_t starts[MEL_CROPS];
		int rc = extract(&w, path, rec, &dur, &n, starts, err, sizeof err);
		pthread_mutex_lock(&m);
		if (rc == 0) {
			const int64_t r = next_rec++;
			if (pwrite(fd_bin, rec, MEL_REC, (off_t)(r * MEL_REC)) != MEL_REC) {
				say("write mel.bin: %s", strerror(errno));
				exit(1);
			}
			fprintf(f_index, "%lld\t%lld\t%lld\t%lld\t%d\t", (long long)r, (long long)it->size, (long long)it->mtime,
			        (long long)llround(dur * 1000), n);
			for (int i = 0; i < n; i++) fprintf(f_index, "%s%lld", i ? "," : "", (long long)(starts[i] * 1000 / MEL_SR));
			fprintf(f_index, "\t%s\n", it->rel);
			fflush(f_index);
			done_ok++;
		} else {
			fprintf(f_err, "%lld\t%lld\t%s\t%s\n", (long long)it->size, (long long)it->mtime, err, it->rel);
			fflush(f_err);
			done_err++;
		}
		const int d = done_ok + done_err;
		if (verbose || d % 100 == 0)
			say("%d/%d ok %d err %d, %.1f s elapsed; %s %.2f s %d crops %s%s", d, nitems, done_ok, done_err, now() - t_start,
			    rc ? "FAIL" : "ok", now() - t, n, it->rel, rc ? err : "");
		pthread_mutex_unlock(&m);
	}
	free(rec);
	free(w.pcm);
	free(w.tmp);
	return NULL;
}

static int audio_name(const char *n)
{
	static const char *ext[] = { "flac", "mp3", "wav", "ogg", "opus", "m4a", "aac", "wma", "aif", "aiff", "ape", "wv", NULL };
	const char *dot = strrchr(n, '.');
	if (!dot) return 0;
	for (int i = 0; ext[i]; i++) if (!strcasecmp(dot + 1, ext[i])) return 1;
	return 0;
}

static char *excl[16];
static int nexcl;

static void scan(const char *rel)
{
	char path[8192];
	snprintf(path, sizeof path, "%s%s%s", root, *rel ? "/" : "", rel);
	DIR *d = opendir(path);
	if (!d) { say("opendir %s: %s", path, strerror(errno)); return; }
	struct dirent *de;
	while ((de = readdir(d))) {
		if (de->d_name[0] == '.' && (!de->d_name[1] || (de->d_name[1] == '.' && !de->d_name[2]))) continue;
		char sub[8192];
		snprintf(sub, sizeof sub, "%s%s%s", rel, *rel ? "/" : "", de->d_name);
		char full[12288];
		snprintf(full, sizeof full, "%s/%s", root, sub);
		struct stat sb;
		if (stat(full, &sb)) continue;
		if (S_ISDIR(sb.st_mode)) {
			int skip = 0;
			for (int i = 0; i < nexcl; i++) skip |= !*rel && !strcmp(de->d_name, excl[i]);
			if (!skip) scan(sub);
		} else if (S_ISREG(sb.st_mode) && audio_name(de->d_name)) {
			if (nitems == capitems) { capitems = capitems ? capitems * 2 : 1024; items = realloc(items, (size_t)capitems * sizeof *items); }
			items[nitems++] = (item){ strdup(sub), (int64_t)sb.st_size, (int64_t)sb.st_mtime };
		}
	}
	closedir(d);
}

/* $VIBE_LIBRARY, else the first line of ~/.config/phivibe/library, else ~/Music */
static const char *library_root(void)
{
	static char buf[4096];
	const char *env = getenv("VIBE_LIBRARY"), *home = getenv("HOME") ? getenv("HOME") : ".";
	if (env && *env) return env;
	snprintf(buf, sizeof buf, "%s/.config/phivibe/library", home);
	FILE *f = fopen(buf, "r");
	if (f) {
		const int ok = fgets(buf, sizeof buf, f) != NULL;
		fclose(f);
		buf[strcspn(buf, "\r\n")] = 0;
		if (ok && *buf) return buf;
	}
	snprintf(buf, sizeof buf, "%s/Music", home);
	return buf;
}

static int cmp_item(const void *a, const void *b) { return strcmp(((const item *)a)->rel, ((const item *)b)->rel); }

/* the (size, mtime, path) triples already handled, from index.tsv and errors.tsv */
typedef struct { int64_t size, mtime; char *rel; } seen_t;
static seen_t *seen;
static int nseen, capseen;

static int cmp_seen(const void *a, const void *b) { return strcmp(((const seen_t *)a)->rel, ((const seen_t *)b)->rel); }

static void load_seen(const char *file, int index)
{
	FILE *f = fopen(file, "r");
	if (!f) return;
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	while ((len = getline(&line, &cap, f)) > 0) {
		if (line[len - 1] == '\n') line[--len] = 0;
		/* index: k size mtime dur n starts rel; errors: size mtime msg rel */
		char *fld[8];
		int nf = 0;
		char *s = line;
		const int want = index ? 7 : 4;
		while (nf < want - 1) {
			char *tab = strchr(s, '\t');
			if (!tab) break;
			*tab = 0;
			fld[nf++] = s;
			s = tab + 1;
		}
		if (nf != want - 1) continue;
		fld[nf++] = s;
		if (nseen == capseen) { capseen = capseen ? capseen * 2 : 1024; seen = realloc(seen, (size_t)capseen * sizeof *seen); }
		seen[nseen++] = (seen_t){ atoll(fld[index ? 1 : 0]), atoll(fld[index ? 2 : 1]), strdup(fld[want - 1]) };
		if (index) {
			int64_t r = atoll(fld[0]);
			if (r + 1 > next_rec) next_rec = r + 1;
		}
	}
	free(line);
	fclose(f);
}

int main(int argc, char **argv)
{
	int threads = 3, max = -1, opt;
	snprintf(outdir, sizeof outdir, "%s/.cache/phivibe/lib", getenv("HOME") ? getenv("HOME") : ".");
	while ((opt = getopt(argc, argv, "r:o:j:x:n:v")) != -1) {
		if (opt == 'r') root = optarg;
		else if (opt == 'o') snprintf(outdir, sizeof outdir, "%s", optarg);
		else if (opt == 'j') threads = atoi(optarg);
		else if (opt == 'x' && nexcl < 16) excl[nexcl++] = optarg;
		else if (opt == 'n') max = atoi(optarg);
		else if (opt == 'v') verbose = 1;
		else { fprintf(stderr, "usage: vibe-mel [-r ROOT] [-o DIR] [-j THREADS] [-x TOPDIR]... [-n MAX] [-v]\n"); return 2; }
	}
	if (!root) root = library_root();
	if (!nexcl) {   /* ~/.config/phivibe/exclude: one top folder per line */
		char ex[4096], line[1024];
		snprintf(ex, sizeof ex, "%s/.config/phivibe/exclude", getenv("HOME") ? getenv("HOME") : ".");
		FILE *xf = fopen(ex, "r");
		while (xf && nexcl < 16 && fgets(line, sizeof line, xf)) {
			line[strcspn(line, "\r\n")] = 0;
			if (*line) excl[nexcl++] = strdup(line);
		}
		if (xf) fclose(xf);
	}
	av_log_set_level(AV_LOG_QUIET);
	mel_init();

	char f[4352];
	snprintf(f, sizeof f, "%s/index.tsv", outdir);
	load_seen(f, 1);
	f_index = fopen(f, "a");
	snprintf(f, sizeof f, "%s/errors.tsv", outdir);
	load_seen(f, 0);
	f_err = fopen(f, "a");
	snprintf(f, sizeof f, "%s/mel.bin", outdir);
	fd_bin = open(f, O_WRONLY | O_CREAT, 0644);
	if (!f_index || !f_err || fd_bin < 0) { say("%s: %s", outdir, strerror(errno)); return 1; }

	t_start = now();
	scan("");
	qsort(items, (size_t)nitems, sizeof *items, cmp_item);
	qsort(seen, (size_t)nseen, sizeof *seen, cmp_seen);
	int keep = 0, skipped = 0;
	for (int i = 0; i < nitems; i++) {
		seen_t key = { 0, 0, items[i].rel };
		seen_t *s = bsearch(&key, seen, (size_t)nseen, sizeof *seen, cmp_seen);
		/* duplicates of a path in seen[] are rare (a changed file); any match with the same size and mtime counts */
		int hit = 0;
		if (s) {
			while (s > seen && !strcmp(s[-1].rel, items[i].rel)) s--;
			for (; s < seen + nseen && !strcmp(s->rel, items[i].rel); s++) hit |= s->size == items[i].size && s->mtime == items[i].mtime;
		}
		if (hit) { skipped++; continue; }
		items[keep++] = items[i];
	}
	say("%s: %d audio files, %d already done, %d to do (excluding %d top folders), %d threads, %.2f s scan", root, nitems, skipped, keep,
	    nexcl, threads, now() - t_start);
	nitems = max >= 0 && max < keep ? max : keep;

	pthread_t *tid = calloc((size_t)threads, sizeof *tid);
	for (int i = 0; i < threads; i++) pthread_create(&tid[i], NULL, run, NULL);
	for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);
	say("done: %d ok, %d errors, %.1f s", done_ok, done_err, now() - t_start);
	fclose(f_index);
	fclose(f_err);
	close(fd_bin);
	return 0;
}
