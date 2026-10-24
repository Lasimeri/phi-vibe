/* mpris.c: see mpris.h. */
#define _GNU_SOURCE
#include "mpris.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <systemd/sd-bus.h>
#include <time.h>

#define PATH "/org/mpris/MediaPlayer2"
#define PLAYER "org.mpris.MediaPlayer2.Player"
#define TRACKLIST "org.mpris.MediaPlayer2.TrackList"
#define PREFIX "org.mpris.MediaPlayer2.vlc"
#define ADDED_MAX 64

struct mpris {
	sd_bus *bus;
	sd_bus_slot *slot;
	char name[256];              /* VLC's bus name */
	mp_added added[ADDED_MAX];   /* TrackAdded signals not yet taken */
	sd_bus_slot *slot2;
	int changed;                 /* a PropertiesChanged arrived since the last mp_wait */
	int fixed;                   /* name was given (mp_set_name): only that player */
	int nadded;
};

static void copy(char *dst, size_t n, const char *src)
{
	snprintf(dst, n, "%s", src ? src : "");
}

/* the url (and trackid when present) out of an a{sv} metadata dict */
static void read_meta(sd_bus_message *msg, char *url, size_t ulen, char *id, size_t idlen, int64_t *length)
{
	if (sd_bus_message_enter_container(msg, 'a', "{sv}") <= 0) return;
	while (sd_bus_message_enter_container(msg, 'e', "sv") > 0) {
		const char *key, *sig;
		char type;
		if (sd_bus_message_read(msg, "s", &key) < 0) break;
		if (sd_bus_message_peek_type(msg, &type, &sig) < 0 || type != 'v') break;
		if (sd_bus_message_enter_container(msg, 'v', sig) < 0) break;
		if (!strcmp(sig, "s") || !strcmp(sig, "o")) {
			const char *s = NULL;
			sd_bus_message_read_basic(msg, sig[0], &s);
			if (!strcmp(key, "xesam:url") && url) copy(url, ulen, s);
			else if (!strcmp(key, "mpris:trackid") && id) copy(id, idlen, s);
		} else if (!strcmp(sig, "x") || !strcmp(sig, "t")) {
			int64_t v = 0;
			sd_bus_message_read_basic(msg, sig[0], &v);
			if (!strcmp(key, "mpris:length") && length) *length = v;
		} else sd_bus_message_skip(msg, sig);
		sd_bus_message_exit_container(msg);
		sd_bus_message_exit_container(msg);
	}
	sd_bus_message_exit_container(msg);
}

/* TrackAdded(a{sv} metadata, o trackid), from any interface name (VLC 3 sends it under "MediaPlayer2.TrackList") */
static int on_added(sd_bus_message *msg, void *ud, sd_bus_error *err)
{
	(void)err;
	mpris *m = ud;
	mp_added a = { "", "" };
	read_meta(msg, a.url, sizeof a.url, NULL, 0, NULL);
	const char *id = NULL;
	if (sd_bus_message_read(msg, "o", &id) >= 0 && id) copy(a.trackid, sizeof a.trackid, id);
	if (!a.trackid[0]) return 0;
	if (m->nadded == ADDED_MAX) { memmove(m->added, m->added + 1, sizeof m->added[0] * (ADDED_MAX - 1)); m->nadded--; }
	m->added[m->nadded++] = a;
	return 0;
}

static int on_changed(sd_bus_message *msg, void *ud, sd_bus_error *err)
{
	(void)msg; (void)err;
	((mpris *)ud)->changed = 1;
	return 0;
}

mpris *mp_open(void)
{
	mpris *m = calloc(1, sizeof *m);
	if (sd_bus_open_user(&m->bus) < 0) { free(m); return NULL; }
	if (sd_bus_add_match(m->bus, &m->slot, "type='signal',path='" PATH "',member='TrackAdded'", on_added, m) < 0 ||
	    sd_bus_add_match(m->bus, &m->slot2, "type='signal',path='" PATH "',interface='org.freedesktop.DBus.Properties',member='PropertiesChanged'", on_changed, m) < 0) {
		sd_bus_unref(m->bus);
		free(m);
		return NULL;
	}
	return m;
}

void mp_close(mpris *m)
{
	if (!m) return;
	sd_bus_slot_unref(m->slot);
	sd_bus_slot_unref(m->slot2);
	sd_bus_flush_close_unref(m->bus);
	free(m);
}

static void pump(mpris *m)
{
	while (sd_bus_process(m->bus, NULL) > 0) {}
}

void mp_set_name(mpris *m, const char *name)
{
	copy(m->name, sizeof m->name, name);
	m->fixed = 1;
}

int mp_wait(mpris *m, double seconds)
{
	m->changed = 0;
	pump(m);
	if (!m->changed) sd_bus_wait(m->bus, (uint64_t)(seconds * 1e6));
	pump(m);
	return m->changed;
}

/* the player's bus name into m->name (the given one, or the first VLC); 1, 0 none, -1 bus error */
static int find_vlc(mpris *m)
{
	if (m->fixed) {
		sd_bus_error e = SD_BUS_ERROR_NULL;
		sd_bus_message *rep = NULL;
		int has = 0;
		if (sd_bus_call_method(m->bus, "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameHasOwner", &e, &rep, "s", m->name) < 0) {
			sd_bus_error_free(&e);
			return -1;
		}
		sd_bus_message_read(rep, "b", &has);
		sd_bus_message_unref(rep);
		return has ? 1 : 0;
	}
	sd_bus_error e = SD_BUS_ERROR_NULL;
	sd_bus_message *r = NULL;
	char **names = NULL;
	int found = 0;
	if (sd_bus_call_method(m->bus, "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "ListNames", &e, &r, "") < 0) {
		sd_bus_error_free(&e);
		return -1;
	}
	if (sd_bus_message_read_strv(r, &names) >= 0 && names) {
		for (char **p = names; *p; p++) {
			if (!found && !strncmp(*p, PREFIX, strlen(PREFIX))) { copy(m->name, sizeof m->name, *p); found = 1; }
			free(*p);
		}
		free(names);
	}
	sd_bus_message_unref(r);
	return found;
}

int mp_read(mpris *m, mp_state *s)
{
	pump(m);
	memset(s, 0, sizeof *s);
	int r = find_vlc(m);
	if (r <= 0) return r;
	sd_bus_error e = SD_BUS_ERROR_NULL;
	char *status = NULL;
	if (sd_bus_get_property_string(m->bus, m->name, PATH, PLAYER, "PlaybackStatus", &e, &status) < 0) { sd_bus_error_free(&e); return 0; }
	copy(s->status, sizeof s->status, status);
	free(status);
	sd_bus_message *msg = NULL;
	if (sd_bus_get_property(m->bus, m->name, PATH, PLAYER, "Metadata", &e, &msg, "a{sv}") >= 0) {
		read_meta(msg, s->url, sizeof s->url, s->trackid, sizeof s->trackid, &s->length_us);
		sd_bus_message_unref(msg);
	}
	sd_bus_error_free(&e);
	sd_bus_get_property_trivial(m->bus, m->name, PATH, PLAYER, "Position", &e, 'x', &s->pos_us);
	sd_bus_error_free(&e);
	int sh = 0;
	if (sd_bus_get_property_trivial(m->bus, m->name, PATH, PLAYER, "Shuffle", &e, 'b', &sh) >= 0) s->shuffle = sh;
	sd_bus_error_free(&e);
	return 1;
}

int mp_tracks(mpris *m, char (*ids)[128], int max)
{
	sd_bus_error e = SD_BUS_ERROR_NULL;
	sd_bus_message *msg = NULL;
	if (!m->name[0] || sd_bus_get_property(m->bus, m->name, PATH, TRACKLIST, "Tracks", &e, &msg, "ao") < 0) { sd_bus_error_free(&e); return -1; }
	int n = 0;
	if (sd_bus_message_enter_container(msg, 'a', "o") > 0) {
		const char *id;
		while (sd_bus_message_read(msg, "o", &id) > 0) if (n < max) copy(ids[n++], 128, id);
		sd_bus_message_exit_container(msg);
	}
	sd_bus_message_unref(msg);
	return n;
}

static double mono(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

int mp_add_after(mpris *m, const char *uri, const char *after, int play, char *id, size_t idlen)
{
	sd_bus_error e = SD_BUS_ERROR_NULL;
	pump(m);
	if (!m->name[0] || sd_bus_call_method(m->bus, m->name, PATH, TRACKLIST, "AddTrack", &e, NULL, "sob", uri, after, play ? 1 : 0) < 0) {
		sd_bus_error_free(&e);
		return -1;
	}
	/* VLC announces the new item with TrackAdded: wait for the one with our url (a tenth of a second at most) */
	const double t0 = mono();
	while (mono() - t0 < 0.1) {
		for (int i = m->nadded - 1; i >= 0; i--)
			if (!strcmp(m->added[i].url, uri)) {
				copy(id, idlen, m->added[i].trackid);
				memmove(m->added + i, m->added + i + 1, sizeof m->added[0] * (size_t)(m->nadded - i - 1));
				m->nadded--;
				return 0;
			}
		if (sd_bus_process(m->bus, NULL) == 0) sd_bus_wait(m->bus, 50000);
	}
	/* no signal (VLC 3 does not always send it): its play-list ids only grow, so the newest item with our url
	 * after `after` is ours. Probe upward until 64 ids in a row are unused. */
	const char *num = strrchr(after, '/');
	long first = num ? strtol(num + 1, NULL, 10) + 1 : 1, best = -1;
	char tid[128], u[8192];
	for (long i = first, misses = 0; misses < 64 && i < first + 100000; i++) {
		snprintf(tid, sizeof tid, "/org/videolan/vlc/playlist/%ld", i);
		if (mp_track_url(m, tid, u, sizeof u)) { misses++; continue; }
		misses = 0;
		if (!strcmp(u, uri)) best = i;
	}
	if (best >= 0) { snprintf(id, idlen, "/org/videolan/vlc/playlist/%ld", best); return 0; }
	id[0] = 0;
	return 0;   /* added, id unknown: it will not be removed by us */
}

int mp_remove(mpris *m, const char *trackid)
{
	sd_bus_error e = SD_BUS_ERROR_NULL;
	int r = m->name[0] ? sd_bus_call_method(m->bus, m->name, PATH, TRACKLIST, "RemoveTrack", &e, NULL, "o", trackid) : -1;
	sd_bus_error_free(&e);
	return r < 0 ? -1 : 0;
}

int mp_take_added(mpris *m, mp_added *out, int max)
{
	pump(m);
	int n = m->nadded < max ? m->nadded : max;
	memcpy(out, m->added, sizeof *out * (size_t)n);
	memmove(m->added, m->added + n, sizeof m->added[0] * (size_t)(m->nadded - n));
	m->nadded -= n;
	return n;
}

int mp_track_url(mpris *m, const char *trackid, char *url, size_t n)
{
	sd_bus_error e = SD_BUS_ERROR_NULL;
	sd_bus_message *msg = NULL;
	if (!m->name[0] || sd_bus_call_method(m->bus, m->name, PATH, TRACKLIST, "GetTracksMetadata", &e, &msg, "ao", 1, trackid) < 0) {
		sd_bus_error_free(&e);
		return -1;
	}
	url[0] = 0;
	if (sd_bus_message_enter_container(msg, 'a', "a{sv}") > 0) {
		read_meta(msg, url, n, NULL, 0, NULL);
		sd_bus_message_exit_container(msg);
	}
	sd_bus_message_unref(msg);
	return url[0] ? 0 : -1;
}

void mp_uri(const char *path, char *out, size_t n)
{
	size_t o = (size_t)snprintf(out, n, "file://");
	for (const unsigned char *p = (const unsigned char *)path; *p && o + 4 < n; p++) {
		if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || strchr("-._~/", *p)) out[o++] = (char)*p;
		else o += (size_t)snprintf(out + o, n - o, "%%%02X", *p);
	}
	out[o < n ? o : n - 1] = 0;
}

static int hexval(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

int mp_path(const char *uri, char *out, size_t n)
{
	if (strncmp(uri, "file://", 7)) return -1;
	size_t o = 0;
	for (const char *p = uri + 7; *p && o + 1 < n; p++) {
		if (*p == '%' && hexval(p[1]) >= 0 && hexval(p[2]) >= 0) { out[o++] = (char)(hexval(p[1]) * 16 + hexval(p[2])); p += 2; }
		else out[o++] = *p;
	}
	out[o] = 0;
	return 0;
}
