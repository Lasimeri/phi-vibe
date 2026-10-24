/*
 * mpris.h: VLC over MPRIS on the session bus (sd-bus). Reads the player's state and track list; adds a track right
 * after a given one and removes tracks it added. Never touches playback itself (play, pause, seek, volume).
 *
 * VLC 3 quirk (modules/control/dbus/dbus_tracklist.c): the TrackList.Tracks property is built from the playlist's
 * play-order array, which VLC rebuilds only when it moves to another item; right after a track change it is exact,
 * after an AddTrack it is stale. New items are learnt from VLC's TrackAdded signals instead.
 */
#ifndef MPRIS_H
#define MPRIS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
	char status[16];        /* Playing, Paused, Stopped */
	char trackid[128];      /* mpris:trackid */
	char url[8192];         /* xesam:url, percent-encoded */
	int64_t length_us, pos_us;
} mp_state;

typedef struct {
	char trackid[128];
	char url[8192];
} mp_added;

typedef struct mpris mpris;

mpris *mp_open(void);
void mp_close(mpris *m);

/* 1 with the state of VLC, 0 when VLC is not on the bus, -1 when the bus failed */
int mp_read(mpris *m, mp_state *s);

/* the track list as VLC last built it: up to max ids into ids (each 128 bytes); the count, or -1 */
int mp_tracks(mpris *m, char (*ids)[128], int max);

/* adds uri right after track `after` (not made current); its new track id into id. 0, or -1. */
int mp_add_after(mpris *m, const char *uri, const char *after, char *id, size_t idlen);
int mp_remove(mpris *m, const char *trackid);

/* tracks VLC announced as added since the last call (any source); returns how many, at most max */
int mp_take_added(mpris *m, mp_added *out, int max);

/* the url of a track id: 0, or -1 when VLC no longer has it */
int mp_track_url(mpris *m, const char *trackid, char *url, size_t n);

/* a file path as a file:// URI (RFC 3986: unreserved characters and '/' kept, every other byte %XX) */
void mp_uri(const char *path, char *out, size_t n);
/* percent-decodes a file:// URI into a path; 0, or -1 when it is not a file URI */
int mp_path(const char *uri, char *out, size_t n);

#endif
