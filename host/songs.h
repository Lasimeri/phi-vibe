/* songs.h: the extracted library (vibe-mel's index.tsv and mel.bin): one entry per song, newest record per path. */
#ifndef SONGS_H
#define SONGS_H

#include <stdint.h>

typedef struct {
	char *rel;            /* path under the library root */
	int64_t rec;          /* record in mel.bin */
	int64_t size, mtime;
	int dur_ms;
	int ncrops;
	int start_ms[32];     /* each crop's start */
	int folder;           /* index of its directory in songs.folders */
	int artist;           /* index of the artist parsed from "Artist - Title.ext", -1 when the name has no " - " */
} song;

typedef struct {
	song *s;
	int n;
	char **folders;       /* distinct directories */
	int nfolders;
	char **artists;       /* distinct artists, lower-cased */
	int nartists;
	const uint8_t *mel;   /* mel.bin mapped read-only */
	uint64_t mel_size;
} songs;

/* dir: ~/.cache/phivibe/lib when NULL. 0, or -1 with a message. */
int songs_load(songs *L, const char *dir);
void songs_free(songs *L);

/* crop c of song i: 128 frames x 64 bands of u8 dB codes */
const uint8_t *songs_crop(const songs *L, int i, int c);

/* the song whose path (under the root) is rel, or -1 */
int songs_find(const songs *L, const char *rel);

uint64_t songs_hash(const char *s);

#endif
