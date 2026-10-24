#!/bin/sh
# skiptest.sh: tests vibed's handling of skips without touching the real player or the speakers.
#
# usage: tools/skiptest.sh "PATH UNDER THE LIBRARY" [STATE_DIR]
#
# It runs itself again inside a private D-Bus session (dbus-run-session): a second VLC there, on dummy audio and video
# outputs, plays the given song; a test vibed watches only that VLC, with its own state (default
# ~/.cache/phivibe/skiptest, seeded with copies of the live play log and network). Four times Next is pressed over MPRIS
# and the playing track is sampled every 50 ms for 4 s, then every track that played since that Next must still be in
# the playlist (vibed must never remove a playing track); before the fourth, a track is lined up by hand (MPRIS AddTrack)
# right after the current one, which vibed must not replace. The card serves one session at a time: stop the live
# daemon first (systemctl --user stop phivibe) and start it again afterwards.
set -u
song=$1
T=${2:-$HOME/.cache/phivibe/skiptest}
here=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${SKIPTEST_INNER:-}" ]; then
	mkdir -p "$T"
	rm -f "$T/events.tsv" "$T/pref.bin" "$T/pending" "$T/status.txt" "$T/vibed.log"
	cp "$HOME/.local/share/phivibe/events.tsv" "$HOME/.local/share/phivibe/pref.bin" "$T/" 2>/dev/null
	SKIPTEST_INNER=1 exec dbus-run-session -- sh "$0" "$song" "$T"
fi
LIB=$(cat "$HOME/.config/phivibe/library")
V=org.mpris.MediaPlayer2.vlc
O=/org/mpris/MediaPlayer2
now() { date +%s.%N; }
url() { busctl --user get-property $V $O org.mpris.MediaPlayer2.Player Metadata 2>/dev/null | grep -o '"xesam:url" s "[^"]*"' | sed 's/^"xesam:url" s "//; s/"$//; s|.*/||'; }
tid() { busctl --user get-property $V $O org.mpris.MediaPlayer2.Player Metadata 2>/dev/null | grep -o '/org/videolan/vlc/playlist/[0-9]*' | head -1; }
exists() { busctl --user call $V $O org.mpris.MediaPlayer2.TrackList GetTracksMetadata ao 1 "$1" > /dev/null 2>&1; }
cvlc -I dummy --control dbus --aout adummy --vout vdummy --no-media-library --no-one-instance "$LIB/$song" > "$T/vlc.log" 2>&1 &
vlc=$!
i=0; until busctl --user status $V > /dev/null 2>&1 || [ $i -gt 50 ]; do sleep 0.2; i=$((i+1)); done
echo "test VLC on a private bus: $(url)"
"$here/build/vibed" -d "$T" -b $V > "$T/vibed.log" 2>&1 &
vd=$!
i=0; until grep -q 'queued' "$T/vibed.log" || [ $i -gt 150 ]; do sleep 0.2; i=$((i+1)); done
sleep 4
for n in 1 2 3 4; do
	if [ $n = 4 ]; then
		curid=$(busctl --user get-property $V $O org.mpris.MediaPlayer2.Player Metadata | grep -o '/org/videolan/vlc/playlist/[0-9]*' | head -1)
		other=$(busctl --user get-property $V $O org.mpris.MediaPlayer2.Player Metadata | grep -o '"xesam:url" s "[^"]*"' | sed 's/^"xesam:url" s "//; s/"$//')
		busctl --user call $V $O org.mpris.MediaPlayer2.TrackList AddTrack sob "$other" "$curid" false
		echo "skip 4: the playing track lined up again by hand after $curid"
		sleep 2
	fi
	before=$(url)
	t0=$(now)
	busctl --user call $V $O org.mpris.MediaPlayer2.Player Next
	echo "skip $n: Next on $before"
	last=$before
	seen=$(tid)
	k=0
	while [ $k -lt 80 ]; do
		u=$(url)
		if [ "$u" != "$last" ]; then t=$(tid); printf '  +%6.3f s  now playing %s (%s)\n' "$(echo "$(now) - $t0" | bc)" "$u" "$t"; last=$u; seen="$seen $t"; fi
		sleep 0.05
		k=$((k+1))
	done
	# every track that played since this Next is still in the playlist: nothing playing was removed
	kept=0; gone=0
	for t in $seen; do if exists "$t"; then kept=$((kept+1)); else gone=$((gone+1)); echo "  REMOVED: $t"; fi; done
	echo "  tracks that played since Next still in the playlist: $kept, removed: $gone; playing now: $(url)"
	sleep 6
done
kill -INT $vd; sleep 1; kill $vlc
wait 2>/dev/null
echo "--- test vibed log"
grep -v 'learned from' "$T/vibed.log"
