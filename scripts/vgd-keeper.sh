#!/bin/sh
# vgd-keeper.sh: keeps the card's GEMM daemon running. Every 30 s it reads vgd's heartbeat in the shared host-memory
# window (part at 1792 MiB, offset 256); when the heartbeat has not moved and the card answers ssh, it copies the
# card binary to /tmp/vibe (the card's /tmp is tmpfs: gone after a card reboot) and starts it detached.
# Environment: VGD_BIN (default ~/.local/share/phivibe/vgd-card), PHI_SSH_PORT (2222), PHI_SSH_KEY (~/.ssh/phi_ed25519).
bin=${VGD_BIN:-$HOME/.local/share/phivibe/vgd-card}
port=${PHI_SSH_PORT:-2222}
key=${PHI_SSH_KEY:-$HOME/.ssh/phi_ed25519}
shm=/dev/shm/phi-hostmem
at=$((1792 * 1048576 + 256))
card() {
	ssh -p "$port" -i "$key" -o UserKnownHostsFile="$HOME/.ssh/known_hosts_phi" -o StrictHostKeyChecking=accept-new \
		-o BatchMode=yes -o ConnectTimeout=5 -o LogLevel=ERROR root@127.0.0.1 "$@"
}
beat() { [ -r "$shm" ] && od -An -tu8 -j "$at" -N8 "$shm" 2>/dev/null | tr -d ' '; }
last=$(beat)
while :; do
	sleep 30
	now=$(beat)
	if [ -n "$now" ] && [ "$now" != "$last" ]; then last=$now; continue; fi
	# no heartbeat: is the card up?
	if ! card true 2>/dev/null; then last=$now; continue; fi
	if card 'pidof vgd >/dev/null'; then last=$(beat); continue; fi   # running but not beating yet: give it a round
	echo "vgd-keeper: starting vgd on the card"
	card 'mkdir -p /tmp/vibe && cat > /tmp/vibe/vgd.new && chmod +x /tmp/vibe/vgd.new && mv /tmp/vibe/vgd.new /tmp/vibe/vgd' < "$bin" &&
		card 'cd /tmp/vibe && setsid nohup ./vgd > /tmp/vibe/vgd.log 2>&1 < /dev/null &' &&
		sleep 5 && card 'tail -1 /tmp/vibe/vgd.log'
	last=$(beat)
done
