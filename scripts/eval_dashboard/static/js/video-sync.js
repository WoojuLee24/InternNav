/* DualPlayer (from VLN-Challenge dev/jay) — keeps up to two <video> elements frame-synced under one
   slider, with bookmark jumps and play / pause / stop transport. Frame i maps to time i/fps.
   With no videos (raw-JSON episodes) it is a plain frame timer driving onFrame. */
class DualPlayer {
  constructor(videos, fps, numFrames, bookmarks, onFrame) {
    this.videos = videos.filter(Boolean);   // 0, 1 or 2 real <video> elements
    this.fps = fps || 6;
    this.n = Math.max(1, numFrames || 1);
    this.bookmarks = bookmarks || [];
    this.onFrame = onFrame || null;          // (frame) => void, for the per-frame strip
    this.frame = 0;
    this.playing = false;
    this._raf = null;
  }

  // ---------------------------------------------------------------- transport DOM
  build() {
    const last = this.n - 1;
    this.slider = el("input", { type: "range", min: 0, max: last, value: 0, step: 1 });
    this.slider.addEventListener("input", () => { this.pause(); this.seekFrame(+this.slider.value); });

    this.timeLabel = el("span.tp-time", {}, `0 / ${last}`);
    const btn = (txt, title, fn) => el("button.tp-btn", { title, onclick: fn }, txt);
    this.playBtn = btn("▶", "play (space)", () => this.toggle());

    // legend + frame counter sit on the RIGHT of the controls row (no extra vertical space)
    const legend = this.bookmarks.length ? el("div.bm-legend", {}, [
      ["right", "var(--bm-right)"], ["left", "var(--bm-left)"], ["S2 query", "var(--bm-pixel)"], ["collision", "var(--bad)"],
    ].map(([n, c]) => el("span.it", {}, [el("span.sw", { style: `background:${c}` }), n]))) : null;

    const buttons = el("div.tp-buttons", {}, [
      btn("‹", "prev frame (←)", () => this.step(-1)),
      this.playBtn,
      btn("⏸", "pause", () => this.pause()),
      btn("■", "stop / back to start", () => this.stop()),
      btn("›", "next frame (→)", () => this.step(1)),
      el("span.tp-spacer"),
      legend,
      this.timeLabel,
    ]);

    // bookmark markers: a wide transparent hit-area with a centered colored bar (easy to click)
    this.bmTrack = el("div.bookmarks");
    this.bookmarks.forEach((b) => this._addMarker(b.frame, b.type));

    const scrub = el("div.scrub", {}, [this.bmTrack, this.slider]);
    return el("div.transport", {}, [buttons, scrub]);
  }

  _addMarker(frame, type) {
    const last = this.n - 1;
    const left = last > 0 ? (frame / last) * 100 : 0;
    this.bmTrack.appendChild(el("div.bm-hit", {
      style: `left:${left}%`, title: `${type} @ frame ${frame}`,
      onclick: () => { this.pause(); this.seekFrame(frame); },
    }, el("div.bm." + type)));
  }

  // add a marker at runtime (e.g. divergence frames); de-dupes repeated clicks
  addBookmark(frame, type) {
    if (!this.bmTrack) return;
    if (this.bookmarks.some((b) => b.frame === frame && b.type === type)) return;
    this.bookmarks.push({ frame, type });
    this._addMarker(frame, type);
  }

  // replace all "diverge" (split/error) markers with the given frames
  setDivergeBookmarks(frames) {
    this.bookmarks = this.bookmarks.filter((b) => b.type !== "diverge");
    if (this.bmTrack) { clear(this.bmTrack); this.bookmarks.forEach((b) => this._addMarker(b.frame, b.type)); }
    (frames || []).forEach((f) => this.addBookmark(f, "diverge"));
  }

  step(delta) { this.pause(); this.seekFrame(this.frame + delta); }

  // ----------------------------------------------------------------- frame control
  seekFrame(f) {
    this.frame = Math.max(0, Math.min(this.n - 1, Math.round(f)));
    const t = this.frame / this.fps;
    this.videos.forEach((v) => { try { v.currentTime = t; } catch (e) {} });
    this._sync();
  }

  _sync() {
    if (this.slider) this.slider.value = this.frame;
    if (this.timeLabel) this.timeLabel.textContent = `${this.frame} / ${this.n - 1}`;
    if (this.onFrame) this.onFrame(this.frame);
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  play() {
    if (!this.videos.length) return this._playTimer();
    if (this.frame >= this.n - 1) this.seekFrame(0);
    this.playing = true;
    this.playBtn.textContent = "⏸";
    this.videos.forEach((v) => { v.play().catch(() => {}); });
    const tick = () => {
      if (!this.playing) return;
      const lead = this.videos[0];
      this.frame = Math.min(this.n - 1, Math.round(lead.currentTime * this.fps));
      // nudge the 2nd video back in sync if it drifts > 1 frame
      if (this.videos[1]) {
        const dt = this.videos[1].currentTime - lead.currentTime;
        if (Math.abs(dt) > 1 / this.fps) { try { this.videos[1].currentTime = lead.currentTime; } catch (e) {} }
      }
      this._sync();
      if (lead.ended || this.frame >= this.n - 1) { this.pause(); return; }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  // no videos: advance one frame every 1/fps s
  _playTimer() {
    if (this.frame >= this.n - 1) this.seekFrame(0);
    this.playing = true;
    this.playBtn.textContent = "⏸";
    this._timer = setInterval(() => {
      if (this.frame >= this.n - 1) { this.pause(); return; }
      this.seekFrame(this.frame + 1);
    }, 1000 / this.fps);
  }

  pause() {
    this.playing = false;
    if (this.playBtn) this.playBtn.textContent = "▶";
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    if (this._raf) cancelAnimationFrame(this._raf);
    this.videos.forEach((v) => v.pause());
  }

  stop() { this.pause(); this.seekFrame(0); }

  destroy() { this.pause(); }
}
