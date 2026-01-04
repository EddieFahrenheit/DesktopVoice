from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .config import AppConfig


GEMINI_URL = "https://gemini.google.com/"
CHATGPT_URL = "https://chatgpt.com/"


class BrowserController:
    """
    Owns Playwright + a persistent Chrome profile.

    Persistent profile matters because:
    - you stay logged in to Gemini/ChatGPT
    - mic permissions persist
    - you can keep the window open between commands
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._pw = None
        self._browser = None
        self._context = None
        self._attached_via_cdp = False

    def __enter__(self) -> "BrowserController":
        self._pw = sync_playwright().start()

        # If CHROME_CDP_URL is set, attach to an already-running Chrome instance (started
        # with --remote-debugging-port=9222). This avoids Playwright "launch" automation
        # flags, which can cause Google login pages to behave differently.
        if self._cfg.chrome_cdp_url:
            self._attached_via_cdp = True
            try:
                self._browser = self._pw.chromium.connect_over_cdp(self._cfg.chrome_cdp_url)
            except Exception:
                # If nothing is listening yet, auto-start a CDP-enabled Chrome and retry.
                self._autostart_cdp_chrome()
                self._browser = self._pw.chromium.connect_over_cdp(self._cfg.chrome_cdp_url)
            if self._browser.contexts:
                self._context = self._browser.contexts[0]
            else:
                self._context = self._browser.new_context()
            return self

        # Otherwise, launch a dedicated persistent context (acts like a “real” Chrome profile).
        # `channel="chrome"` uses your installed Google Chrome app.
        Path(self._cfg.profile_dir).mkdir(parents=True, exist_ok=True)
        self._context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self._cfg.profile_dir),
            channel=self._cfg.browser_channel or None,
            headless=False,
            args=[
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # In CDP mode, we generally *do not* close the user's Chrome instance on exit.
        # We just disconnect by stopping Playwright.
        if self._context is not None and not self._attached_via_cdp:
            self._context.close()
        if self._browser is not None and not self._attached_via_cdp:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    def open_gemini_and_click_mic(self) -> None:
        page = self._get_or_open(GEMINI_URL)
        self._click_first_matching_button(
            page,
            patterns=[
                r"use microphone",
                r"microphone",
                r"voice",
            ],
        )

    def open_chatgpt_and_click_mic(self) -> None:
        page = self._get_or_open(CHATGPT_URL)
        self._click_first_matching_button(
            page,
            patterns=[
                r"voice",
                r"start voice",
                r"microphone",
            ],
        )

    def _get_or_open(self, url: str):
        assert self._context is not None

        origin = self._origin(url)

        # 1) Reuse an existing tab if one is already on the right site.
        for page in self._context.pages:
            if page.url.startswith(origin):
                page.bring_to_front()
                return page

        # 2) Avoid creating a "pointless empty tab": when Chrome starts, it often has an
        #    initial New Tab / about:blank page. Reuse that page by navigating it instead
        #    of opening a new tab.
        for page in self._context.pages:
            if page.url in {"about:blank", "chrome://newtab/", "chrome://new-tab-page/"}:
                page.goto(url, wait_until="domcontentloaded")
                page.bring_to_front()
                return page

        # 3) Otherwise, open a new tab.
        page = self._context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        page.bring_to_front()
        return page

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _autostart_cdp_chrome(self) -> None:
        """
        Start a local Chrome instance with a remote debugging port, then wait until
        the CDP endpoint is reachable.

        Note: if normal Chrome is already running with the same profile, Chrome may
        ignore these flags. In that case, the port won't open and we tell you to quit
        Chrome and retry.
        """
        assert self._cfg.chrome_cdp_url is not None
        parsed = urlparse(self._cfg.chrome_cdp_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 9222

        if host not in {"127.0.0.1", "localhost"}:
            raise RuntimeError("Refusing to auto-start CDP Chrome unless CHROME_CDP_URL uses 127.0.0.1/localhost.")

        chrome_bin = self._find_chrome_bin()
        user_data_dir = str(self._cfg.chrome_cdp_user_data_dir)
        profile_dir = self._cfg.chrome_cdp_profile_directory

        args = [
            chrome_bin,
            f"--remote-debugging-address={host}",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            f"--profile-directory={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        # Fire-and-forget. We don't want to block the listener; we just need CDP up.
        subprocess.Popen(  # noqa: S603
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ},
        )

        self._wait_for_cdp(host=host, port=port, timeout_s=10.0)

    @staticmethod
    def _wait_for_cdp(*, host: str, port: int, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        url = f"http://{host}:{port}/json/version"
        last_err: Exception | None = None

        while time.time() < deadline:
            try:
                with urlopen(url, timeout=0.5) as resp:  # noqa: S310
                    if resp.status == 200:
                        return
            except (URLError, OSError) as exc:
                last_err = exc
                time.sleep(0.2)

        raise RuntimeError(
            "Could not connect to Chrome DevTools after starting Chrome. "
            "If Chrome is already running, quit it completely (Cmd+Q) and retry."
        ) from last_err

    @staticmethod
    def _find_chrome_bin() -> str:
        """
        Find the Chrome executable path.

        macOS: prefer the standard app bundle binary.
        Linux: fall back to PATH lookups.
        """
        if sys.platform == "darwin":
            mac_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            if Path(mac_bin).exists():
                return mac_bin

        for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            path = shutil.which(candidate)
            if path:
                return path

        raise RuntimeError("Could not find Google Chrome. Install it or add it to PATH.")

    def _click_first_matching_button(self, page, patterns: list[str]) -> None:
        """
        Try a few robust “by accessible name” matches first.
        These are usually more stable than CSS selectors across UI changes.
        """
        for pattern in patterns:
            locator = page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
            try:
                locator.first.click(timeout=2500)
                return
            except PlaywrightTimeoutError:
                continue

        raise RuntimeError("Could not find the microphone/voice button on the page (UI may have changed).")
