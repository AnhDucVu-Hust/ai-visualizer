# Endpoint Configuration Guide (macOS + Windows)

This app uses frontend runtime settings, so customers can change backend endpoint without reinstalling.

## Customer steps (same for macOS and Windows)

1. Open the app.
2. At the footer, find **Backend** and the endpoint input box.
3. Enter new endpoint, for example: `https://api.your-company.com`
4. Click **Save endpoint**.
5. Continue using the app (no rebuild/reinstall needed).

## Reset to default

- Click **Reset** in the footer.
- Default endpoint is `http://127.0.0.1:8000` (or from `VITE_API_BASE_URL` at build time).

## Notes for support team

- Endpoint is stored locally in browser storage (WebView localStorage) per installed app profile.
- Keep URL format as `http://...` or `https://...`.
- Do not leave trailing slash, for example use `https://api.company.com` (not `https://api.company.com/`).
