# AMMS Setup — Schritt für Schritt

Diese Anleitung bringt den Bot zum Laufen. Sie ist für Leute geschrieben,
die noch nie auf einem Server gearbeitet haben. Du kopierst Befehle und
schaust, ob die erwartete Antwort kommt — mehr nicht.

**Was du am Ende hast:**
- Einen kleinen Linux-Server, der 24/7 läuft und ca. 4 €/Monat kostet
- Den Bot, der dort vollautomatisch Paper-Trades macht
- Telegram-Benachrichtigungen aufs Handy bei jeder Order

**Zeit gesamt:** etwa 60–90 Minuten beim ersten Mal.

**Was du nicht installieren musst:** Python, Docker, Git auf deinem
Laptop. Das alles läuft auf dem Server. Dein Laptop ist nur die Fernbedienung.

---

## Übersicht

```
1. Accounts holen   (Alpaca hast du; Telegram-Bot anlegen)
2. VPS bestellen    (Hetzner Cloud, ca. 4 €/Monat)
3. SSH-Schlüssel    (einmalig auf deinem Laptop)
4. Server verbinden (per SSH vom Windows-Terminal)
5. Docker installieren (ein einziger Befehl)
6. Code holen       (`git clone`)
7. Keys eintragen   (.env-Datei)
8. Erster Lauf      (Dry-Run zum Testen)
9. Live gehen       (echte Paper-Trades)
10. Alltag & Hilfe
```

---

## 1. Accounts vorbereiten

### Alpaca (haste schon)

Du hast deinen Paper-Account schon. Hol dir nur die Schlüssel:

1. Login auf https://app.alpaca.markets/paper/dashboard/overview
2. Rechts oben → „API Keys" → „Generate New Key"
3. **Speicher beide Werte sicher ab** (z.B. in einem Passwort-Manager):
   - `API Key ID`
   - `Secret Key`
   - Das Secret wird nur einmal angezeigt!

### Telegram-Bot (optional, aber empfehlenswert)

Damit du Push-Nachrichten aufs Handy bekommst, wenn der Bot was tut.

1. Telegram-App öffnen → Suche `@BotFather`
2. Schreib `/newbot`
3. Wähl einen Namen (egal welchen, z.B. „mein-amms")
4. Wähl einen Username, der auf `bot` endet (z.B. `mein_amms_bot`)
5. BotFather schickt dir den **Token** — sieht aus wie `123456:ABC-DEF...`.
   **Diesen Token sicher abspeichern.**
6. Suche jetzt deinen neuen Bot in Telegram und schicke ihm `/start`
7. Öffne in deinem Browser (Token einsetzen!):
   ```
   https://api.telegram.org/bot<DEIN_TOKEN>/getUpdates
   ```
   In der JSON-Antwort suchst du `"chat":{"id":12345678,...}` — die Zahl
   ist deine **Chat-ID**. **Auch abspeichern.**

Wenn du das überspringst, läuft der Bot trotzdem — er schweigt halt einfach.

---

## 2. VPS bei Hetzner bestellen

Hetzner Cloud ist günstig, schnell und in Deutschland gehostet.

1. https://accounts.hetzner.com/signUp — Account anlegen (Personalausweis
   kann verlangt werden bei der ersten Bestellung).
2. Nach Login → Projekt erstellen (Name egal, z.B. „amms").
3. Im Projekt → „Add Server".
4. Einstellungen:
   - **Location:** Falkenstein oder Nürnberg (Deutschland)
   - **Image:** Ubuntu 24.04
   - **Type:** CX22 (2 vCPU, 4 GB RAM, ca. 4 €/Monat)
   - **SSH Key:** sieh nächster Schritt — den fügst du gleich hinzu
   - **Name:** z.B. `amms-vps`
5. Bestelle noch nicht. **Erst SSH-Schlüssel anlegen → Schritt 3.**

---

## 3. SSH-Schlüssel auf deinem Laptop erzeugen

Damit der Server dich erkennt, ohne dass du jedesmal ein Passwort tippen
musst. **Nur einmal nötig.**

1. Auf deinem Windows-Laptop: **Windows Terminal** öffnen (in den Startmenü
   tippen — ist auf Win 11 dabei; falls nicht: kostenlos im Microsoft Store).
2. Tippe folgenden Befehl und drück Enter:
   ```powershell
   ssh-keygen -t ed25519 -C "amms@meinlaptop"
   ```
3. Drück bei jeder Frage einfach **Enter** (Standardpfad ist okay; Passwort
   für den Schlüssel kannst du leer lassen, ist privat genug).
4. Zeig den öffentlichen Schlüssel an:
   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub
   ```
5. Markier die ganze Zeile (beginnt mit `ssh-ed25519 AAAA...`), Rechtsklick →
   kopieren.

Geh zurück zu Hetzner:
1. Bei der Server-Erstellung → „SSH keys" → „Add SSH key" → einfügen → Name
   geben (z.B. „mein-laptop") → speichern.
2. Den Haken bei diesem Schlüssel setzen.
3. **Jetzt** auf „Create & Buy now" klicken.
4. Nach ca. 30 Sekunden bekommst du eine **IPv4-Adresse** angezeigt
   (z.B. `49.12.34.56`). **Die merken.**

---

## 4. Mit dem Server verbinden

Im Windows Terminal (das du gerade offen hast):

```powershell
ssh root@49.12.34.56
```

(IP durch deine ersetzen.)

Beim allerersten Mal fragt SSH: „Are you sure you want to continue
connecting (yes/no)?" → tippe `yes` und Enter.

Wenn alles klappt, siehst du sowas wie:
```
root@amms-vps:~#
```

**Das bedeutet: du bist auf dem Server.** Alle Befehle, die du jetzt tippst,
laufen auf dem Server, nicht auf deinem Laptop.

---

## 5. Docker installieren

Auf dem Server (du bist eingeloggt):

```bash
curl -fsSL https://get.docker.com | sh
```

Das dauert 1–2 Minuten. Am Ende siehst du sowas wie „Docker has been
successfully installed". Verifiziere:

```bash
docker --version
docker compose version
```

Beide Befehle sollten eine Versionsnummer zeigen. Wenn ja, perfekt.

---

## 6. Den Bot-Code holen

Auf dem Server:

```bash
apt-get install -y git
cd /opt
git clone https://github.com/zuckerpirat/AMMS.git amms
cd amms
git checkout claude/paper-trading-bot-design-VJPCL
```

(Den Branch-Namen kannst du später auf `main` ändern, wenn alles
zusammengeführt ist.)

Du bist jetzt in `/opt/amms`. Schau kurz nach, was drin ist:

```bash
ls
```

Du solltest u.a. `docker-compose.yml`, `.env.example`, `config.example.yaml`
und einen `src/`-Ordner sehen.

---

## 7. Schlüssel und Strategie eintragen

### `.env` anlegen

```bash
cp .env.example .env
nano .env
```

`nano` ist ein einfacher Editor im Terminal. Fülle aus:

```
ALPACA_API_KEY=DEIN_ALPACA_KEY_HIER
ALPACA_API_SECRET=DEIN_ALPACA_SECRET_HIER
ALPACA_BASE_URL=https://paper-api.alpaca.markets

TELEGRAM_BOT_TOKEN=DEIN_TELEGRAM_TOKEN_HIER
TELEGRAM_CHAT_ID=DEINE_CHAT_ID_HIER
```

Wenn du Telegram weglässt, lass die beiden Zeilen einfach leer.

Speichern in nano: **Strg+O**, dann **Enter**, dann **Strg+X**.

### `config.yaml` anlegen

```bash
cp config.example.yaml config.yaml
nano config.yaml
```

Beispiel für den Start (vorsichtige Einstellungen):

```yaml
watchlist:
  - AAPL
  - MSFT
  - NVDA
  - AMD
  - GOOGL

strategy:
  name: composite
  params:
    momentum_n: 20
    momentum_min: 0.05
    rsi_max: 70
    vol_max: 0.40
    rvol_min: 1.2

risk:
  max_open_positions: 5
  max_position_pct: 0.02
  daily_loss_pct: -0.03
  max_buys_per_tick: 2

scheduler:
  tick_seconds: 60
  timezone: America/New_York
```

`max_position_pct: 0.02` heißt: jede Position höchstens 2 % deines Equity.
Bei $100k Paper-Geld sind das $2k pro Position. Konservativ.

Speichern wie vorher: **Strg+O**, **Enter**, **Strg+X**.

---

## 8. Erster Lauf — Dry-Run

Vor dem echten Trading: Trockenübung. Der Bot zeigt, was er tun *würde*,
aber plaziert keine Orders.

```bash
docker compose build
docker compose run --rm amms status
```

Wenn deine Keys stimmen, siehst du eine Tabelle mit deinem Paper-Equity
(typischerweise $100,000). Wenn nicht: Fehlermeldung lesen, `.env` prüfen.

Dann einen einzelnen Tick im Dry-Run:

```bash
docker compose run --rm amms tick
```

Du siehst eine Signale-Tabelle. Wenn alle „hold" sind: normal, kein Setup.
Wenn welche „buy" sind: der Bot würde kaufen — aber er macht's nicht, weil
es Dry-Run ist.

---

## 9. Live gehen (Paper)

Wenn der Dry-Run okay aussah, starte den autonomen Bot:

```bash
docker compose up -d
```

Das `-d` heißt „im Hintergrund". Der Bot läuft jetzt 24/7. Wenn der Server
neu startet, startet der Bot automatisch mit (Docker `restart: unless-stopped`).

**Beachte:** Standardmäßig ist das **immer noch Dry-Run** im Hintergrund.
Um echte Paper-Orders zu plazieren, ändere in `docker-compose.yml` die
Command-Zeile:

```yaml
command: ["run", "--execute"]
```

Dann:

```bash
docker compose up -d
```

Jetzt ist der Bot scharf — auf Paper. Echtes Geld kann er per Architektur
nicht anfassen (siehe `paper-api`-Schutz in der Codebase).

---

## 10. Alltag & häufige Befehle

**Auf den Server einloggen** (vom Laptop aus):
```powershell
ssh root@DEINE_IP
cd /opt/amms
```

**Status checken:**
```bash
docker compose run --rm amms status
```

**Bot-Logs in Echtzeit anschauen:**
```bash
docker compose logs -f
```

(Mit `Strg+C` raus, der Bot läuft weiter.)

**Bot stoppen:**
```bash
docker compose down
```

**Bot neu starten** (nach Config-Änderung):
```bash
docker compose down
docker compose up -d
```

**Code aktualisieren** (wenn ich neue Features gepusht hab):
```bash
cd /opt/amms
git pull
docker compose build
docker compose up -d
```

**Backtest laufen lassen:**
```bash
docker compose run --rm amms backtest \
  --from 2024-01-01 --to 2025-12-31 \
  --symbols AAPL,MSFT,NVDA --fetch \
  --output /data/trades.csv
```

Die CSV liegt dann in `/data/trades.csv` im Container — am Host findest du
sie im Docker-Volume `amms_amms-data`. Zum runterladen auf den Laptop:

```bash
docker compose cp amms:/data/trades.csv ./trades.csv
```

Dann vom Laptop:
```powershell
scp root@DEINE_IP:/opt/amms/trades.csv .
```

---

## 11. Wenn was nicht klappt

**„Permission denied (publickey)" beim SSH:**
Dein SSH-Schlüssel ist nicht bei Hetzner hinterlegt. Schritt 3 nochmal,
oder Hetzner-Webconsole nutzen.

**„Refusing to start: ALPACA_BASE_URL must point at the paper endpoint":**
In `.env` muss `ALPACA_BASE_URL=https://paper-api.alpaca.markets` stehen
(nicht `api.alpaca.markets`). Das ist Absicht — Schutz gegen Echtgeld.

**„Missing required environment variable: ALPACA_API_KEY":**
`.env` ist leer oder die Datei wurde nicht gefunden. Prüfe `cat .env` im
Repo-Verzeichnis.

**„No bars in DB" beim Backtest:**
Erst Kursdaten holen, dann backtest mit `--fetch` laufen lassen (siehe oben).

**Bot crasht silent:**
```bash
docker compose logs --tail=200 amms
```
Zeigt die letzten 200 Zeilen Output. Da steht die Fehlermeldung.

**Telegram bekommt nichts:**
- `.env` prüfen (Token + Chat-ID korrekt?)
- Hast du deinem Bot in Telegram mal `/start` geschickt? Sonst kann er
  dir nichts senden.

---

## 12. Sicherheits-Hinweise

- **`.env` niemals committen** (steht im `.gitignore`, gut so).
- **SSH-Passwort-Login deaktivieren** (Hetzner macht das per Default
  wenn du nur Schlüssel hinzufügst).
- **Server-Updates**: einmal im Monat einloggen und `apt update &&
  apt upgrade -y` laufen lassen.
- Der Bot ist **paper-only**. Selbst wenn jemand auf den Server kommt,
  kann er kein echtes Geld bewegen — Alpaca trennt Paper- und Live-Accounts
  vollständig.

---

## Wenn du fertig bist: kurzer Sanity-Check

✅ `docker compose ps` zeigt Container `amms` als „Up"
✅ `docker compose logs --tail=20` zeigt Zeilen wie „scheduler starting"
✅ Telegram bekommt „amms started" (falls aktiviert)
✅ `amms status` zeigt deine Paper-Equity

Wenn alle vier ✅ sind, läuft alles. Jetzt lehn dich zurück und beobachte
den Bot ein paar Tage, bevor du weitere Features anbaust.
