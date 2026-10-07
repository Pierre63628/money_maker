# Déploiement sur VPS

Cible : Debian 12 ou Ubuntu 22.04/24.04, 2 vCPU, 2 Go de RAM, 40 Go de disque.
La boucle consomme environ 200 Mo et dort entre les barres ; le disque est la seule ressource qui
compte vraiment, à cause des données de marché (environ 2 Go).

## 1. Depuis le poste de travail

```bash
bash deploy/sync.sh root@IP_DU_VPS
```

Envoie le code **sans** `var/` ni les secrets. À refaire après chaque modification, avec `--restart`
pour redémarrer les services.

## 2. Sur le VPS, une seule fois

```bash
bash /opt/money_maker/deploy/install.sh
```

Installe Python, les dépendances, **chrony** (indispensable : Binance rejette les requêtes signées dont
l'horodatage dérive), l'utilisateur système `quant`, la rotation des journaux et les services systemd.

## 3. Les identifiants, à la main

```bash
tee /opt/money_maker/var/secrets/binance_testnet.json >/dev/null <<'JSON'
{"api_key": "...", "api_secret": "..."}
JSON
sudo chmod 600 /opt/money_maker/var/secrets/binance_testnet.json
```

Ils ne transitent jamais par git ni par `sync.sh`. L'adaptateur refuse de démarrer si le fichier est
lisible par d'autres.

## 4. Les données

```bash
bash /opt/money_maker/deploy/bootstrap_data.sh
```

Environ 2 Go, une à deux heures. Le téléchargement direct depuis le VPS est plus rapide que la copie
depuis le poste de travail, et les données sont reproductibles à l'identique.

## 5. Vérifier puis démarrer

```bash
/opt/money_maker/.venv/bin/python /opt/money_maker/scripts/testnet_check.py
sudo systemctl enable --now quant-testnet quant-paper
journalctl -u quant-testnet -f
```

`quant-testnet` envoie de vrais ordres sur le testnet ; `quant-paper` simule les exécutions sur données
réelles. Les deux tournent en parallèle : le premier teste la plomberie, le second mesure les coûts.

## Exploitation courante

| Besoin | Commande |
|---|---|
| État du compte et des positions | `scripts/testnet_status.py` |
| Santé des boucles | `scripts/healthcheck.py` |
| Rapport d'écart modèle/réel | `scripts/paper_report.py var/paper/<id>` |
| Tout solder en urgence | `scripts/testnet_flatten.py` |
| Arrêt immédiat | `sudo systemctl stop quant-testnet` |
| Kill switch sans arrêter le service | `touch var/paper/<id>/KILL` |

### Surveillance automatique

```bash
crontab -e
*/30 * * * * /opt/money_maker/.venv/bin/python /opt/money_maker/scripts/healthcheck.py || \
    echo "boucle en panne" | mail -s "money_maker" votre@email
```

Le danger n'est pas le crash — systemd redémarre — mais la boucle qui tourne en sautant chaque barre
en silence. C'est précisément ce que `healthcheck.py` détecte.

## Limites connues

- **Le fuseau horaire doit rester UTC** (`timedatectl set-timezone UTC`) : toute la logique de barres
  l'utilise.
- **Les données doivent être rafraîchies** : la boucle complète l'historique par l'API REST, mais le
  store doit être mis à jour périodiquement (`bootstrap_data.sh` est idempotent).
- **Le broker mainnet est désactivé** dans le code et le restera jusqu'à une décision explicite.
