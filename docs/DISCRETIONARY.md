# Livre directionnel : paris quotidiens fondés sur l'actualité

Créé le 2026-10-06 à la demande du propriétaire. Une fois par jour, Claude lit l'actualité, écrit une
décision (des paris long ou short), et le code l'exécute sur un **compte paper** aux prix réels du
mainnet Binance USDT-M. Aucun argent réel, aucune clé, aucun ordre réel.

Ce n'est **pas** une stratégie au sens du protocole de recherche : elle n'est pas backtestable (les
connaissances du modèle contaminent tout test historique). La seule preuve possible est le forward,
jugé par la porte ci-dessous, fixée avant le premier pari.

## Pourquoi paper et non testnet

Mesuré le 2026-10-06 : le compte testnet porte déjà les 16 positions de la boucle crowding (95 % de
brut). Une boucle qui y verrait des positions étrangères les solderait, et les deux PnL seraient
mélangés. Par ailleurs 5 des 30 perpétuels les plus liquides du mainnet n'existent pas sur le testnet,
et un symbole y cotait à 22 fois son prix réel. Un pari sur l'actualité se juge aux prix réels.

## Fonctionnement

| Étape | Commande | Ce qui se passe |
|---|---|---|
| Brief | `scripts/discretionary.py brief` | Rejoue les sorties depuis la dernière session, marque le livre, affiche paris ouverts, limites et marché autorisé |
| Décision | fichier JSON dans `var/discretionary/directional-v1/drafts/` | Écrite par Claude après recherche web ; format dans `src/quant/discretionary/calls.py` |
| Essai | `execute <fichier> --dry-run` | Rejets et verdict du moteur de risque, rien n'est engagé |
| Exécution | `execute <fichier>` | Fige la décision du jour (hash), ferme et ouvre aux cotations réelles |
| Score | `scripts/discretionary.py score` | Résultat, calibration, test contre le hasard, porte |

La procédure de session est dans `.claude/skills/trading-session/SKILL.md` (commande `/trading-session`).

## Limites, dans le code et hors de portée de la décision

| Limite | Valeur |
|---|---|
| Univers | 40 perpétuels USDT les plus liquides, au moins 50 M USDT de volume sur 24 h (crypto, actions et matières premières tokenisées) |
| Par pari | poids ≤ 10 %, poids × stop ≤ 1 % de l'équité, stop obligatoire entre 1 % et 15 %, horizon ≤ 14 jours |
| Livre | 8 paris ouverts, 4 nouveaux par jour, brut ≤ 60 %, net ≤ 40 %, cluster corrélé ≤ 30 % |
| Moteur de risque | le même que les boucles : plafond de gap à 6 σ, volume, participation |
| Arrêts | perte du jour −3 % : plus de nouveau risque ; drawdown −15 % : tout est soldé, réarmement manuel par le propriétaire |
| Discipline | une décision par jour UTC, immuable une fois exécutée ; refusée si elle a plus de 3 heures ; **au moins un pari par jour** (règle du propriétaire, 2026-10-06) |

Une décision qui enfreint une limite est rejetée en entier, jamais rognée en silence. La décision est
traitée comme une entrée non fiable : son auteur lit le web ouvert.

Le pari quotidien obligatoire a une conséquence assumée : certains jours le meilleur pari disponible
vaut à peine mieux qu'un tirage au sort. Il est alors pris petit et avec une confiance annoncée proche
de 50 %, et il compte dans le score comme les autres. La règle ne passe pas avant les limites : kill
switch, perte du jour ou refus du moteur de risque peuvent laisser un jour sans pari.

## Sorties entre deux sessions

Rien ne tourne entre les sessions. Les stops, objectifs et échéances sont reconstruits à la session
suivante en rejouant les barres de 5 minutes : stop exécuté à son prix, ou à l'ouverture de la barre si
le marché a traversé le stop ; quand une barre touche le stop et l'objectif, le stop gagne ; le funding
est réglé aux taux réels. Les sorties paient spread, slippage et frais taker, plus 5 bps sur un stop.

## Le test contre le hasard

Un PnL positif ne prouve rien : vingt shorts dans un mois baissier ressemblent à du talent. Chaque pari
a donc un **miroir** : le sens opposé, depuis le même prix, avec les mêmes stop, objectif, échéance et
coûts. C'est ce qu'aurait gagné un tirage à pile ou face au même moment. Si le sens des paris ne porte
aucune information, pari et miroir sont interchangeables : le test échange les deux au hasard et
mesure combien de tirages font au moins aussi bien que le livre (p-value).

## Porte (fixée le 2026-10-06, à ne pas ajuster)

100 paris clos et comparés, 60 jours de suivi, p-value ≤ 0,05, rendement net positif, score de Brier
< 0,25, drawdown moins profond que −15 %. La franchir rend le livre éligible à une discussion sur le
réel avec le propriétaire ; elle n'autorise rien. Avec 8 emplacements et des paris tenus quelques jours,
le rythme réaliste est d'une quarantaine de paris par mois : compter environ trois mois.

## Réserves

- Les cinq premières minutes après une entrée ne sont pas rejouées (la barre d'entrée contient des prix
  antérieurs) : un stop touché puis effacé dans cet intervalle n'est pas vu.
- Le kill switch de drawdown n'est évalué qu'en session. Entre deux sessions, la perte est bornée par la
  somme des stops (8 % au pire, hors gaps).
- La p-value suppose des paris indépendants. Des paris du même jour dans le même sens ne le sont pas :
  elle flatte un livre unidirectionnel. La part de longs et le rendement de BTC sont affichés à côté.
- Les perpétuels d'actions et de matières premières peuvent ouvrir en gap quand leur marché de
  référence rouvre ; le plafond de gap du moteur est calibré sur la volatilité quotidienne et le
  sous-estime pour eux.
- Les sorties sur objectif sont facturées comme des ordres taker : légèrement pessimiste.

## Journal

- **2026-10-06, session 1.** Première décision à 08:33 UTC : aucun pari (déblocage HYPE écarté, sources
  contradictoires sur la date et sur l'intention de l'acheteur). Le propriétaire a ensuite fixé la règle
  d'un pari par jour : la décision vide a été remplacée à 08:42 UTC, l'originale est conservée dans
  `decisions/2026-10-06.no-trade.json`. **Pari : short XAUUSDT, 8 %, entrée 4 153,49, stop 4 236,55
  (2 %), objectif 4 028,88 (3 %), échéance le 9 à 08:42 UTC, confiance 0,53.** Thèse : six semaines de
  baisse, rebond sur l'emploi faible aussitôt effacé, positionnement non purgé, minutes d'une hausse
  votée à l'unanimité le 7. Chaîne d'ordres vérifiée au préalable sur un livre jetable, hors historique.
