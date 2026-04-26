# Zenith Learn - Machine Learning Engine (zenith-ml)

## La formation qui crée des entrepreneurs, pas des spectateurs

La vision de **Zenith Learn** est de révolutionner l'apprentissage en ligne. Fini les plateformes qui vous gardent captifs avec des notifications incessantes et des séries (streaks) culpabilisantes. Zenith Learn est conçu selon la pédagogie **Skills2Job** : l'objectif n'est pas de consommer du contenu, mais de passer à l'action. 

Notre application privilégie la complétion réelle et le passage à l'acte entrepreneurial. Nos algorithmes (sans biais et expliquables) identifient votre profil (Explorateur, Entrepreneur actif) et vous recommandent exactement ce dont vous avez besoin pour bâtir votre projet, sans aucune rétention artificielle.

Ce micro-service expose précisément les recommandations d'apprentissage éthiques en priorisant l'impact plutôt que l'engagement pur.

## Comment lancer le Moteur de Recommandation ML

Le moteur utilise **Python 3.12** et **FastAPI**. Il est équipé d'un algorithme de filtrage collaboratif hybride.

1. Placez-vous dans le répertoire du projet ML :
   ```bash
   cd zenith-ml
   ```
2. Construisez l'image Docker locale contenant le modèle et les données brutes :
   ```bash
   docker build -t zenith-ml .
   ```
3. Lancez le conteneur en arrière-plan :
   ```bash
   docker run -d -p 8000:8000 --name zenith_ml zenith-ml
   ```
4. L'API ML sera disponible sur `http://localhost:8000` (ex: `/health` ou `/api/v1/recommendations/{uuid}`).

*Alternative (sans docker) : Créer un environnement virtuel `venv`, faire un `pip install -r requirements.txt` puis `python main.py`.*

## Utilisateurs de test (Générés au startup du backend)

Le modèle utilise les identifiants d'apprenants (UUIDs) pour formuler ses prédictions. Sur votre interface (Frontend), connectez-vous avec ces identifiants pour déclencher les recommandations :

- **Alice (Développement)** : `alice.dev@example.com`
- **Bob (Marketing)** : `bob.market@example.com`
- **Charlie (Finance)** : `charlie.fin@example.com`
- **Diana (Leadership)** : `diana.lead@example.com`

**Mot de passe utilisé pour tous les comptes** : `password123`
