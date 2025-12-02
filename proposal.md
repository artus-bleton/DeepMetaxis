# **DEEP Metaxis : vers une architecture d'apprentissage oscillant entre curiosité et prédiction**

------

## **Résumé**

**DEEP Metaxis** explore un nouvel **axe de recherche** en intelligence artificielle : concevoir des agents capables d’auto-réguler leur motivation et leur apprentissage sans supervision externe.
 Plutôt qu’un modèle achevé, ce cadre cherche à établir les bases d’un **métabolisme cognitif artificiel**, où la curiosité et la prédiction s’équilibrent en un cycle veille–sommeil inspiré du vivant.
 Il s’agit d’une tentative de dépasser les limites actuelles des approches centrées sur la récompense et la planification explicite.

------

## **1. Contexte et motivation**

Les architectures actuelles d’apprentissage profond — qu’elles reposent sur le **renforcement**, la **modélisation du monde** ou le **predictive coding** — restent fondamentalement **dépendantes d’objectifs extrinsèques** et d’un environnement qui “guide” leur progression.
 Elles ne parviennent pas à résoudre le problème central de l’**autonomie motivationnelle** : comment un agent peut-il *vouloir apprendre*, *quoi apprendre*, et *quand s’arrêter* sans supervision explicite ni signal de tâche ?

La plupart des mécanismes de curiosité intrinsèque actuels (ICM, RND, etc.) échouent à maintenir une motivation durable : ils amplifient la surprise sans la réguler, menant soit à une exploration chaotique, soit à un effondrement de la plasticité.

**DEEP Metaxis** tente autre chose.
 Il propose une **architecture oscillatoire**, inspirée du cycle veille–sommeil biologique, où la curiosité et la prédiction alternent comme deux forces homéostatiques.
 L’agent explore le monde lorsqu’il manque de compréhension, puis consolide ses représentations lorsque la surprise devient excessive.

Ce projet ne vise pas la performance sur une tâche donnée, mais la **formulation d’un principe général d’équilibre cognitif**, susceptible d’unifier curiosité, stabilité et apprentissage profond au sein d’un même cadre dynamique.

------

## **2. Notations et architecture**

### **Variables principales**

- (x_t \in \mathbb{R}^{n}) : observation brute à l’instant (t)
- (a_t \in \mathcal{A}) : action choisie à l’instant (t)
- (z_t = a_\theta(x_t) \in \mathbb{R}^{d}) : représentation latente (encodeur)
- (\hat{x}*t = d*\psi(z_t)) : reconstruction (décodeur)
- (\hat{z}*{t+1} = g*\phi(z_t, a_t)) : prédiction latente (modèle dynamique)

[
 g(z_t, a_t) = \mathrm{Aggr}\big({,g_i(p_i(z_t)),}_i, a_t\big),
 \quad
 p_i : \mathbb{R}^d \to \mathbb{R}^{d_i}
 ]

Chaque (p_i) définit un **sous-espace latent spécialisé**, (g_i) prédit une dynamique partielle, et l’agrégateur (conditionné par (a_t)) combine ces contributions pour reconstruire la dynamique globale.
 Cette structure permet un **désenchevêtrement fonctionnel** du latent (z_t) — vision, planification, proprioception, etc.

------

### **Récompenses et modules**

- (R_{\text{ext}}(t)) : récompense extrinsèque (objectif explicite)
- (r_{\text{int}}(t) = |z_{t+1} - \hat{z}_{t+1}|^2) : récompense intrinsèque (surprise latente)

Modules principaux :

- **Encodeur** : (a_\theta : x_t \mapsto z_t)
- **Décodeur** : (d_\psi : z_t \mapsto \hat{x}_t)
- **Modèle dynamique** : (g_\phi : (z_t, a_t) \mapsto \hat{z}_{t+1})
- **Politique** : (\pi_\omega(a_t | z_t)), contrôlant les actions et l’exploration

------

## **3. Phase 1 – Curiosité (veille)**

Durant la veille, l’agent interagit activement avec son environnement et met à jour sa politique selon :

[
 J_{\text{veille}}(\omega) =
 \mathbb{E}!\left[\sum_t R_{\text{ext}}(t)

- \beta \cdot |z_{t+1} - g_\phi(z_t, a_t)|^2\right]
   ]

(\beta) régule la curiosité : un **signal dopaminergique artificiel** qui augmente lorsque la récompense extrinsèque diminue, et inversement.
 Cette phase correspond à un état de **plasticité cognitive élevée** : l’agent cherche les zones du monde où son modèle échoue, et s’y aventure activement.

------

## **4. Phase 2 – Sommeil (consolidation)**

Pendant le sommeil, la politique est gelée.
 L’agent réorganise son modèle interne pour **réduire la surprise** accumulée et renforcer la cohérence de ses représentations :

[
 L_{\text{sommeil}}(\phi,\psi,\theta) =
 \mathbb{E}!\left[
 |z_{t+1} - g_\phi(z_t, a_t)|^2

- \lambda \cdot |x_t - d_\psi(a_\theta(x_t))|^2
   \right]
   ]

- Le premier terme affine la **dynamique latente**.
- Le second stabilise la **reconstruction sensorielle**.
- (\lambda) contrôle la force de régularisation.

Ce mécanisme rappelle le **replay hippocampique** du sommeil : les séquences à forte surprise sont rejouées, intégrées et abstraites, créant une mémoire latente stable.

------

## **5. Dynamique du cycle et équilibre**

L’alternance entre veille et sommeil produit une oscillation auto-régulée :

- **Veille** → expansion de l’entropie, apprentissage exploratoire.
- **Sommeil** → compression, intégration, stabilisation.

[
 \frac{\partial,\text{Curiosité}}{\partial t}
 = -\alpha \cdot \text{Prédictibilité}, \quad
 \frac{\partial,\text{Prédictibilité}}{\partial t}
 = \gamma \cdot \text{Curiosité}
 ]

Sous certaines conditions de taux ((\alpha, \gamma)), le système atteint un **point d’équilibre homéostasique**, où l’exploration et la prédiction se soutiennent mutuellement.
 Cet état n’est pas statique : il correspond à une **métastabilité cognitive**, analogue à un cycle circadien interne.

------

## **6. Lien biologique et analogie cognitive**

| Processus artificiel                        | Analogue biologique                                          | Fonction                              |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------- |
| Phase veille : maximisation de la surprise  | Dopamine, exploration, apprentissage par renforcement        | Découverte active, adaptation rapide  |
| Phase sommeil : minimisation de la surprise | Sommeil paradoxal, replay hippocampique, régulation synaptique | Consolidation, généralisation         |
| AE + (g_\phi)                               | Cortex et thalamus                                           | Modèle interne du monde sensorimoteur |

Cette analogie n’est pas un argument biologique, mais une **heuristique structurelle** : elle suggère que certaines propriétés du vivant (plasticité cyclique, régulation dopaminergique, homéostasie synaptique) pourraient être **transposées en principes computationnels**.

------

## **7. Perspectives expérimentales**

- **Environnements** : *MineRL*, *MiniGrid*, *DeepMind Control Suite*
- **Alternance temporelle** : (N) épisodes de curiosité suivis de (M) épisodes de sommeil
- **Métriques** : couverture de l’espace latent, stabilité prédictive, complexité comportementale, entropie politique
- **Comparaisons** : *Dreamer*, *ICM*, *RND*, *Active Inference*

L’enjeu n’est pas de battre des benchmarks, mais de **tester la stabilité à long terme d’un agent auto-motivé**, et d’observer si un équilibre stable entre curiosité et compréhension peut émerger.

------

## **8. Conclusion et portée scientifique**

**DEEP Metaxis** n’est pas un modèle opérationnel, mais une **tentative conceptuelle** :
 une hypothèse selon laquelle *la cognition artificielle pourrait émerger non d’un objectif fixe, mais d’une oscillation régulée entre désordre et structure.*

Ce cadre ouvre un **axe de recherche fondamental** sur la question de la **motivation intrinsèque auto-régulée**.
 En cherchant à faire coexister curiosité et prédiction dans un même métabolisme cognitif, **DEEP Metaxis** explore une idée simple mais radicale :

> **L’intelligence n’est pas un état, mais un équilibre en mouvement.**