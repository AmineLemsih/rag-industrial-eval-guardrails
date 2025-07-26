# Carte du modèle – RAG Industrial Eval Guardrails

## Aperçu

Cette application met en œuvre un pipeline de **Retrieval‑Augmented Generation** (RAG) conçu pour un usage industriel.  Elle combine une recherche hybride (BM25 + recherche vectorielle) basée sur PostgreSQL et l’extension **pgvector**, un ré‑ordonnancement par cross‑encoder (`BAAI/bge-reranker-large`) et un grand modèle de langage configurable.  Les utilisateurs peuvent choisir entre un modèle **OpenAI** (`gpt-4.1-mini`), un modèle **Bedrock Claude 3.5 Sonnet** ou un modèle local via **Ollama**.  Des garde‑fous détectent les informations personnelles (PII) grâce à des expressions régulières et Presidio【49615885576500†L245-L287】, interdisent certaines catégories de questions et valident que les citations référencent réellement le contexte retourné.

## Données d’entraînement

L’application ne s’entraîne pas elle‑même : elle utilise des modèles pré‑entraînés fournis par OpenAI, Amazon ou des modèles open source.  Le corpus récupéré est constitué de documents PDF/HTML internes (procédures, manuels, guides) générés synthétiquement par le script `generate_synthetic_corpus.py`.  Ces documents sont découpés en passages et indexés dans PostgreSQL via l’extension **pgvector**, qui permet de stocker des vecteurs et de faire des recherches de similarité【805684055162342†L260-L266】.

## Architecture

* **Ingestion :** extraction de texte à partir de PDF et HTML, découpage en passages (chunks) configurables, génération d’embeddings (OpenAI ou Sentence‑Transformers), insertion dans la base de données et calcul des champs BM25 (`tsvector`).
* **Recherche :** un **retriever hybride** combine la recherche par mot‑clé (BM25) et la recherche vectorielle.  L’article de Medium cite que la recherche hybride combine la correspondance de mots clés et la recherche sémantique【516269420985915†L47-L50】.  Les résultats sont fusionnés selon des pondérations configurables et envoyés à un ré‑ordonneur.
* **Rerankeur :** un **cross‑encoder** comme `bge-reranker-large` permet de classer précisément les passages pertinents.  Les modèles cross‑encoders prennent la question et le document en entrée et produisent un score de similarité utilisé pour ré‑ordonner les k premiers résultats【66459006035083†L123-L126】【66459006035083†L197-L203】.
* **Génération :** un LLM répond à la question en utilisant exclusivement le contexte récupéré et en citant les passages correspondants.  Si aucune réponse n’est trouvée, il indique ne pas savoir.  Les citations sont validées pour garantir qu’elles correspondent au contexte renvoyé.
* **Guardrails :** détection et masquage de PII, classification des questions interdites (politique, violence, etc.), validation des citations et limitation du débit (token bucket).  La détection PII combine des regex et le pipeline de **Presidio**【49615885576500†L245-L287】【49615885576500†L303-L325】.
* **Observabilité :** utilisation de `prometheus-client` pour exporter les métriques (compteurs, histogrammes de latence) et `structlog` pour la journalisation structurée.  Un script de benchmark mesure les latences p50/p95 et le coût estimé.

## Évaluation

L’évaluation se base sur la bibliothèque **RAGAS**, qui propose plusieurs métriques :

* **faithfulness** : cohérence factuelle entre la réponse et le contexte【828334198447683†L86-L96】 ;
* **answer_relevance** : pertinence et exhaustivité de la réponse par rapport à la question【765050220413865†L88-L125】 ;
* **context_precision** : capacité de la recherche à remonter les passages pertinents en tête【771662367627463†L88-L99】 ;
* **context_recall** : proportion des éléments de vérité couverts par les contextes récupérés【393318849557330†L88-L97】.

Les scripts d’évaluation et de benchmark (`evaluate_ragas.py` et `bench_latency.py`) calculent ces métriques et vérifient que les seuils définis (faithfulness ≥ 0,80, answer_relevance ≥ 0,85) sont respectés.  En cas d’échec, l’intégration continue échoue.

## Limitations et biais potentiels

* Les modèles utilisés peuvent refléter des biais présents dans leurs données d’entraînement.  Il convient de surveiller régulièrement les réponses et d’ajuster la classification des questions interdites.
* La détection PII simplifiée peut ne pas couvrir tous les cas.  Pour des données sensibles, privilégier l’intégration complète de Presidio ou d’autres solutions professionnelles.
* Le corpus de test est synthétique et limité; il ne reflète pas la complexité d’un environnement de production.  Des tests supplémentaires avec des documents réels sont recommandés.

## Usage prévu

Ce modèle est conçu pour servir de base à un système RAG d’entreprise avec un budget limité.  Il convient à des environnements maîtrisés où l’on souhaite garder la maîtrise de ses données (pas d’envoi dans le cloud par défaut) et mesurer la qualité des réponses.  Les utilisateurs doivent compléter la configuration (`.env`) avec les clés API et ajuster les paramètres selon leurs besoins (poids du retriever, choix du modèle, etc.).