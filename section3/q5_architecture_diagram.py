"""
Section 3 — Q5: System Architecture Diagram
AI-Powered Ad Personalization Engine for a Major E-Commerce Client

This script generates a text-based architecture diagram and saves it as a structured document.
"""

ARCHITECTURE = """
================================================================================
  SYSTEM ARCHITECTURE: AI-Powered Ad Personalization Engine
  Client: Major E-Commerce Platform
================================================================================

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        DATA INGESTION LAYER                        │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
  │  │  User Browse  │  │  Purchase    │  │  CRM / CDP   │             │
  │  │  Clickstream  │  │  History     │  │  Segments    │             │
  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
  │         │                  │                  │                     │
  │         └──────────────────┼──────────────────┘                    │
  │                            ▼                                       │
  │                  ┌──────────────────┐                              │
  │                  │  Apache Kafka /  │                              │
  │                  │  Event Stream    │                              │
  │                  └────────┬─────────┘                              │
  └───────────────────────────┼─────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     DATA PROCESSING LAYER                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ┌──────────────────┐  ┌──────────────────┐                       │
  │  │  Feature Store   │  │  User Profile    │                       │
  │  │  (Redis/Feast)   │  │  Builder         │                       │
  │  │  - demographics  │  │  - real-time     │                       │
  │  │  - preferences   │  │    behavior      │                       │
  │  │  - purchase freq │  │  - intent score  │                       │
  │  └────────┬─────────┘  └────────┬─────────┘                       │
  │           └────────────┬────────┘                                  │
  │                        ▼                                           │
  │              ┌──────────────────┐                                  │
  │              │  Data Warehouse  │                                  │
  │              │  (BigQuery /     │                                  │
  │              │   Snowflake)     │                                  │
  │              └────────┬─────────┘                                  │
  └───────────────────────┼─────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      AI / ML ENGINE LAYER                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
  │  │  Audience        │  │  LLM Creative    │  │  Multi-Armed   │   │
  │  │  Segmentation    │  │  Generator       │  │  Bandit (MAB)  │   │
  │  │  Model           │  │  (GPT-4 / Claude)│  │  Optimizer     │   │
  │  │                  │  │                  │  │                │   │
  │  │  - K-means /     │  │  - Ad headlines  │  │  - A/B testing │   │
  │  │    lookalike     │  │  - Body copy     │  │  - Auto-select │   │
  │  │  - Propensity    │  │  - CTAs per      │  │    winning ad  │   │
  │  │    scoring       │  │    segment       │  │  - Real-time   │   │
  │  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘   │
  │           └────────────────┬────┘─────────────────────┘            │
  │                            ▼                                       │
  │                  ┌──────────────────┐                              │
  │                  │  Personalization │                              │
  │                  │  Decision Engine │                              │
  │                  │  (Which ad for   │                              │
  │                  │   which user?)   │                              │
  │                  └────────┬─────────┘                              │
  └───────────────────────────┼─────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       AD SERVING LAYER                             │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
  │  │  Ad Rendering    │  │  CDN / Edge      │  │  Ad Delivery   │   │
  │  │  Service         │  │  Cache           │  │  API           │   │
  │  │  (Dynamic HTML/  │  │  (CloudFront /   │  │  (REST / gRPC) │   │
  │  │   image gen)     │  │   Fastly)        │  │  < 50ms SLA    │   │
  │  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘   │
  │           └────────────────┬────┘─────────────────────┘            │
  │                            ▼                                       │
  │  ┌────────────────────────────────────────────────────────────┐    │
  │  │              DELIVERY CHANNELS                             │    │
  │  │  Website  │  Mobile App  │  Email  │  Social  │  Display  │    │
  │  └────────────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   MONITORING & FEEDBACK LOOP                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
  │  │  Analytics    │  │  Brand Safety│  │  Performance         │     │
  │  │  Dashboard    │  │  Monitor     │  │  Tracker             │     │
  │  │  (Grafana /   │  │  (Real-time  │  │  - CTR, CVR, ROAS   │     │
  │  │   Looker)     │  │   content    │  │  - Per-segment perf  │     │
  │  │              │  │   scanning)  │  │  - Revenue impact    │     │
  │  └──────────────┘  └──────────────┘  └──────────────────────┘     │
  │                                                                     │
  │  ┌──────────────────────────────────────────────────────────────┐  │
  │  │  FEEDBACK LOOP: Performance data feeds back into ML models  │  │
  │  │  for continuous optimization (retrain weekly)                │  │
  │  └──────────────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────┘


================================================================================
  KEY DESIGN DECISIONS
================================================================================

1. REAL-TIME vs BATCH:
   - User profiling: Real-time (Kafka streams) for current session behavior
   - Model retraining: Batch (weekly) using BigQuery/Snowflake warehouse
   - Ad serving: Real-time with < 50ms latency SLA via edge caching

2. LLM INTEGRATION:
   - GPT-4 / Claude generates ad copy variations per audience segment
   - Pre-generated at campaign launch, not at serving time (latency)
   - Cached in CDN, rotated based on MAB optimization results

3. PERSONALIZATION STRATEGY:
   - Level 1: Segment-based (demographics + purchase history)
   - Level 2: Behavioral (real-time browsing patterns)
   - Level 3: Contextual (time of day, device, location)

4. BRAND SAFETY:
   - Pre-serve content scanning for generated ad copy
   - Real-time monitoring of ad placements and contexts
   - Automatic pull-back on safety score drops below threshold

5. SCALABILITY:
   - Kafka handles 100K+ events/second for real-time ingestion
   - Redis feature store serves user profiles at < 5ms latency
   - CDN edge caching ensures global ad delivery at < 50ms

6. PRIVACY & COMPLIANCE:
   - GDPR/CCPA compliant — user consent management layer
   - PII anonymization before ML processing
   - Data retention policies enforced at warehouse level

================================================================================
  TECH STACK SUMMARY
================================================================================

  Component              │  Technology
  ───────────────────────┼──────────────────────────
  Event Streaming        │  Apache Kafka
  Feature Store          │  Redis + Feast
  Data Warehouse         │  BigQuery / Snowflake
  ML Models              │  Python + scikit-learn
  LLM (Copy Generation)  │  GPT-4 / Claude API
  A/B Optimization       │  Multi-Armed Bandit (MAB)
  Ad Serving API         │  FastAPI / gRPC
  Edge Cache             │  CloudFront / Fastly
  Monitoring             │  Grafana + Prometheus
  Brand Safety           │  Custom AI scanner
  Orchestration          │  Kubernetes (EKS/GKE)
  CI/CD                  │  GitHub Actions
"""


if __name__ == "__main__":
    print(ARCHITECTURE)

    # Save to file
    output_path = "section3/q5_architecture_diagram.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ARCHITECTURE)

    print(f"\nArchitecture saved to: {output_path}")
