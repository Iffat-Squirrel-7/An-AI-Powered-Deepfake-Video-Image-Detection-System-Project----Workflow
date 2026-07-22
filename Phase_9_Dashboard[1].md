# Phase 9: Dashboard & Monitoring

## Objectives
- Provide analytics on system usage and monitor model health over time.

## Dashboard Metrics to Track
- Total scans performed (daily/weekly/monthly)
- Real vs Fake detection ratio over time
- Average confidence score distribution
- Most common file types/sources analyzed
- API latency and error rate
- Model version currently in production

## Tech Stack Options
| Purpose | Tool |
|---|---|
| Application-level dashboard (business metrics) | Custom React dashboard + Chart.js/Recharts, or Streamlit |
| Infrastructure monitoring | Prometheus + Grafana |
| Logging | ELK Stack (Elasticsearch, Logstash, Kibana) or simple structured JSON logs |

## Sample Database Schema for Logging Results
```sql
CREATE TABLE detection_logs (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255),
    media_type VARCHAR(20),
    result VARCHAR(10),
    confidence FLOAT,
    frames_analyzed INT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Model Drift Monitoring
- Periodically re-evaluate the production model against a held-out "fresh" sample of recent real-world content.
- Track confidence score distribution drift — if average confidence starts dropping across many scans, it may signal new/unseen manipulation techniques emerging.
- Set up alerts (e.g., Slack/email webhook) if error rate or latency crosses a threshold.

## Sample Analytics Query
```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/deepfake_db")
df = pd.read_sql("SELECT * FROM detection_logs", engine)

daily_counts = df.groupby([df['created_at'].dt.date, 'result']).size().unstack(fill_value=0)
print(daily_counts)
```

## Deliverables
- Dashboard UI (`src/dashboard/`)
- PostgreSQL logging schema and ETL script
- Model drift monitoring script (`src/monitoring/drift_check.py`)
