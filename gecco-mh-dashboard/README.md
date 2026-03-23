# GeCCo-MH Dashboard

Interactive Streamlit dashboard for monitoring distributed GeCCo runs.

## Install

```bash
pip install -r gecco-mh-dashboard/requirements.txt
```

## Run (remote host)

```bash
streamlit run gecco-mh-dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

## SSH tunnel (from local machine)

```bash
ssh -N -L 8501:127.0.0.1:8501 <user>@<remote-host>
```

Then open: <http://127.0.0.1:8501>
