# FastAPI backend for AI Q&A Chat (streaming)

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- POST /chat/stream streams a plain-text response in chunks.
- On Android emulator, the Flutter app calls http://10.0.2.2:8000.
- On iOS simulator/desktop/web, it calls http://127.0.0.1:8000.

## Replace fake LLM
- Swap `fake_llm_stream` with your actual generator producing token strings.


