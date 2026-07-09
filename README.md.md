# local-rag

Point it at a folder of documents (`.txt`, `.md`, `.pdf`, `.docx`), then chat with them — fully local, nothing leaves your machine.

1. Install Ollama and pull models
```
# install Ollama: https://ollama.com

# Tested on 32gb ram machine. Based on your hardware, you might need to use a different model

ollama pull llama3.1:8b        # chat model 
ollama pull nomic-embed-text   # embedding model
```

2. Install Python dependencies
```
pip install -r requirements.txt
```

3. Index your documents
```
python -m src.cli index --docs /path/to/your/docs
```

This walks the folder recursively, chunks everything, embeds it, and saves a local
Chroma DB to `./chroma_db`. Re-run this any time your documents change — it rebuilds
from scratch 

4. Chat
```
python -m src.cli chat
```

Or ask a single question without interactive loop:

```
python -m src.cli ask "your question"
```
