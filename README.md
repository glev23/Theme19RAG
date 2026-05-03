# Тема 19 — RAG (Retrieval-Augmented Generation)

Домен: информационная безопасность (корпус PDF/текст).  
Стек: **LangChain**, **FAISS**, эмбеддинги [`sergeyzh/BERTA`](https://huggingface.co/sergeyzh/BERTA), LLM [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (локально, CPU).

## Запуск

Требуется Python 3.10+. В каталоге проекта:

```bash
cd Тема19_rag
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Активация на Linux/macOS: `source venv/bin/activate`. В Windows **cmd**: `venv\Scripts\activate.bat`.

Корпус (не менее 20 файлов `.pdf`, `.txt`, `.md`) — в `starter_pack/docs/corpus/` (см. `starter_pack/docs/corpus/README.md`).

Затем в среде Jupyter / VS Code для файла `Тема19_rag.ipynb` выполняется **Run All**. При первом запуске загружаются веса с Hugging Face (нужен доступ в интернет).

## Артефакты

После прогона в `artifacts/`: `kpi_report.csv`, `results.csv`, `chunks.csv`, `documents.csv`, `trace.jsonl`, `fig_01.png`, `architecture.mmd`, `architecture.svg`, `input_profile.json`, `export_summary.txt`.

## Примечание по CPU

Индексация и генерация на CPU могут занимать заметное время; при нехватке памяти — уменьшить в `CONFIG` значения `chunk_size` или число строк в `starter_pack/tests/questions.csv` для тестового прогона.
