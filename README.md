```bash
python report.py --assert-table
```

```bash
python -m pip install -r requirements.txt
hf auth login
python run.py --model llama --output reruns/llama
python run.py --model gemma --output reruns/gemma
python run.py --model olmo --output reruns/olmo
python run.py --model qwen --output reruns/qwen
python report.py --results reruns --assert-table
```
