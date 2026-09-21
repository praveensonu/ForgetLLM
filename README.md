# ForgetLLM: A Practical Implementation of LLM Unlearning

We present ForgetLLM, an open source interactive system for end-to-end Large Language Model (LLM) unlearning. ForgetLLM provides a unified interface for established unlearning methods, seed guided data-selection approaches, and evaluation benchmarks into a single workflow, reducing the engineering effort required to configure and compare unlearning pipelines. The system is designed for both model holders evaluating practical unlearning configurations and researchers for rapid testing.

## Run with Docker

From the repository root, build the application image:

```bash
docker build -t forgetllm-unlearning .
```

Run it with GPU access and persistent storage:

```bash
docker run --rm --name forgetllm-unlearning \
  --gpus all --shm-size=2g \
  -p 5174:8080 \
  -v forgetllm-outputs:/app/outputs \
  -v forgetllm-datasets:/app/uploaded_datasets \
  -v forgetllm-hf-cache:/app/hf-cache \
  forgetllm-unlearning
```

Open `http://localhost:5174`. The API documentation is available at `http://localhost:5174/api/docs`.

For UI/API-only use on a host without a GPU, omit `--gpus all`. For gated Hugging Face models or datasets, export `HF_TOKEN` and add `-e HF_TOKEN` before the image name.

Press `Ctrl+C` to stop the application.

## Run in Development

From the repository root, start the backend and frontend together:

```bash
./start.sh
```

Open the frontend at `http://localhost:5173`. The backend runs at `http://localhost:8000`.

Press `Ctrl+C` to stop both services.

To override the default ports:

```bash
BACKEND_PORT=8080 FRONTEND_PORT=5174 ./start.sh
```
