# Moondream Deployment Container
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    pillow>=10.0.0 \
    accelerate>=0.25.0

# Copy model files
WORKDIR /app
COPY moondream_fp16 /app/moondream_fp16
COPY deploy_moondream.py /app/

# Expose API port (if using FastAPI/Flask)
EXPOSE 8000

# Run inference
CMD ["python", "deploy_moondream.py"]
