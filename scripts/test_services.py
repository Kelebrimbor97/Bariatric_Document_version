import requests

def main():
    qdrant = requests.get("http://localhost:6333")
    print("Qdrant:", qdrant.status_code)

    models = requests.get("http://localhost:8000/v1/models")
    print("vLLM:", models.status_code)
    try:
        print(models.json())
    except Exception:
        print(models.text)

if __name__ == "__main__":
    main()