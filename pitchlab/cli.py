import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="PitchLab CLI for Football Video Annotation MVP")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for server during development")

    args = parser.parse_args()

    print("==================================================")
    print("[PITCHLAB] Football Video Annotation MVP")
    print(f"Server running at: http://{args.host}:{args.port}")
    print("==================================================")

    uvicorn.run("pitchlab.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
