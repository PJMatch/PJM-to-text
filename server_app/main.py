from fastapi import FastAPI
import uvicorn

app = FastAPI()

#Get processing status
@app.get("/status")
async def get_status():
    return{
        "health": "ok",
        "processed": 2,
        "total": 500
    }



def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()