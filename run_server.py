"""Start VisionVerse AI (API + frontend) on one origin."""

import uvicorn

from backend.config.settings import settings

if __name__ == "__main__":
    print("VisionVerse AI  ->  http://%s:%d" % (settings.HOST, settings.PORT))
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=False)
