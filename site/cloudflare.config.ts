import { defineConfig } from "cf/config";

export default defineConfig({
  worker: {
    "name": "medical-rag-answers",
    "compatibilityDate": "2026-11-13",
    "observability": {
      "enabled": true
    }
  }
});
