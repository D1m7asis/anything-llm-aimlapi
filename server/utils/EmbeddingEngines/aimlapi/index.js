const { toChunks, maximumChunkLength } = require("../../helpers");

class AimlApiEmbedder {
  constructor() {
    if (!process.env.AIML_API_KEY) throw new Error("No AI/ML API key was set.");
    const { OpenAI: OpenAIApi } = require("openai");
    this.openai = new OpenAIApi({
      apiKey: process.env.AIML_API_KEY,
      baseURL: "https://api.aimlapi.com/v1",
    });
    this.model = process.env.EMBEDDING_MODEL_PREF || "text-embedding-ada-002";
    this.maxConcurrentChunks = 500;
    this.embeddingMaxChunkLength = maximumChunkLength();
  }

  async embedTextInput(textInput) {
    const result = await this.embedChunks(Array.isArray(textInput) ? textInput : [textInput]);
    return result?.[0] || [];
  }

  async embedChunks(textChunks = []) {
    const embeddingRequests = [];
    for (const chunk of toChunks(textChunks, this.maxConcurrentChunks)) {
      embeddingRequests.push(
        new Promise((resolve) => {
          this.openai.embeddings
            .create({ model: this.model, input: chunk })
            .then((result) => resolve({ data: result?.data, error: null }))
            .catch((e) => {
              e.type = e?.response?.data?.error?.code || e?.response?.status || "failed_to_embed";
              e.message = e?.response?.data?.error?.message || e.message;
              resolve({ data: [], error: e });
            });
        })
      );
    }

    const { data = [], error = null } = await Promise.all(embeddingRequests).then((results) => {
      const errors = results.filter((res) => !!res.error).map((res) => res.error);
      if (errors.length > 0) {
        const unique = new Set();
        errors.forEach((err) => unique.add(`[${err.type}]: ${err.message}`));
        return { data: [], error: Array.from(unique).join(", ") };
      }
      return { data: results.map((r) => r.data || []).flat(), error: null };
    });

    if (error) throw new Error(`AimlApi Failed to embed: ${error}`);
    return data.length > 0 && data.every((d) => d.hasOwnProperty("embedding"))
      ? data.map((d) => d.embedding)
      : null;
  }
}

module.exports = { AimlApiEmbedder };
