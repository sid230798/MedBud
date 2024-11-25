// APIConfig.js

const API_BASE_URL = "http://127.0.0.1:8000";

const API_ENDPOINTS = {
    SIMPLE_CHAT: "/v1/chat/simple",
    OPENAI_CHAT: "/v1/chat/openai",
    RAG_CHAT: "/v1/chat/openai/rag",
};

const MODELS_AVAILABLE = {
    GPT_3_5: "gpt-3.5-turbo-instruct",
    DA_VINCI: "text-davinci-003",
};

export {API_BASE_URL, API_ENDPOINTS, MODELS_AVAILABLE};
