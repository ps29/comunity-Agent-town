from openai import AsyncOpenAI


class LlamaCppClient:
    def __init__(self, base_url: str, model: str, api_key: str = "llama.cpp"):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 300,
        response_format: dict | None = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
