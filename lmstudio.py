from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    temperature=0.1
)

print(response.choices[0].message.content)